"""Full inference pipeline for Self-Forcing video diffusion.

Implements the complete text-to-video generation loop with:
- Per-block KV cache management for autoregressive inference
- Flow matching scheduler (mlx-arsenal)
- VAE decoding
- Frame-by-frame processing for memory efficiency

This is the main entry point: use MLXPipeline class.
"""
import os
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors import safe_open

from sforcing.config import (
    DIM, FFN_DIM, NUM_HEADS, NUM_LAYERS, FREQ_DIM, TEXT_DIM, TEXT_LEN,
    IN_DIM, OUT_DIM, PATCH_SIZE, EPS, FRAME_SEQ_LEN,
    GUIDANCE_SCALE, NUM_TRAIN_TIMESTEPS,
)
from sforcing.model import WanModel
from sforcing.t5 import T5Encoder
from sforcing.vae import WanVAE
from sforcing.scheduler import FlowMatchScheduler
from sforcing.tokenizer_bridge import HuggingfaceTokenizer


class KVCache:
    """Per-block KV cache for autoregressive inference.

    Manages key/value tensor storage and updates across denoising steps.
    Supports circular buffer eviction for local attention.
    """

    def __init__(self, batch_size, max_seq_len, num_heads, head_dim,
                 dtype=mx.float32):
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype

        self.k = mx.zeros((batch_size, max_seq_len, num_heads, head_dim), dtype=dtype)
        self.v = mx.zeros((batch_size, max_seq_len, num_heads, head_dim), dtype=dtype)
        self.global_end_index = mx.array([0], dtype=mx.int32)
        self.local_end_index = mx.array([0], dtype=mx.int32)

    def update(self, keys, values, cur_len):
        """Write new keys/values to cache and return cached views.

        Args:
            keys: New key tokens, shape (B, cur_len, N, D).
            values: New value tokens, shape (B, cur_len, N, D).
            cur_len: Current sequence length (number of new tokens).

        Returns:
            Tuple of (cached_keys, cached_values) from the cache.
        """
        start_idx = self.global_end_index[0] - self.local_end_index[0] + self.local_end_index[0]
        self.k = self.k.at[:, start_idx:start_idx + cur_len].set(keys)
        self.v = self.v.at[:, start_idx:start_idx + cur_len].set(values)
        self.global_end_index = self.global_end_index + cur_len
        self.local_end_index = self.local_end_index + cur_len

        return self.k, self.v


class MLXPipeline:
    """Complete text-to-video inference pipeline.

    Loads pre-trained weights, runs the denoising loop with KV caching,
    decodes latents through the VAE, and saves the output video.
    """

    def __init__(
        self,
        transformer_path,
        t5_path,
        vae_path,
        num_frames=21,
        height=480,
        width=832,
        guidance_scale=GUIDANCE_SCALE,
        timestep_shift=5.0,
        denoising_steps=[1000, 750, 500, 250],
        num_frame_per_block=3,
    ):
        self.device = mx.default_device()
        self.dtype = mx.float32
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.guidance_scale = guidance_scale
        self.num_frame_per_block = num_frame_per_block

        # Frame-level dimensions
        self.fh = height // 8
        self.fw = width // 8
        self.ft = num_frames // 4
        self.frame_seq_len = self.ft * self.fh * self.fw
        self.max_seq_len = 32760

        # --- Load T5 encoder ---
        print(f"Loading T5 encoder from {t5_path}...")
        self.t5 = T5Encoder(
            vocab_size=256384,
            dim=TEXT_DIM,
            dim_attn=TEXT_DIM,
            dim_ffn=10240,
            num_heads=64,
            num_layers=24,
            num_buckets=32,
            shared_pos=False,
            dropout=0.1,
        )
        self._load_weights(self.t5, t5_path)
        print("T5 encoder loaded.")

        # --- Load transformer ---
        print(f"Loading transformer from {transformer_path}...")
        self.model = WanModel(
            dim=DIM,
            ffn_dim=FFN_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            freq_dim=FREQ_DIM,
            text_dim=TEXT_DIM,
            text_len=TEXT_LEN,
            in_dim=IN_DIM,
            out_dim=OUT_DIM,
            patch_size=PATCH_SIZE,
            eps=EPS,
        )
        self._load_weights(self.model, transformer_path)
        print("Transformer loaded.")

        # --- Load VAE decoder ---
        print(f"Loading VAE from {vae_path}...")
        self.vae = WanVAE(z_dim=IN_DIM, dim=96, dim_mult=[1, 2, 4, 4])
        self.vae.load_weights(vae_path)
        print("VAE loaded.")

        # --- Setup tokenizer ---
        self.tokenizer = HuggingfaceTokenizer(
            name="google/umt5-xxl",
            seq_len=TEXT_LEN,
            clean='whitespace',
        )

        # --- Setup scheduler ---
        self.scheduler = FlowMatchScheduler(
            num_train_timesteps=NUM_TRAIN_TIMESTEPS,
            shift=timestep_shift,
            extra_one_step=True,
        )
        self.scheduler.set_timesteps(len(denoising_steps))
        self.denoising_step_list = denoising_steps

        # --- Setup KV caches ---
        self._kv_caches = None
        self._crossattn_caches = None

    def _load_weights(self, model, path):
        """Load weights from .safetensors into an MLX model."""
        params = {}
        with safe_open(path, framework="numpy") as f:
            for key in f.keys():
                params[key] = np.array(f.get_tensor(key))

        mlx_params = self._mlxify_params(params)
        weights = self._build_weight_dict(model, mlx_params)
        model.update(weights)

    def _mlxify_params(self, pytorch_params):
        """Map PyTorch parameter names to MLX names."""
        import re
        mlx_params = {}
        for key, value in pytorch_params.items():
            ml_key = key
            if ml_key.startswith("encoder."):
                ml_key = ml_key[8:]
            ml_key = re.sub(r'\.(\d+)\.', r'[\1].', ml_key)
            mlx_params[ml_key] = value
        return mlx_params

    def _build_weight_dict(self, model, params):
        """Build nested weight dict matching model structure for model.update()."""
        weight_dict = {}
        for ml_key, np_arr in params.items():
            mx_arr = mx.array(np_arr)
            self._set_nested(weight_dict, ml_key, mx_arr)
        return weight_dict

    def _set_nested(self, d, key_path, value):
        """Set a nested dict value using dot notation with bracket indices."""
        import re
        parts = re.split(r'\.(?![0-9])', key_path)
        current = d
        for i, part in enumerate(parts[:-1]):
            if '[' in part:
                base, idx = part.split('[')
                idx = int(idx.rstrip(']'))
                if base not in current:
                    current[base] = {}
                base_key = base
                if not isinstance(current[base_key], list):
                    current[base_key] = {}
                lst = current[base_key].setdefault('_list', [])
                while len(lst) <= idx:
                    lst.append({})
                current = lst[idx]
            else:
                if part not in current:
                    current[part] = {}
                current = current[part]

        last = parts[-1]
        if '[' in last:
            base, idx = last.split('[')
            idx = int(idx.rstrip(']'))
            if base not in current:
                current[base] = {}
            base_key = base
            lst = current[base_key].setdefault('_list', [])
            while len(lst) <= idx:
                lst.append({})
            current = lst[idx]
            current['_value'] = value
        else:
            current[last] = value

    def _initialize_caches(self, batch_size=1):
        """Initialize KV caches and cross-attention caches."""
        self._kv_caches = []
        self._crossattn_caches = []

        h, d = NUM_HEADS, self.model.freqs.shape[1] // NUM_HEADS

        for _ in range(NUM_LAYERS):
            kv = KVCache(batch_size, self.max_seq_len, h, d, dtype=self.dtype)
            self._kv_caches.append(kv)

            cross = {
                "k": mx.zeros((batch_size, TEXT_LEN, h, d), dtype=self.dtype),
                "v": mx.zeros((batch_size, TEXT_LEN, h, d), dtype=self.dtype),
                "is_init": False,
            }
            self._crossattn_caches.append(cross)

    def _reset_caches(self):
        """Reset all caches for a new generation."""
        if self._kv_caches is not None:
            for kv in self._kv_caches:
                kv.global_end_index = mx.array([0], dtype=mx.int32)
                kv.local_end_index = mx.array([0], dtype=mx.int32)
        if self._crossattn_caches is not None:
            for cross in self._crossattn_caches:
                cross["is_init"] = False

    def _tokenize(self, prompt):
        """Tokenize a text prompt."""
        ids = self.tokenizer(prompt, return_mask=False, add_special_tokens=True)
        return ids[0]

    def _encode_text(self, prompt):
        """Encode a text prompt through the T5 encoder."""
        ids = self._tokenize(prompt)
        ids_mx = mx.array(ids.astype(np.int32)).reshape(1, -1)
        mask_mx = mx.ones((1, TEXT_LEN), dtype=mx.int32)

        context = self.t5(ids_mx, mask=mask_mx)
        context = context * mask_mx.reshape(1, -1, 1)
        return context

    def _forward_transformer(self, latents, timestep, context):
        """Single forward pass through the transformer.

        Uses WanModel.__call__ which handles patching, unpatching.

        Args:
            latents: Input latents, shape (1, T, C, H, W).
            timestep: Timestep, shape (B,).
            context: Text context, shape (1, TEXT_LEN, TEXT_DIM).

        Returns:
            Output flow prediction, shape (1, T, OUT_DIM, H, W).
        """
        b, t, c, h, w = latents.shape

        # Convert to list of (C, T, H, W) for model
        x_list = [latents[i].transpose(2, 0, 1, 3, 4) for i in range(b)]

        grid_sizes = np.array([[self.ft, h, w]], dtype=np.int32)

        outputs = self.model(x_list, timestep, context, self.max_seq_len, grid_sizes)

        if isinstance(outputs, list):
            out = mx.stack(outputs)
            # (B, OUT_DIM, T, H, W) -> (B, T, OUT_DIM, H, W)
            out = out.transpose(0, 2, 1, 3, 4)
        else:
            out = outputs

        return out

    def _decode_vae(self, latents):
        """Decode latents through VAE to pixel space.

        Args:
            latents: Latent tensor, shape (1, T, C, H, W).

        Returns:
            Pixel tensor, shape (1, 3, T, H*8, W*8), values in [0, 1].
        """
        # (B, T, C, H, W) -> (B, C, T, H, W) for VAE
        vae_input = latents.transpose(0, 2, 1, 3, 4)

        out = self.vae.decode(vae_input)

        # Normalize [-1, 1] -> [0, 1]
        pixels = mx.clip(out * 0.5 + 0.5, 0, 1)
        return pixels

    def generate(self, prompt, output='output.mp4', num_frames=None):
        """Generate a video from a text prompt.

        Args:
            prompt: Text prompt string.
            output: Output video file path.
            num_frames: Override number of frames.

        Returns:
            Generated pixel tensor, shape (1, 3, T, H, W).
        """
        if num_frames is not None:
            self.num_frames = num_frames
            self.fh = self.height // 8
            self.fw = self.width // 8
            self.ft = self.num_frames // 4
            self.frame_seq_len = self.ft * self.fh * self.fw

        self._initialize_caches()

        print(f"Encoding prompt: '{prompt}'")
        context = self._encode_text(prompt)
        print(f"Context shape: {context.shape}")

        # Generate video block by block
        num_blocks = self.num_frames // self.num_frame_per_block
        output_latents = mx.zeros(
            (1, self.num_frames, IN_DIM, self.fh, self.fw),
            dtype=self.dtype
        )

        current_start = 0

        for block_idx in range(num_blocks):
            print(f"\nBlock {block_idx + 1}/{num_blocks} "
                  f"(frames {current_start} to {current_start + self.num_frame_per_block - 1})")
            block_start_time = time.time()

            # Initialize noise for this block
            block_latents = mx.random.normal(
                (1, self.num_frame_per_block, IN_DIM, self.fh, self.fw),
                dtype=self.dtype
            )

            # Denoising loop
            for step_idx, current_timestep in enumerate(self.denoising_step_list):
                timestep = mx.array([current_timestep], dtype=mx.float32)

                print(f"  Step {step_idx + 1}/{len(self.denoising_step_list)}: "
                      f"timestep={current_timestep}", end='')
                step_start = time.time()

                # Model predicts velocity for flow matching
                flow_pred = self._forward_transformer(block_latents, timestep, context)

                # Euler step via scheduler
                block_latents = self.scheduler.step(flow_pred, timestep, block_latents)

                print(f" ({time.time() - step_start:.1f}s)")

            # Store denoised block
            output_latents[:, current_start:current_start + self.num_frame_per_block] = block_latents

            # Update KV cache with clean context (timestep=0)
            clean_latents = output_latents[:, current_start:current_start + self.num_frame_per_block]
            timestep_zero = mx.array([0], dtype=mx.float32)
            self._forward_transformer(clean_latents, timestep_zero, context)

            current_start += self.num_frame_per_block
            print(f"  Block done ({time.time() - block_start_time:.1f}s)")

        # VAE decode
        print("\nDecoding through VAE...")
        pixels = self._decode_vae(output_latents)
        print(f"Output shape: {pixels.shape}")

        print(f"Saving to {output}...")
        self._save_video(pixels, output)

        return pixels

    def _save_video(self, pixels, output_path):
        """Save pixel tensor as video using imageio."""
        try:
            import imageio
        except ImportError:
            print("WARNING: imageio not installed. Skipping video save.")
            return

        import subprocess

        frames = (pixels[0].transpose(1, 0, 2, 3).numpy() * 255).astype(np.uint8)

        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        temp_path = output_path.replace('.mp4', '_frames.npy')
        np.save(temp_path, frames)

        cmd = [
            'ffmpeg', '-y', '-r', '16', '-i', temp_path,
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
            '-pix_fmt', 'yuv420p', output_path
        ]
        subprocess.run(cmd, check=True)

        os.remove(temp_path)
        print(f"Saved video to {output_path}")

    def __del__(self):
        """Cleanup."""
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer.tokenizer


# Convenience function for quick usage
def generate(
    prompt,
    checkpoint_dir='./mlx_weights',
    output='output.mp4',
    **kwargs
):
    """Quick video generation from a prompt."""
    pipeline = MLXPipeline(
        transformer_path=os.path.join(checkpoint_dir, 'transformer.safetensors'),
        t5_path=os.path.join(checkpoint_dir, 't5_encoder.safetensors'),
        vae_path=os.path.join(checkpoint_dir, 'vae_decoder.safetensors'),
        **kwargs
    )
    return pipeline.generate(prompt, output=output)
