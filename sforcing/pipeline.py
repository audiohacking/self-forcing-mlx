"""Enhanced causal inference pipeline with KV caching and CFG.

Port of pipeline/causal_diffusion_inference.py to MLX.
Supports:
- KV-cache-based autoregressive inference (block-by-block)
- Classifier-free guidance (CFG) with conditional + unconditional
- Image-to-video via initial_latent
- Configurable denoising step schedules
"""

import time
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from sforcing.config import (
    DIM, FFN_DIM, NUM_HEADS, NUM_LAYERS, FREQ_DIM, TEXT_DIM, TEXT_LEN,
    IN_DIM, OUT_DIM, PATCH_SIZE, EPS, FRAME_SEQ_LEN, MAX_SEQ_LEN,
    GUIDANCE_SCALE, NUM_TRAIN_TIMESTEPS, HEAD_DIM,
)
from sforcing.model import WanModel
from sforcing.t5 import T5Encoder
from sforcing.vae import WanVAE
from sforcing.scheduler import FlowMatchScheduler
from sforcing.tokenizer_bridge import HuggingfaceTokenizer


def _initialize_kv_caches(batch_size, num_layers, max_seq_len, num_heads,
                          head_dim, dtype=mx.float32):
    """Initialize self-attention KV caches for all transformer blocks.

    Args:
        batch_size: Batch size.
        num_layers: Number of transformer blocks.
        max_seq_len: Maximum cache sequence length.
        num_heads: Number of attention heads.
        head_dim: Head dimension.
        dtype: Data type.

    Returns:
        List of KV cache dicts per block.
    """
    caches = []
    for _ in range(num_layers):
        caches.append({
            "k": mx.zeros((batch_size, max_seq_len, num_heads, head_dim), dtype=dtype),
            "v": mx.zeros((batch_size, max_seq_len, num_heads, head_dim), dtype=dtype),
            "global_end_index": mx.array([0], dtype=mx.int32),
            "local_end_index": mx.array([0], dtype=mx.int32),
        })
    return caches


def _initialize_crossattn_caches(batch_size, num_layers, text_len,
                                 num_heads, head_dim, dtype=mx.float32):
    """Initialize cross-attention KV caches for all transformer blocks.

    Args:
        batch_size: Batch size.
        num_layers: Number of transformer blocks.
        text_len: Text sequence length.
        num_heads: Number of attention heads.
        head_dim: Head dimension.
        dtype: Data type.

    Returns:
        List of cross-attn cache dicts per block.
    """
    caches = []
    for _ in range(num_layers):
        caches.append({
            "k": mx.zeros((batch_size, text_len, num_heads, head_dim), dtype=dtype),
            "v": mx.zeros((batch_size, text_len, num_heads, head_dim), dtype=dtype),
            "is_init": False,
        })
    return caches


def _reset_kv_caches(kv_caches):
    """Reset all KV cache indices for a new generation."""
    if kv_caches is not None:
        for cache in kv_caches:
            cache["global_end_index"] = mx.array([0], dtype=mx.int32)
            cache["local_end_index"] = mx.array([0], dtype=mx.int32)


def _reset_crossattn_caches(crossattn_caches):
    """Reset all cross-attention caches."""
    if crossattn_caches is not None:
        for cache in crossattn_caches:
            cache["is_init"] = False


class CausalInferencePipeline:
    """KV-cache enhanced causal inference pipeline.

    Generates video latents block-by-block with autoregressive KV caching,
    classifier-free guidance, and VAE decoding.

    Usage:
        pipeline = CausalInferencePipeline(
            transformer_path="mlx_weights/transformer.safetensors",
            t5_path="mlx_weights/t5_encoder.safetensors",
            vae_path="mlx_weights/vae_decoder.safetensors",
        )
        video = pipeline.generate("A cat walking")
    """

    def __init__(
        self,
        transformer_path: str,
        t5_path: str,
        vae_path: str,
        num_frames: int = 21,
        height: int = 480,
        width: int = 832,
        guidance_scale: float = GUIDANCE_SCALE,
        timestep_shift: float = 5.0,
        denoising_steps: Optional[List[int]] = None,
        num_frame_per_block: int = 3,
        negative_prompt: str = "",
    ):
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.guidance_scale = guidance_scale
        self.num_frame_per_block = num_frame_per_block
        self.negative_prompt = negative_prompt
        self.dtype = mx.float32

        # Frame-level dimensions
        self.fh = height // 8
        self.fw = width // 8
        self.ft = num_frames // 4
        self.frame_seq_len = self.ft * self.fh * self.fw
        self.max_seq_len = MAX_SEQ_LEN

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
        self.denoising_step_list = denoising_steps or [1000, 750, 500, 250]
        self.scheduler = FlowMatchScheduler(
            num_train_timesteps=NUM_TRAIN_TIMESTEPS,
            shift=timestep_shift,
            extra_one_step=True,
        )
        self.scheduler.set_timesteps(len(self.denoising_step_list))

        # --- KV caches (initialized per generation) ---
        self._kv_caches_pos = None
        self._kv_caches_neg = None
        self._crossattn_caches_pos = None
        self._crossattn_caches_neg = None

    def _load_weights(self, model, path):
        """Load weights from .safetensors into an MLX model."""
        from safetensors import safe_open
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
        """Initialize all KV caches for a new generation."""
        self._kv_caches_pos = _initialize_kv_caches(
            batch_size, NUM_LAYERS, self.max_seq_len, NUM_HEADS, HEAD_DIM, self.dtype)
        self._kv_caches_neg = _initialize_kv_caches(
            batch_size, NUM_LAYERS, self.max_seq_len, NUM_HEADS, HEAD_DIM, self.dtype)
        self._crossattn_caches_pos = _initialize_crossattn_caches(
            batch_size, NUM_LAYERS, TEXT_LEN, NUM_HEADS, HEAD_DIM, self.dtype)
        self._crossattn_caches_neg = _initialize_crossattn_caches(
            batch_size, NUM_LAYERS, TEXT_LEN, NUM_HEADS, HEAD_DIM, self.dtype)

    def _reset_caches(self):
        """Reset all caches for a new generation."""
        _reset_kv_caches(self._kv_caches_pos)
        _reset_kv_caches(self._kv_caches_neg)
        _reset_crossattn_caches(self._crossattn_caches_pos)
        _reset_crossattn_caches(self._crossattn_caches_neg)

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

    def _forward_transformer(self, latents, timestep, context,
                             kv_caches=None, crossattn_caches=None):
        """Single forward pass through the transformer with optional caches.

        Args:
            latents: Input latents, shape (1, T, C, H, W).
            timestep: Timestep, shape (B,).
            context: Text context, shape (1, TEXT_LEN, TEXT_DIM).
            kv_caches: Optional list of KV cache dicts per block.
            crossattn_caches: Optional list of cross-attn cache dicts per block.

        Returns:
            Output flow prediction, shape (1, T, OUT_DIM, H, W).
        """
        b, t, c, h, w = latents.shape

        # Convert to list of (C, T, H, W) for model
        x_list = [latents[i].transpose(2, 0, 1, 3, 4) for i in range(b)]

        grid_sizes = np.array([[self.ft, h, w]], dtype=np.int32)

        outputs = self.model(
            x_list, timestep, context, self.max_seq_len, grid_sizes,
            kv_caches=kv_caches, crossattn_caches=crossattn_caches,
        )

        if isinstance(outputs, list):
            out = mx.stack(outputs)
            # (B, OUT_DIM, T, H, W) -> (B, T, OUT_DIM, H, W)
            out = out.transpose(0, 2, 1, 3, 4)
        else:
            out = outputs

        return out

    def inference(
        self,
        noise: mx.array,
        context_cond: mx.array,
        context_uncond: mx.array,
        initial_latent: Optional[mx.array] = None,
    ) -> mx.array:
        """Run the full denoising pipeline with KV caching and CFG.

        Args:
            noise: Input noise, shape (1, num_frames, C, H, W).
            context_cond: Conditional text embeddings.
            context_uncond: Unconditional text embeddings.
            initial_latent: Optional initial latents for image-to-video.

        Returns:
            Denoised latents, shape (1, num_output_frames, C, H, W).
        """
        batch_size, num_frames, num_channels, height, width = noise.shape

        # Validate frame count
        if initial_latent is not None:
            num_input_frames = initial_latent.shape[1]
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            num_input_frames = 0
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block

        num_output_frames = num_frames + num_input_frames

        # Initialize caches
        self._initialize_caches(batch_size)

        output = mx.zeros(
            (batch_size, num_output_frames, num_channels, height, width),
            dtype=self.dtype,
        )

        # Step 1: Cache initial latent features
        current_start_frame = 0
        if initial_latent is not None:
            timestep_zero = mx.array([0], dtype=mx.float32)
            output[:, :1] = initial_latent[:, :1]

            # Run through model with timestep=0 to populate KV cache
            self._forward_transformer(
                initial_latent[:, :1], timestep_zero, context_cond,
                kv_caches=self._kv_caches_pos,
                crossattn_caches=self._crossattn_caches_pos,
            )
            self._forward_transformer(
                initial_latent[:, :1], timestep_zero, context_uncond,
                kv_caches=self._kv_caches_neg,
                crossattn_caches=self._crossattn_caches_neg,
            )
            current_start_frame += 1

        # Step 2: Temporal denoising loop (block by block)
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if initial_latent is None:
            all_num_frames = [1] + all_num_frames

        for block_idx, current_num_frames in enumerate(all_num_frames):
            block_start = time.time()
            print(f"  Block {block_idx + 1}/{len(all_num_frames)}: "
                  f"frames {current_start_frame}..{current_start_frame + current_num_frames - 1}")

            # Get noise for this block
            noisy_input = noise[
                :, current_start_frame - num_input_frames:
                   current_start_frame + current_num_frames - num_input_frames
            ]
            latents = noisy_input

            # Spatial denoising loop over timesteps
            for step_idx, current_timestep in enumerate(self.denoising_step_list):
                timestep = mx.array([current_timestep], dtype=mx.float32)

                # Conditional forward
                flow_pred_cond = self._forward_transformer(
                    latents, timestep, context_cond,
                    kv_caches=self._kv_caches_pos,
                    crossattn_caches=self._crossattn_caches_pos,
                )

                # Unconditional forward
                flow_pred_uncond = self._forward_transformer(
                    latents, timestep, context_uncond,
                    kv_caches=self._kv_caches_neg,
                    crossattn_caches=self._crossattn_caches_neg,
                )

                # CFG: combine conditional and unconditional predictions
                flow_pred = flow_pred_uncond + self.guidance_scale * (
                    flow_pred_cond - flow_pred_uncond
                )

                # Euler step via scheduler
                latents = self.scheduler.step(flow_pred, timestep, latents)

            # Store denoised block
            output[:, current_start_frame:current_start_frame + current_num_frames] = latents

            # Update KV cache with clean context (timestep=0)
            clean_latents = output[:, current_start_frame:current_start_frame + current_num_frames]
            timestep_zero = mx.array([0], dtype=mx.float32)

            self._forward_transformer(
                clean_latents, timestep_zero, context_cond,
                kv_caches=self._kv_caches_pos,
                crossattn_caches=self._crossattn_caches_pos,
            )
            self._forward_transformer(
                clean_latents, timestep_zero, context_uncond,
                kv_caches=self._kv_caches_neg,
                crossattn_caches=self._crossattn_caches_neg,
            )

            current_start_frame += current_num_frames
            print(f"    Done ({time.time() - block_start:.1f}s)")

        return output

    def generate(self, prompt: str, initial_latent: Optional[mx.array] = None,
                 output_path: Optional[str] = None) -> mx.array:
        """Generate a video from a text prompt.

        Args:
            prompt: Text prompt.
            initial_latent: Optional initial latents for image-to-video.
            output_path: Optional path to save video.

        Returns:
            Pixel tensor, shape (1, 3, T, H*8, W*8), values in [0, 1].
        """
        print(f"Encoding prompt: '{prompt}'")
        context_cond = self._encode_text(prompt)
        context_uncond = self._encode_text(self.negative_prompt)
        print(f"Context shapes - cond: {context_cond.shape}, uncond: {context_uncond.shape}")

        # Create noise
        noise = mx.random.normal(
            (1, self.num_frames, IN_DIM, self.fh, self.fw),
            dtype=self.dtype,
        )

        # Run inference
        print("Running inference...")
        latents = self.inference(noise, context_cond, context_uncond, initial_latent)

        # Decode through VAE
        print("Decoding through VAE...")
        pixels = self._decode_vae(latents)
        print(f"Output shape: {pixels.shape}")

        # Save if path provided
        if output_path is not None:
            self._save_video(pixels, output_path)

        return pixels

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
        pixels = (out * 0.5 + 0.5).clip(0, 1)
        return pixels

    def _save_video(self, pixels, output_path):
        """Save pixel tensor as video using imageio/ffmpeg."""
        try:
            import imageio
        except ImportError:
            print("WARNING: imageio not installed. Skipping video save.")
            return

        import subprocess
        import os

        frames = (pixels[0].transpose(1, 0, 2, 3).numpy() * 255).astype(np.uint8)

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        temp_path = output_path.replace('.mp4', '_frames.npy')
        np.save(temp_path, frames)

        cmd = [
            'ffmpeg', '-y', '-r', '16', '-i', temp_path,
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
            '-pix_fmt', 'yuv420p', output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        os.remove(temp_path)
        print(f"Saved video to {output_path}")
