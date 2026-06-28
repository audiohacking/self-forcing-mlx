"""Diffusion transformer — WanModel ported to MLX.

Ported from wan/modules/model.py and wan/modules/causal_model.py.
Implements the full diffusion transformer with AdaGN-Zero modulation.
"""
import math

import mlx.core as mx
import mlx.nn as nn

from sforcing.config import (
    DIM, FFN_DIM, NUM_HEADS, NUM_LAYERS, FREQ_DIM, TEXT_DIM, TEXT_LEN,
    IN_DIM, OUT_DIM, PATCH_SIZE, EPS,
)
from sforcing.utils import sinusoidal_embedding_1d, rope_params, unpatchify
from sforcing.attention import (
    WanLayerNorm, WanSelfAttention, WanT2VCrossAttention, WanFeedForward,
)


def _rope_freqs(max_seq_len=1024, dim=None, num_heads=12):
    """Compute 3D RoPE frequencies matching CUDA.

    Splits head_dim into three parts for (t, h, w) axes:
      temporal: head_dim - 4*(head_dim//6)
      height:   2*(head_dim//6)
      width:    2*(head_dim//6)

    Returns complex64 tensor, shape (max_seq_len, head_dim//2).
    """
    d = dim // num_heads
    freqs = mx.concatenate([
        rope_params(max_seq_len, d - 4 * (d // 6)),
        rope_params(max_seq_len, 2 * (d // 6)),
        rope_params(max_seq_len, 2 * (d // 6)),
    ], axis=1)
    return freqs


class AttentionBlock(nn.Module):
    """Single transformer block with self-attention, cross-attention, and FFN.

    Implements the pre-norm residual architecture:
      norm1 -> self_attn + residual
      norm3 -> cross_attn + residual
      norm2 -> ffn + residual

    With AdaGN-Zero modulation from time embedding.
    """

    def __init__(self, dim=DIM, ffn_dim=FFN_DIM, num_heads=NUM_HEADS,
                 qk_norm=True, eps=EPS):
        super().__init__()
        self.dim = dim

        # normalization layers — CUDA: norm1/norm2 have elementwise_affine=False
        self.norm1 = WanLayerNorm(dim, eps=eps)
        self.norm2 = WanLayerNorm(dim, eps=eps)
        # norm3 has elementwise_affine=True (matches CUDA cross_attn_norm=True)
        self.norm3 = WanLayerNorm(dim, eps=eps, elementwise_affine=True)

        # attention
        self.self_attn = WanSelfAttention(dim, num_heads, qk_norm=qk_norm, eps=eps)
        self.cross_attn = WanT2VCrossAttention(dim, num_heads, qk_norm=qk_norm, eps=eps)

        # feed-forward
        self.ffn = WanFeedForward(dim, ffn_dim)

        # AdaGN-Zero modulation: [B, 6, dim] learned offset per block
        self.modulation = mx.random.normal((1, 6, dim)) * (dim ** -0.5)

    def __call__(self, x, e, seq_lens, grid_sizes, freqs, context,
                 context_lens=None, crossattn_cache=None, kv_cache=None):
        """Forward pass through one transformer block.

        Args:
            x: Video tokens, shape (B, L, dim).
            e: Time embedding, shape (B, 6, dim).
            seq_lens: Sequence lengths per sample.
            grid_sizes: (F, H, W) per sample, shape (B, 3).
            freqs: RoPE frequencies.
            context: Text embeddings, shape (B, TEXT_LEN, TEXT_DIM).
            context_lens: Optional context lengths.
            crossattn_cache: Optional KV cache for cross-attn.
            kv_cache: Optional KV cache for self-attention.

        Returns:
            Updated x tensor.
        """
        # Unfold time embedding into 6 modulation scalars
        mod = (self.modulation + e).reshape(-1, 6, self.dim)  # (B, 6, dim)
        # MLX split differs from PyTorch — use indexing instead
        e0, e1, e2, e3, e4, e5 = [mod[:, i:i+1, :] for i in range(6)]
        e0 = e0.squeeze(1)
        e1 = e1.squeeze(1)
        e2 = e2.squeeze(1)
        e3 = e3.squeeze(1)
        e4 = e4.squeeze(1)
        e5 = e5.squeeze(1)

        # Self-attention: norm1(x) * (1 + e1) + e0 -> attn -> x + out * e2
        norm1_out = self.norm1(x)
        x_attn_input = norm1_out * (1.0 + e1) + e0
        sa_out = self.self_attn(x_attn_input, grid_sizes, freqs, kv_cache=kv_cache)
        x = x + sa_out * e2

        # Cross-attention: norm3(x) -> cross_attn -> x + out
        ca_in = self.norm3(x)
        ca_out = self.cross_attn(ca_in, context, crossattn_cache=crossattn_cache)
        x = x + ca_out

        # FFN: norm2(x) * (1 + e4) + e3 -> ffn -> x + out * e5
        norm2_out = self.norm2(x)
        ffn_in = norm2_out * (1.0 + e4) + e3
        ffn_out = self.ffn(ffn_in)
        x = x + ffn_out * e5

        return x


class Head(nn.Module):
    """Final layer: LayerNorm -> Linear -> 2-channel modulation.

    Produces the output that will be unpatchified into video latent space.
    """

    def __init__(self, dim=DIM, out_dim=OUT_DIM, patch_size=PATCH_SIZE, eps=EPS):
        super().__init__()
        self.dim = dim
        out_channels = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps=eps)  # elementwise_affine=False matches CUDA
        self.head = nn.Linear(dim, out_channels)  # bias=True (CUDA default)

        # 2-channel modulation
        self.modulation = mx.random.normal((1, 2, dim)) * (dim ** -0.5)

    def __call__(self, x, e):
        """Forward pass through the head.

        Args:
            x: Input tokens, shape (B, L, dim).
            e: Time embedding, shape (B, dim).

        Returns:
            Output, shape (B, L, out_channels).
        """
        mod = (self.modulation + e.reshape(-1, 1, self.dim))
        e0 = mod[:, 0:1, :].squeeze(1)
        e1 = mod[:, 1:2, :].squeeze(1)
        out = self.head(self.norm(x) * (1.0 + e1) + e0)
        return out


class WanModel(nn.Module):
    """Full diffusion transformer backbone.

    Consists of:
    - Patch embedding (Conv3d -> flatten)
    - Text embedding (Linear -> GELU -> Linear)
    - Time embedding (Linear -> SiLU -> Linear -> projection)
    - Stacked transformer blocks
    - Head + unpatchify
    """

    def __init__(
        self,
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
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.text_len = text_len
        self.out_dim = out_dim
        self.freq_dim = freq_dim

        # Patch embedding: Conv3d(16, dim, kernel=(1,2,2), stride=(1,2,2))
        # bias=True matches CUDA default
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size,
                                         stride=patch_size)

        # Text embedding: Linear -> GELU -> Linear (both with bias=True)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # Time embedding: Linear -> SiLU -> Linear (both with bias=True)
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        # Time projection: SiLU -> Linear(dim, dim*6) (bias=True)
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )

        # Transformer blocks
        self.blocks = [
            AttentionBlock(dim, ffn_dim, num_heads, qk_norm=True, eps=eps)
            for _ in range(num_layers)
        ]

        # Head
        self.head = Head(dim, out_dim, patch_size, eps)

        # Precomputed RoPE frequencies (complex64)
        self.freqs = _rope_freqs(dim=dim, num_heads=num_heads)

    def __call__(self, x, t, context, seq_len, grid_sizes=None,
                 kv_caches=None, crossattn_caches=None):
        """Forward pass through the diffusion model.

        Args:
            x: Input video latents, list of tensors shape (C_in, F, H, W) or
               batch tensor (B, C_in, F, H, W).
            t: Timestep tensor, shape (B,).
            context: Text embeddings, shape (B, TEXT_LEN, TEXT_DIM).
            seq_len: Maximum sequence length for padding.
            grid_sizes: Optional (F_patches, H_patches, W_patches) per sample.
            kv_caches: Optional list of KV cache dicts per block for self-attention.
            crossattn_caches: Optional list of cross-attn cache dicts per block.

        Returns:
            Reconstructed outputs, list of shape (B, out_dim, F, H/8, W/8).
        """
        # Handle both list and batch tensor inputs
        if isinstance(x, (list, tuple)):
            batch_inputs = list(x)
        else:
            if x.ndim == 4:
                x = mx.expand_dims(x, 0)
            batch_inputs = [x[i] for i in range(x.shape[0])]

        # MLX Conv3d expects NHWC format (B, D, H, W, C), not NCHW (B, C, D, H, W).
        # Transpose each sample: (C, F, H, W) -> (F, H, W, C)
        patches = []
        for u in batch_inputs:
            # u: (C, F, H, W) -> (F, H, W, C) for MLX conv3d
            u_nhwc = u.transpose(1, 2, 3, 0)  # (F, H, W, C)
            u_nhwc = mx.expand_dims(u_nhwc, 0)  # (1, F, H, W, C)
            p = self.patch_embedding(u_nhwc)  # (1, F', H', W', C')
            patches.append(p)
        # Compute grid sizes from patch embedding output (B, F', H', W', C')
        if grid_sizes is None:
            grid_sizes = mx.stack([
                mx.array([p.shape[1], p.shape[2], p.shape[3]], dtype=mx.int32)
                for p in patches
            ])
        else:
            grid_sizes = mx.array(grid_sizes)

        # Flatten: (B, F', H', W', C') -> (B, F'*H'*W', C')
        flattened = [p.reshape(p.shape[0], -1, p.shape[-1]) for p in patches]

        # Compute seq_lens
        seq_lens_arr = mx.array([f.shape[1] for f in flattened], dtype=mx.int32)

        # Pad to seq_len
        padded = []
        for f in flattened:
            pad_len = seq_len - f.shape[1]
            if pad_len > 0:
                pad_zeros = mx.zeros((f.shape[0], pad_len, f.shape[2]), dtype=f.dtype)
                f = mx.concatenate([f, pad_zeros], axis=1)
            padded.append(f)
        x = mx.concatenate(padded, axis=0)

        # Time embedding
        time_emb = sinusoidal_embedding_1d(self.freq_dim, t.astype(mx.float32))
        e = self.time_embedding(time_emb.astype(x.dtype))
        e0 = self.time_projection(e).reshape(-1, 6, self.dim)

        # Text embedding — pad/truncate context to text_len
        text_padded = []
        for c in context:
            cur_len = c.shape[0]
            if cur_len < self.text_len:
                pad_len = self.text_len - cur_len
                pad_zeros = mx.zeros((pad_len, context.shape[-1]))
                c = mx.concatenate([c, pad_zeros], axis=0)
            text_padded.append(c[:self.text_len])
        context_embed = self.text_embedding(mx.stack(text_padded))

        # Forward through blocks with optional KV caches
        for i, block in enumerate(self.blocks):
            block_kv = kv_caches[i] if kv_caches is not None else None
            block_cross = crossattn_caches[i] if crossattn_caches is not None else None
            x = block(x, e0, seq_lens_arr, grid_sizes, self.freqs,
                      context_embed, kv_cache=block_kv, crossattn_cache=block_cross)

        # Head
        x = self.head(x, e)

        # Unpatchify
        outputs = unpatchify(x, grid_sizes.tolist(), self.patch_size, self.out_dim)

        return outputs

    def load_weights(self, weights_path):
        """Load weights from a .safetensors file.

        Expects keys in MLX-compatible naming (produced by converter.py):
        - blocks.N.self_attn.q.weight  (dot notation for list indices)
        - blocks.N.ffn.fc1.weight      (named FFN layers, not Sequential indices)
        - text_embedding.layers.N.weight  (Sequential .layers. access)
        """
        from mlx.utils import tree_unflatten
        from safetensors.numpy import load_file

        weights = load_file(weights_path)
        # Convert numpy arrays to mx.array
        weights = [(k, mx.array(v)) for k, v in weights.items()]
        self.update(tree_unflatten(weights))

    def _mlxify_params(self, pytorch_params):
        """Map PyTorch parameter names to MLX names.

        Handles:
        - blocks.N.self_attn.q.weight -> blocks[N].self_attn.q.weight
        - blocks.N.self_attn.k.weight -> blocks[N].self_attn.k.weight
        - blocks.N.ffn.0.weight -> blocks[N].ffn.fc1.weight
        - blocks.N.ffn.2.weight -> blocks[N].ffn.fc2.weight
        """
        mlx_params = {}
        name_map = {
            "patch_embedding.weight": "patch_embedding.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "text_embedding.0.weight": "text_embedding[0].weight",
            "text_embedding.0.bias": "text_embedding[0].bias",
            "text_embedding.2.weight": "text_embedding[2].weight",
            "text_embedding.2.bias": "text_embedding[2].bias",
            "time_embedding.0.weight": "time_embedding[0].weight",
            "time_embedding.0.bias": "time_embedding[0].bias",
            "time_embedding.2.weight": "time_embedding[2].weight",
            "time_embedding.2.bias": "time_embedding[2].bias",
            "time_projection.1.weight": "time_projection[1].weight",
            "time_projection.1.bias": "time_projection[1].bias",
            # head.norm has elementwise_affine=False — no weight/bias in checkpoint
            "head.head.weight": "head.head.weight",
            "head.head.bias": "head.head.bias",
        }

        # Handle direct mappings
        for key, value in pytorch_params.items():
            if key in name_map:
                mlx_params[name_map[key]] = value
                continue

            ml_key = key

            # Handle block weights
            if key.startswith("blocks."):
                parts = key.split(".")
                block_idx = int(parts[1])
                param_name = ".".join(parts[2:])

                # Map norm layers
                if param_name.startswith("norm1"):
                    suffix = param_name[len("norm1"):]
                    ml_key = f"blocks[{block_idx}].norm1{suffix}"
                elif param_name.startswith("norm2"):
                    suffix = param_name[len("norm2"):]
                    ml_key = f"blocks[{block_idx}].norm2{suffix}"
                elif param_name.startswith("norm3"):
                    suffix = param_name[len("norm3"):]
                    ml_key = f"blocks[{block_idx}].norm3{suffix}"
                elif param_name.startswith("self_attn."):
                    inner = param_name[len("self_attn."):]
                    ml_key = f"blocks[{block_idx}].self_attn.{inner}"
                elif param_name.startswith("cross_attn."):
                    inner = param_name[len("cross_attn."):]
                    ml_key = f"blocks[{block_idx}].cross_attn.{inner}"
                elif param_name.startswith("ffn."):
                    # Map Sequential indices: ffn.0.weight -> ffn.fc1.weight
                    inner = param_name[len("ffn."):]
                    if inner.startswith("0."):
                        ml_key = f"blocks[{block_idx}].ffn.fc1.{inner[2:]}"
                    elif inner.startswith("2."):
                        ml_key = f"blocks[{block_idx}].ffn.fc2.{inner[2:]}"
                    else:
                        ml_key = f"blocks[{block_idx}].ffn.{inner}"
                elif param_name == "modulation":
                    ml_key = f"blocks[{block_idx}].modulation"
                else:
                    ml_key = key

            mlx_params[ml_key] = value

        return mlx_params
