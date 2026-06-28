"""Attention primitives ported from PyTorch to MLX.

Ported from wan/modules/model.py and wan/modules/attention.py.
Uses standard SDPA via matmul+softmax (no flash attention on Apple Silicon).
"""
import mlx.core as mx
import mlx.nn as nn

from sforcing.utils import rope_apply, rope_apply_causal


class WanRMSNorm(nn.Module):
    """RMS Norm without bias, learnable weight scaling.

    Used for QK normalization in self-attention and cross-attention.
    """

    def __init__(self, dim, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        if elementwise_affine:
            self.weight = mx.ones(dim)
        else:
            self.weight = None

    def __call__(self, x):
        # Compute RMS in float32 for stability, then cast back
        dtype = x.dtype
        x_f32 = x.astype(mx.float32)
        r = mx.rsqrt(mx.mean(x_f32 ** 2, axis=-1, keepdims=True) + self.eps)
        x_norm = (x_f32 * r).astype(dtype)
        if self.weight is not None:
            x_norm = x_norm * self.weight
        return x_norm


class WanLayerNorm(nn.LayerNorm):
    """Standard LayerNorm, ported from PyTorch with dtype preservation."""

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, eps=eps, affine=elementwise_affine)

    def __call__(self, x):
        return super().__call__(x).astype(x.dtype)


class WanSelfAttention(nn.Module):
    """Self-attention with QK normalization and 3D RoPE.

    For single-frame or multi-frame self-attention within the transformer.
    Uses SDPA via explicit matmul+softmax (compatible with MLX on Apple Silicon).
    Matches CUDA: Linear layers with bias=True, QK RMSNorm.
    """

    def __init__(self, dim, num_heads, window_size=(-1, -1),
                 qk_norm=True, eps=1e-6):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.eps = eps

        # linear layers — match CUDA: nn.Linear with bias=True
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        # QK normalization
        self.norm_q = WanRMSNorm(dim, eps=eps, elementwise_affine=True) if qk_norm else None
        self.norm_k = WanRMSNorm(dim, eps=eps, elementwise_affine=True) if qk_norm else None

    def __call__(self, x, grid_sizes, freqs, causal_mask=None, kv_cache=None):
        """Forward pass for self-attention.

        Args:
            x: Input tensor, shape (B, L, dim).
            grid_sizes: Grid dimensions (F, H, W), shape (B, 3) or list of tuples.
            freqs: Precomputed RoPE frequencies.
            causal_mask: Optional causal mask for autoregressive attention.
            kv_cache: Optional dict with 'k', 'v', 'global_end_index', 'local_end_index'.

        Returns:
            Output tensor, shape (B, L, dim).
        """
        b, s, n, d = x.shape[0], x.shape[1], self.num_heads, self.head_dim

        # Q, K, V with normalization (CUDA: norm_q(self.q(x)))
        q = self.norm_q(self.q(x)).reshape(b, s, n, d)
        k = self.norm_k(self.k(x)).reshape(b, s, n, d)
        v = self.v(x).reshape(b, s, n, d)

        # Apply 3D RoPE to q and k
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)

        # KV cache management for autoregressive inference
        if kv_cache is not None:
            # Compute cache positions
            cache_local = kv_cache["local_end_index"].item()
            cache_global = kv_cache["global_end_index"].item()

            # Store new K,V in cache
            kv_cache["k"][:, cache_local:cache_local + s] = k
            kv_cache["v"][:, cache_local:cache_local + s] = v
            kv_cache["global_end_index"] = mx.array([cache_global + s], dtype=mx.int32)
            kv_cache["local_end_index"] = mx.array([cache_local + s], dtype=mx.int32)

            # Use full cached K,V for attention
            full_s = cache_local + s
            k = kv_cache["k"][:, :full_s]
            v = kv_cache["v"][:, :full_s]

        # SDPA via matmul+softmax
        q_t = q.transpose(0, 2, 1, 3)  # (B, N, S, D)
        k_t = k.transpose(0, 2, 1, 3)  # (B, N, S, D)
        v_t = v.transpose(0, 2, 1, 3)  # (B, N, S, D)

        attn = mx.matmul(q_t, k_t.transpose(0, 1, 3, 2)) / (d ** 0.5)

        if causal_mask is not None:
            attn = attn + causal_mask

        attn = mx.softmax(attn, axis=-1)
        out = mx.matmul(attn, v_t)  # (B, N, S, D)
        out = out.transpose(0, 2, 1, 3)  # (B, S, N, D)
        out = out.reshape(b, s, -1)  # (B, S, dim)

        # Output projection
        out = self.o(out)
        return out


class WanT2VCrossAttention(nn.Module):
    """Cross-attention for text-to-video.

    Query comes from video tokens, keys/values come from text context.
    Supports KV caching across denoising steps.
    Matches CUDA: Linear layers with bias=True, QK RMSNorm.
    """

    def __init__(self, dim, num_heads, window_size=(-1, -1),
                 qk_norm=True, eps=1e-6):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps

        # linear layers — match CUDA with bias=True
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        # QK normalization
        self.norm_q = WanRMSNorm(dim, eps=eps, elementwise_affine=True) if qk_norm else None
        self.norm_k = WanRMSNorm(dim, eps=eps, elementwise_affine=True) if qk_norm else None

    def __call__(self, x, context, context_lens=None, crossattn_cache=None):
        """Forward pass for cross-attention.

        Args:
            x: Video tokens, shape (B, L1, dim).
            context: Text embeddings, shape (B, L2, TEXT_DIM).
            context_lens: Optional context lengths per sample.
            crossattn_cache: Optional KV cache dict with 'k', 'v', 'is_init' keys.

        Returns:
            Output tensor, shape (B, L1, dim).
        """
        b, n, d = x.shape[0], self.num_heads, self.head_dim

        # Query from video
        q = self.norm_q(self.q(x)).reshape(b, -1, n, d)

        # Key/Value from text context (with optional caching)
        if crossattn_cache is not None and crossattn_cache.get("is_init", False):
            k = crossattn_cache["k"]
            v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).reshape(b, -1, n, d)
            v = self.v(context).reshape(b, -1, n, d)
            if crossattn_cache is not None:
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
                crossattn_cache["is_init"] = True

        # SDPA
        q_t = q.transpose(0, 2, 1, 3)  # (B, N, L1, D)
        k_t = k.transpose(0, 2, 1, 3)  # (B, N, L2, D)
        v_t = v.transpose(0, 2, 1, 3)  # (B, N, L2, D)

        attn = mx.matmul(q_t, k_t.transpose(0, 1, 3, 2)) / (d ** 0.5)
        attn = mx.softmax(attn, axis=-1)
        out = mx.matmul(attn, v_t)  # (B, N, L1, D)
        out = out.transpose(0, 2, 1, 3)  # (B, L1, N, D)
        out = out.reshape(b, -1, self.dim)

        # Output projection
        out = self.o(out)
        return out


class WanFeedForward(nn.Module):
    """Simple feed-forward network matching CUDA.

    Matches PyTorch: nn.Sequential(
        nn.Linear(dim, ffn_dim),
        nn.GELU(approximate='tanh'),
        nn.Linear(ffn_dim, dim),
    )
    NOT a gated FFN.
    """

    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ffn_dim)   # bias=True (CUDA default)
        self.fc2 = nn.Linear(ffn_dim, dim)   # bias=True (CUDA default)

    def __call__(self, x):
        return self.fc2(self._gelu_tanh(self.fc1(x)))

    def _gelu_tanh(self, x):
        """GELU via tanh approximation — matches nn.GELU(approximate='tanh')."""
        return 0.5 * x * (1.0 + mx.tanh(
            mx.sqrt(2.0 / mx.pi) * (x + 0.044715 * mx.power(x, 3.0))
        ))
