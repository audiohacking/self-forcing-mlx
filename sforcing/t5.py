"""T5 encoder (UMT5-XXL encoder-only) ported to MLX.

Ported from wan/modules/t5.py.
Only the encoder path is needed for text-to-video inference.
"""
import math

import mlx.core as mx
import mlx.nn as nn


def fp16_clamp(x, max_val=65504.0):
    """Clamp inf values in float16/bfloat16."""
    if x.dtype in [mx.float16, mx.bfloat16]:
        mask = mx.isinf(x)
        x = mx.where(mask, mx.array(max_val - 1000, dtype=x.dtype), x)
    return x


class T5LayerNorm(nn.Module):
    """T5's LayerNorm (no bias, weight scaling, computed in float32)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = mx.ones(dim)

    def __call__(self, x):
        x_f32 = x.astype(mx.float32)
        r = mx.rsqrt(mx.mean(x_f32 ** 2, axis=-1, keepdims=True) + self.eps)
        x_norm = (x_f32 * r).astype(x.dtype)
        return x_norm * self.weight


class GELU(nn.Module):
    """GELU approximation using tanh (same as nn.GELU(approximate='tanh'))."""

    def __call__(self, x):
        return 0.5 * x * (1.0 + mx.tanh(
            mx.sqrt(2.0 / mx.pi) * (x + 0.044715 * mx.power(x, 3.0))
        ))


class T5Attention(nn.Module):
    """T5 attention — no QK scaling, relative positional bias."""

    def __init__(self, dim, dim_attn, num_heads, dropout=0.1):
        super().__init__()
        assert dim_attn % num_heads == 0
        self.dim = dim
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads

        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x, context=None, mask=None, pos_bias=None, training=False):
        """Forward pass for T5 attention.

        Args:
            x: Input tensor, shape (B, L1, C).
            context: Context tensor (or None for self-attention), shape (B, L2, C).
            mask: Attention mask, shape (B, L1, L2) or (B, L2).
            pos_bias: Relative positional bias tensor.
            training: Training mode (for dropout).

        Returns:
            Output tensor, shape (B, L1, C).
        """
        if context is None:
            context = x
        b, n, c = x.shape[0], self.num_heads, self.head_dim

        q = self.q(x).reshape(b, -1, n, c)
        k = self.k(context).reshape(b, -1, n, c)
        v = self.v(context).reshape(b, -1, n, c)

        # Attention bias
        attn_bias = mx.zeros((b, n, q.shape[1], k.shape[1]), dtype=mx.float32)
        if pos_bias is not None:
            attn_bias = attn_bias + pos_bias
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.reshape(b, 1, 1, -1)
            # mask: 0 = valid, -inf = masked; apply to attention
            attn_bias = attn_bias + mx.where(
                mask == 0,
                mx.array(-1e9, dtype=mx.float32),
                mx.zeros((), dtype=mx.float32)
            )

        # Attention without scaling (T5 uses raw dot product)
        attn = mx.matmul(q.transpose(0, 2, 1, 3), k.transpose(0, 2, 3, 1)) + attn_bias
        attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(attn.dtype)
        out = mx.matmul(attn, v.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
        out = out.reshape(b, -1, n * c)
        out = self.o(out)
        # MLX Dropout doesn't support training flag; skip for inference
        return out


class T5FeedForward(nn.Module):
    """T5 feed-forward with SwiGLU gating."""

    def __init__(self, dim, dim_ffn, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.dim_ffn = dim_ffn

        self.gate = nn.Sequential(
            nn.Linear(dim, dim_ffn, bias=False),
            GELU(),
        )
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x, training=False):
        """Forward pass matching CUDA — dropout after gating and after fc2.

        Args:
            x: Input tensor, shape (B, L, C).
            training: Training mode.

        Returns:
            Output tensor, same shape.
        """
        out = self.fc1(x) * self.gate(x)
        out = self.fc2(out)
        return out


class T5RelativeEmbedding(nn.Module):
    """Relative positional embedding with linear buckets."""

    def __init__(self, num_buckets, num_heads, bidirectional, max_dist=128):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def __call__(self, lq, lk):
        """Compute relative positional bias.

        Args:
            lq: Query sequence length.
            lk: Key sequence length.

        Returns:
            Position bias tensor, shape (1, num_heads, lq, lk).
        """
        # relative position: pos[k] - pos[q]
        rel_pos = mx.arange(lk, dtype=mx.int32).reshape(1, -1) - \
                  mx.arange(lq, dtype=mx.int32).reshape(-1, 1)
        rel_pos = self._relative_position_bucket(rel_pos)
        rel_pos_embeds = self.embedding(rel_pos)  # (lq, lk, num_heads)
        rel_pos_embeds = rel_pos_embeds.transpose(2, 0, 1).reshape(1, -1, lq, lk)
        return rel_pos_embeds

    def _relative_position_bucket(self, rel_pos):
        """Compute position buckets for relative positions."""
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).astype(mx.int32) * num_buckets
            rel_pos = mx.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = mx.zeros_like(rel_pos)
            rel_pos = -mx.minimum(rel_pos, mx.zeros_like(rel_pos))

        max_exact = num_buckets // 2
        rel_pos_large = max_exact + (
            mx.log(rel_pos.astype(mx.float32) / max_exact + 1e-9) /
            mx.log(self.max_dist / max_exact) * (num_buckets - max_exact)
        ).astype(mx.int32)
        rel_pos_large = mx.minimum(rel_pos_large, num_buckets - 1)
        rel_buckets = rel_buckets + mx.where(rel_pos < max_exact, rel_pos, rel_pos_large)
        return rel_buckets


class T5SelfAttention(nn.Module):
    """T5 encoder block: norm -> attn + add, norm -> FFN + add."""

    def __init__(self, dim, dim_attn, dim_ffn, num_heads, num_buckets,
                 shared_pos=True, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.shared_pos = shared_pos
        self.num_heads = num_heads

        self.norm1 = T5LayerNorm(dim)
        self.attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)

        if shared_pos:
            self.pos_embedding = None
        else:
            self.pos_embedding = T5RelativeEmbedding(
                num_buckets, num_heads, bidirectional=True)

    def __call__(self, x, mask=None, pos_bias=None, training=False):
        """Forward pass.

        Args:
            x: Input tensor.
            mask: Attention mask.
            pos_bias: Relative positional bias (shared or per-block).
            training: Training mode.

        Returns:
            Output tensor with residual connections.
        """
        pb = pos_bias if self.shared_pos else (
            self.pos_embedding(x.shape[1], x.shape[1]) if self.pos_embedding is not None else None
        )
        residual = x
        x = self.norm1(x)
        attn_out = self.attn(x, mask=mask, pos_bias=pb, training=training)
        x = fp16_clamp(residual + attn_out)

        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x, training=training)
        x = fp16_clamp(residual + ffn_out)
        return x


class T5Encoder(nn.Module):
    """UMT5-XXL encoder-only for text embeddings.

    24 layers, 64 heads, dim=4096.
    """

    def __init__(self, vocab_size=256384, dim=4096, dim_attn=4096, dim_ffn=10240,
                 num_heads=64, num_layers=24, num_buckets=32,
                 shared_pos=False, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.shared_pos = shared_pos

        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=True) if shared_pos else None
        self.dropout = nn.Dropout(dropout)

        self.blocks = [
            T5SelfAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets,
                            shared_pos, dropout)
            for _ in range(num_layers)
        ]

        self.norm = T5LayerNorm(dim)

    def __call__(self, ids, mask=None, training=False):
        """Forward pass through the encoder.

        Args:
            ids: Token IDs, shape (B, L).
            mask: Attention mask, shape (B, L).
            training: Training mode.

        Returns:
            Context embeddings, shape (B, L, dim).
        """
        x = self.token_embedding(ids)

        if self.shared_pos and self.pos_embedding is not None:
            pb = self.pos_embedding(x.shape[1], x.shape[1])
        else:
            pb = None

        for block in self.blocks:
            x = block(x, mask=mask, pos_bias=pb, training=training)

        x = self.norm(x)
        return x

    def load_weights(self, weights_path):
        """Load weights from a .safetensors file."""
        from mlx.utils import tree_flatten, tree_unflatten
        from safetensors.numpy import load_file

        weights = load_file(weights_path)
        mlx_items = self._mlxify_params(dict(weights))

        # Get model parameter names from tree_flatten (leaf arrays only)
        model_params = dict(tree_flatten(self))
        model_param_names = set(k for k, v in model_params.items()
                                if isinstance(v, mx.array))

        # Filter to only matching keys
        filtered = [(k, mx.array(v)) for k, v in mlx_items.items()
                     if k in model_param_names]
        skipped = len(mlx_items) - len(filtered)
        if skipped:
            print(f"  Skipped {skipped} unmatched keys")

        self.update(tree_unflatten(filtered))

    def _mlxify_params(self, pytorch_params):
        """Map PyTorch T5 parameter names to MLX names.

        The safetensors file already uses dot notation (MLX-compatible).
        Only needed transformation: gate.0 -> gate.layers.0 (MLX Sequential).
        """
        import re
        mlx_params = {}
        for key, value in pytorch_params.items():
            ml_key = key

            # Handle encoder prefix from full checkpoints
            if ml_key.startswith("encoder."):
                ml_key = ml_key[8:]

            # MLX nn.Sequential stores layers in .layers attribute
            # gate.0.weight -> gate.layers.0.weight
            ml_key = re.sub(r'(\.gate)\.(\d+)\.', r'\1.layers.\2.', ml_key)

            mlx_params[ml_key] = value

        return mlx_params