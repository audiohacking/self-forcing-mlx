"""Tests for attention primitives."""
import mlx.core as mx
import numpy as np
import pytest

from sforcing.attention import (
    WanRMSNorm,
    WanLayerNorm,
    WanSelfAttention,
    WanT2VCrossAttention,
    WanFeedForward,
)
from sforcing.utils import rope_params
from sforcing.config import DIM, FFN_DIM, NUM_HEADS


def assert_finite(tensor, name="tensor"):
    arr = np.array(tensor)
    assert not np.any(np.isnan(arr)), f"{name} contains NaN"
    assert not np.any(np.isinf(arr)), f"{name} contains inf"


class TestWanRMSNorm:
    def test_output_shape(self):
        norm = WanRMSNorm(1536)
        x = mx.random.normal((1, 100, 1536))
        out = norm(x)
        assert out.shape == (1, 100, 1536)

    def test_no_nan(self):
        norm = WanRMSNorm(1536)
        x = mx.random.normal((1, 100, 1536))
        assert_finite(norm(x), "RMSNorm output")

    def test_elementwise_affine_false(self):
        norm = WanRMSNorm(1536, elementwise_affine=False)
        x = mx.random.normal((1, 100, 1536))
        out = norm(x)
        assert out.shape == (1, 100, 1536)


class TestWanLayerNorm:
    def test_output_shape(self):
        norm = WanLayerNorm(1536)
        x = mx.random.normal((1, 100, 1536))
        out = norm(x)
        assert out.shape == (1, 100, 1536)

    def test_no_nan(self):
        norm = WanLayerNorm(1536)
        x = mx.random.normal((1, 100, 1536))
        assert_finite(norm(x), "LayerNorm output")


class TestWanSelfAttention:
    def test_output_shape(self):
        attn = WanSelfAttention(DIM, NUM_HEADS)
        x = mx.random.normal((1, 100, DIM))
        grid_sizes = mx.array([[1, 10, 10]], dtype=mx.int32)
        freqs = rope_params(1024, DIM // NUM_HEADS)
        out = attn(x, grid_sizes, freqs)
        assert out.shape == (1, 100, DIM)

    def test_no_nan(self):
        attn = WanSelfAttention(DIM, NUM_HEADS)
        x = mx.random.normal((1, 100, DIM))
        grid_sizes = mx.array([[1, 10, 10]], dtype=mx.int32)
        freqs = rope_params(1024, DIM // NUM_HEADS)
        assert_finite(attn(x, grid_sizes, freqs), "self-attn output")

    def test_kv_cache_shape(self):
        """KV cache should not change output shape."""
        attn = WanSelfAttention(DIM, NUM_HEADS)
        b, s, n, d = 1, 100, NUM_HEADS, DIM // NUM_HEADS
        x = mx.random.normal((b, s, DIM))
        grid_sizes = mx.array([[1, 10, 10]], dtype=mx.int32)
        freqs = rope_params(1024, d)

        kv_cache = {
            "k": mx.zeros((b, 1000, n, d)),
            "v": mx.zeros((b, 1000, n, d)),
            "global_end_index": mx.array([0], dtype=mx.int32),
            "local_end_index": mx.array([0], dtype=mx.int32),
        }
        seq_lens = mx.array([s], dtype=mx.int32)

        out = attn(x, grid_sizes, freqs, kv_cache=kv_cache, seq_lens=seq_lens)
        assert out.shape == (b, s, DIM)
        assert_finite(out, "self-attn with KV cache")

    def test_kv_cache_accumulation(self):
        """KV cache should accumulate across calls."""
        attn = WanSelfAttention(DIM, NUM_HEADS)
        b, n, d = 1, NUM_HEADS, DIM // NUM_HEADS

        kv_cache = {
            "k": mx.zeros((b, 1000, n, d)),
            "v": mx.zeros((b, 1000, n, d)),
            "global_end_index": mx.array([0], dtype=mx.int32),
            "local_end_index": mx.array([0], dtype=mx.int32),
        }
        freqs = rope_params(1024, d)

        # First call: 100 tokens, grid_sizes = 1*10*10 = 100
        grid_sizes = mx.array([[1, 10, 10]], dtype=mx.int32)
        x1 = mx.random.normal((b, 100, DIM))
        seq_lens = mx.array([100], dtype=mx.int32)
        out1 = attn(x1, grid_sizes, freqs, kv_cache=kv_cache, seq_lens=seq_lens)
        assert out1.shape == (b, 100, DIM)
        assert kv_cache["local_end_index"].item() == 100

        # Second call: 100 tokens (same grid), cache grows to 200
        x2 = mx.random.normal((b, 100, DIM))
        seq_lens = mx.array([100], dtype=mx.int32)
        out2 = attn(x2, grid_sizes, freqs, kv_cache=kv_cache, seq_lens=seq_lens)
        assert out2.shape == (b, 100, DIM)
        assert kv_cache["local_end_index"].item() == 200


class TestWanT2VCrossAttention:
    def test_output_shape(self):
        """Cross-attention expects context projected to DIM (1536)."""
        attn = WanT2VCrossAttention(DIM, NUM_HEADS)
        x = mx.random.normal((1, 100, DIM))
        context = mx.random.normal((1, 512, DIM))  # Already projected to DIM
        out = attn(x, context)
        assert out.shape == (1, 100, DIM)

    def test_with_cache(self):
        attn = WanT2VCrossAttention(DIM, NUM_HEADS)
        x = mx.random.normal((1, 100, DIM))
        context = mx.random.normal((1, 512, DIM))
        cache = {"k": None, "v": None, "is_init": False}

        out = attn(x, context, crossattn_cache=cache)
        assert out.shape == (1, 100, DIM)
        assert cache["is_init"] is True

    def test_cached_reuse(self):
        """Once cached, subsequent calls should use cached K,V."""
        attn = WanT2VCrossAttention(DIM, NUM_HEADS)
        x = mx.random.normal((1, 100, DIM))
        context = mx.random.normal((1, 512, DIM))
        cache = {"k": None, "v": None, "is_init": False}

        out1 = attn(x, context, crossattn_cache=cache)
        out2 = attn(x, context, crossattn_cache=cache)
        assert out1.shape == out2.shape


class TestWanFeedForward:
    def test_output_shape(self):
        ffn = WanFeedForward(DIM, FFN_DIM)
        x = mx.random.normal((1, 100, DIM))
        out = ffn(x)
        assert out.shape == (1, 100, DIM)

    def test_no_nan(self):
        ffn = WanFeedForward(DIM, FFN_DIM)
        x = mx.random.normal((1, 100, DIM))
        assert_finite(ffn(x), "FFN output")
