"""Tests for utility functions: sinusoidal embedding, RoPE, unpatchify."""
import mlx.core as mx
import numpy as np
import pytest

from sforcing.utils import (
    sinusoidal_embedding_1d,
    rope_params,
    rope_apply,
    rope_apply_causal,
    unpatchify,
)
from sforcing.config import PATCH_SIZE, OUT_DIM


class TestSinusoidalEmbedding:
    def test_output_shape(self):
        """sinusoidal_embedding_1d produces (B, freq_dim)."""
        emb = sinusoidal_embedding_1d(256, mx.array([0.0, 0.5, 1.0]))
        assert emb.shape == (3, 256), f"Expected (3, 256), got {emb.shape}"

    def test_no_nan(self):
        emb = sinusoidal_embedding_1d(256, mx.array([0.0, 0.5, 1.0]))
        arr = np.array(emb)
        assert not np.any(np.isnan(arr))
        assert not np.any(np.isinf(arr))

    def test_timestep_zero(self):
        """t=0 should give non-zero embedding (not all zeros)."""
        emb = sinusoidal_embedding_1d(256, mx.array([0.0]))
        arr = np.array(emb)
        # First element should be 1.0 (cos(0))
        assert abs(arr[0, 0] - 1.0) < 1e-5


class TestRoPEParams:
    def test_output_shape(self):
        """rope_params returns (max_seq_len, dim//2) as complex."""
        freqs = rope_params(1024, 64)
        assert freqs.shape == (1024, 32), f"Expected (1024, 32), got {freqs.shape}"

    def test_no_nan(self):
        freqs = rope_params(1024, 64)
        arr = np.array(freqs)
        assert not np.any(np.isnan(arr))


class TestRoPEApply:
    def test_output_shape(self):
        """rope_apply preserves input shape.

        For head_dim=128, freqs must have shape (max_seq_len, head_dim//2) = (1024, 64).
        rope_params(1024, 128) returns (1024, 64) since it returns dim//2 elements.
        """
        x = mx.random.normal((1, 100, 12, 128))
        grid_sizes = mx.array([[1, 10, 10]], dtype=mx.int32)
        freqs = rope_params(1024, 128)  # Returns (1024, 64) for head_dim=128
        out = rope_apply(x, grid_sizes, freqs)
        assert out.shape == (1, 100, 12, 128)

    def test_no_nan(self):
        x = mx.random.normal((1, 100, 12, 128))
        grid_sizes = mx.array([[1, 10, 10]], dtype=mx.int32)
        freqs = rope_params(1024, 128)
        out = rope_apply(x, grid_sizes, freqs)
        assert_finite(out, "rope_apply output")


class TestRoPEApplyCausal:
    def test_output_shape(self):
        x = mx.random.normal((1, 100, 12, 128))
        grid_sizes = mx.array([[1, 10, 10]], dtype=mx.int32)
        freqs = rope_params(1024, 128)
        out = rope_apply_causal(x, grid_sizes, freqs)
        assert out.shape == (1, 100, 12, 128)


class TestUnpatchify:
    def test_output_shape(self):
        """unpatchify converts (B, L, out_channels) -> list of (out_dim, F, H*ps, W*ps).

        Note: output has no batch dimension (each batch item returned separately).
        """
        ft, fh, fw = 1, 30, 52
        seq_len = ft * fh * fw
        out_channels = PATCH_SIZE[0] * PATCH_SIZE[1] * PATCH_SIZE[2] * OUT_DIM
        x = mx.random.normal((1, seq_len, out_channels))
        grid_sizes = [(ft, fh, fw)]

        outputs = unpatchify(x, grid_sizes, PATCH_SIZE, OUT_DIM)
        assert len(outputs) == 1
        expected = (OUT_DIM, ft * PATCH_SIZE[0], fh * PATCH_SIZE[1], fw * PATCH_SIZE[2])
        assert outputs[0].shape == expected, f"Expected {expected}, got {outputs[0].shape}"

    def test_no_nan(self):
        ft, fh, fw = 1, 30, 52
        seq_len = ft * fh * fw
        out_channels = PATCH_SIZE[0] * PATCH_SIZE[1] * PATCH_SIZE[2] * OUT_DIM
        x = mx.random.normal((1, seq_len, out_channels))
        outputs = unpatchify(x, [(ft, fh, fw)], PATCH_SIZE, OUT_DIM)
        assert_finite(outputs[0], "unpatchify output")


def assert_finite(tensor, name="tensor"):
    arr = np.array(tensor)
    assert not np.any(np.isnan(arr)), f"{name} contains NaN"
    assert not np.any(np.isinf(arr)), f"{name} contains inf"
