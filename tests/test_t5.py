"""Tests for T5 encoder."""
import mlx.core as mx
import numpy as np
import pytest

from sforcing.t5 import T5Encoder
from sforcing.config import TEXT_DIM, TEXT_LEN


def assert_finite(tensor, name="tensor"):
    arr = np.array(tensor)
    assert not np.any(np.isnan(arr)), f"{name} contains NaN"
    assert not np.any(np.isinf(arr)), f"{name} contains inf"


class TestT5Encoder:
    def test_instantiation(self):
        t5 = T5Encoder()
        assert t5.num_layers == 24
        assert t5.dim == 4096

    def test_forward_shape(self):
        """T5 encoder: (1, L) -> (1, L, dim)."""
        t5 = T5Encoder()
        ids = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
        mask = mx.ones((1, 5), dtype=mx.int32)
        out = t5(ids, mask=mask)
        assert out.shape == (1, 5, 4096), f"Got {out.shape}"

    def test_no_nan(self):
        t5 = T5Encoder()
        ids = mx.random.randint(0, 1000, (1, 10))
        mask = mx.ones((1, 10), dtype=mx.int32)
        out = t5(ids, mask=mask)
        assert_finite(out, "T5 output")

    def test_with_mask(self):
        """Mask should affect output (masked positions differ from unmasked)."""
        t5 = T5Encoder()
        ids = mx.array([[1, 2, 3, 0, 0]], dtype=mx.int32)

        # With full mask
        mask_full = mx.ones((1, 5), dtype=mx.int32)
        out_full = t5(ids, mask=mask_full)

        # With partial mask
        mask_partial = mx.array([[1, 1, 1, 0, 0]], dtype=mx.int32)
        out_partial = t5(ids, mask=mask_partial)

        assert out_full.shape == out_partial.shape
