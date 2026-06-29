"""Tests for WanModel with random weights."""
import mlx.core as mx
import numpy as np
import pytest

from sforcing.model import WanModel
from sforcing.config import (
    DIM, FFN_DIM, NUM_HEADS, NUM_LAYERS, FREQ_DIM,
    TEXT_DIM, TEXT_LEN, IN_DIM, OUT_DIM, PATCH_SIZE, EPS,
)


def assert_finite(tensor, name="tensor"):
    arr = np.array(tensor)
    assert not np.any(np.isnan(arr)), f"{name} contains NaN"
    assert not np.any(np.isinf(arr)), f"{name} contains inf"


class TestWanModel:
    def test_instantiation(self):
        model = WanModel()
        assert model.num_layers == NUM_LAYERS
        assert model.dim == DIM

    def test_forward_shape_single_frame(self):
        """Model returns list of (out_dim, F*ps, H*ps, W*ps) tensors (no batch dim)."""
        model = WanModel()
        x = [mx.random.normal((IN_DIM, 1, 60, 104))]
        t = mx.array([500], dtype=mx.float32)
        context = mx.random.normal((1, TEXT_LEN, TEXT_DIM))
        outputs = model(x, t, context, seq_len=32760)
        assert len(outputs) == 1
        # unpatchify returns (out_dim, F*ps_t, H*ps_h, W*ps_w) without batch dim
        assert outputs[0].shape[0] == OUT_DIM, f"Expected {OUT_DIM} channels, got {outputs[0].shape[0]}"
        assert outputs[0].shape[1] == 1 * PATCH_SIZE[0], f"Expected F dimension, got {outputs[0].shape[1]}"

    def test_forward_shape_multi_frame(self):
        model = WanModel()
        x = [mx.random.normal((IN_DIM, 3, 60, 104))]
        t = mx.array([500], dtype=mx.float32)
        context = mx.random.normal((1, TEXT_LEN, TEXT_DIM))
        outputs = model(x, t, context, seq_len=32760)
        assert len(outputs) == 1
        assert outputs[0].shape[1] == 3 * PATCH_SIZE[0], f"Expected 3 frames, got {outputs[0].shape[1]}"

    def test_no_nan(self):
        model = WanModel()
        x = [mx.random.normal((IN_DIM, 1, 60, 104))]
        t = mx.array([500], dtype=mx.float32)
        context = mx.random.normal((1, TEXT_LEN, TEXT_DIM))
        outputs = model(x, t, context, seq_len=32760)
        assert_finite(outputs[0], "model output")

    def test_batch_independent(self):
        """Two batch items should produce independent outputs."""
        model = WanModel()
        x = [
            mx.random.normal((IN_DIM, 1, 60, 104)),
            mx.random.normal((IN_DIM, 1, 60, 104)),
        ]
        t = mx.array([500, 500], dtype=mx.float32)
        context = mx.random.normal((2, TEXT_LEN, TEXT_DIM))
        outputs = model(x, t, context, seq_len=32760)
        assert len(outputs) == 2
        assert outputs[0].shape == outputs[1].shape

    def test_with_kv_caches(self):
        """Model forward with KV caches produces same shape."""
        model = WanModel()
        b, n_layers = 1, NUM_LAYERS
        n_heads = NUM_HEADS
        head_dim = DIM // NUM_HEADS

        kv_caches = []
        crossattn_caches = []
        for _ in range(n_layers):
            kv_caches.append({
                "k": mx.zeros((b, 32760, n_heads, head_dim)),
                "v": mx.zeros((b, 32760, n_heads, head_dim)),
                "global_end_index": mx.array([0], dtype=mx.int32),
                "local_end_index": mx.array([0], dtype=mx.int32),
            })
            crossattn_caches.append({
                "k": mx.zeros((b, TEXT_LEN, n_heads, head_dim)),
                "v": mx.zeros((b, TEXT_LEN, n_heads, head_dim)),
                "is_init": False,
            })

        x = [mx.random.normal((IN_DIM, 1, 60, 104))]
        t = mx.array([500], dtype=mx.float32)
        context = mx.random.normal((1, TEXT_LEN, TEXT_DIM))

        outputs = model(x, t, context, seq_len=32760,
                        kv_caches=kv_caches, crossattn_caches=crossattn_caches)
        assert len(outputs) == 1
        assert_finite(outputs[0], "model output with KV caches")
