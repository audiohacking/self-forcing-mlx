"""Tests for VAE decoder."""
import mlx.core as mx
import numpy as np
import pytest

from sforcing.vae import WanVAE
from sforcing.config import IN_DIM


def assert_finite(tensor, name="tensor"):
    arr = np.array(tensor)
    assert not np.any(np.isnan(arr)), f"{name} contains NaN"
    assert not np.any(np.isinf(arr)), f"{name} contains inf"


class TestWanVAE:
    def test_instantiation(self):
        vae = WanVAE(z_dim=IN_DIM, dim=96, dim_mult=[1, 2, 4, 4])
        assert vae is not None

    def test_decode_shape_single_frame(self):
        """VAE decode: (1, C, 1, H, W) -> (1, 3, 1, H*8, W*8)."""
        vae = WanVAE(z_dim=IN_DIM, dim=96, dim_mult=[1, 2, 4, 4])
        z = mx.random.normal((1, IN_DIM, 1, 60, 104))
        out = vae.decode(z)
        assert out.shape == (1, 3, 1, 480, 832), f"Got {out.shape}"

    def test_decode_shape_multi_frame(self):
        """VAE decode: (1, C, 3, H, W) -> (1, 3, 3, H*8, W*8)."""
        vae = WanVAE(z_dim=IN_DIM, dim=96, dim_mult=[1, 2, 4, 4])
        z = mx.random.normal((1, IN_DIM, 3, 60, 104))
        out = vae.decode(z)
        assert out.shape == (1, 3, 3, 480, 832), f"Got {out.shape}"

    def test_no_nan(self):
        vae = WanVAE(z_dim=IN_DIM, dim=96, dim_mult=[1, 2, 4, 4])
        z = mx.random.normal((1, IN_DIM, 1, 60, 104))
        out = vae.decode(z)
        assert_finite(out, "VAE decode output")

    def test_output_range(self):
        """VAE output should be in [-1, 1] range."""
        vae = WanVAE(z_dim=IN_DIM, dim=96, dim_mult=[1, 2, 4, 4])
        z = mx.random.normal((1, IN_DIM, 1, 60, 104))
        out = vae.decode(z)
        arr = np.array(out)
        assert np.all(arr >= -1) and np.all(arr <= 1), \
            f"Output values outside [-1, 1]: [{arr.min()}, {arr.max()}]"
