"""Tests for FlowMatchScheduler."""
import mlx.core as mx
import numpy as np
import pytest

from sforcing.scheduler import FlowMatchScheduler
from sforcing.config import NUM_TRAIN_TIMESTEPS


def assert_finite(tensor, name="tensor"):
    arr = np.array(tensor)
    assert not np.any(np.isnan(arr)), f"{name} contains NaN"
    assert not np.any(np.isinf(arr)), f"{name} contains inf"


class TestFlowMatchScheduler:
    def test_instantiation(self):
        scheduler = FlowMatchScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS)
        assert scheduler is not None

    def test_set_timesteps(self):
        scheduler = FlowMatchScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS)
        scheduler.set_timesteps(10)
        assert len(scheduler.timesteps) == 10

    def test_step_shape(self):
        """Scheduler step preserves latent shape."""
        scheduler = FlowMatchScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS)
        scheduler.set_timesteps(4)

        latents = mx.random.normal((1, 3, 16, 60, 104))
        flow_pred = mx.random.normal((1, 3, 16, 60, 104))
        timestep = mx.array([1000], dtype=mx.float32)

        out = scheduler.step(flow_pred, timestep, latents)
        assert out.shape == latents.shape, f"Expected {latents.shape}, got {out.shape}"

    def test_no_nan(self):
        scheduler = FlowMatchScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS)
        scheduler.set_timesteps(4)

        latents = mx.random.normal((1, 3, 16, 60, 104))
        flow_pred = mx.random.normal((1, 3, 16, 60, 104))
        timestep = mx.array([1000], dtype=mx.float32)

        out = scheduler.step(flow_pred, timestep, latents)
        assert_finite(out, "scheduler output")

    def test_shift(self):
        """Scheduler with shift != 1.0 should produce different timesteps."""
        s1 = FlowMatchScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS, shift=1.0)
        s2 = FlowMatchScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS, shift=5.0)
        s1.set_timesteps(10)
        s2.set_timesteps(10)
        # Different shift should produce different timestep values
        assert not np.allclose(np.array(s1.timesteps), np.array(s2.timesteps))
