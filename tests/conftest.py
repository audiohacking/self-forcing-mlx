"""Shared fixtures for MLX unit tests."""
import sys
sys.path.insert(0, '/Users/moysa/Documents/git/Self-Forcing')

import mlx.core as mx
import numpy as np
import pytest


@pytest.fixture
def dtype():
    return mx.float32


@pytest.fixture
def batch_size():
    return 1


@pytest.fixture
def num_heads():
    return 12


@pytest.fixture
def head_dim():
    return 128


@pytest.fixture
def dim():
    return 1536


@pytest.fixture
def ffn_dim():
    return 8960


@pytest.fixture
def seq_len():
    return 1560  # Typical for 1 frame


def assert_finite(tensor, name="tensor"):
    """Assert no NaN or inf values."""
    arr = np.array(tensor)
    assert not np.any(np.isnan(arr)), f"{name} contains NaN"
    assert not np.any(np.isinf(arr)), f"{name} contains inf"


def assert_shape(tensor, expected, name="tensor"):
    """Assert tensor shape matches expected."""
    assert tensor.shape == expected, \
        f"{name} shape {tensor.shape} != expected {expected}"
