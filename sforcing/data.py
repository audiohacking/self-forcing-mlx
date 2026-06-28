"""MLX data loading utilities for Self-Forcing training.

Port of utils/dataset.py, utils/lmdb.py, utils/misc.py to MLX/numpy.
Provides LMDB-based dataset loading for training with MLX arrays.
"""

import os
import random
from typing import Any, Dict, Iterator, List, Optional, Tuple

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """Set random seed for reproducibility across Python, numpy, and MLX.

    Args:
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


# ---------------------------------------------------------------------------
# LMDB helpers
# ---------------------------------------------------------------------------

def get_array_shape_from_lmdb(env, array_name: str) -> Tuple[int, ...]:
    """Get the shape of a stored array from LMDB metadata.

    Args:
        env: LMDB environment.
        array_name: Name of the array.

    Returns:
        Tuple of dimensions.
    """
    with env.begin() as txn:
        shape_str = txn.get(f"{array_name}_shape".encode()).decode()
        return tuple(map(int, shape_str.split()))


def retrieve_row_from_lmdb(
    lmdb_env, array_name: str, dtype, row_index: int,
    shape: Optional[Tuple[int, ...]] = None,
) -> Any:
    """Retrieve a specific row from an array stored in LMDB.

    Args:
        lmdb_env: LMDB environment.
        array_name: Name of the array.
        dtype: Data type (numpy dtype or str).
        row_index: Row index to retrieve.
        shape: Optional shape to reshape the data.

    Returns:
        Retrieved row (numpy array or string).
    """
    data_key = f'{array_name}_{row_index}_data'.encode()

    with lmdb_env.begin() as txn:
        row_bytes = txn.get(data_key)

    if dtype == str:
        return row_bytes.decode()
    else:
        array = np.frombuffer(row_bytes, dtype=dtype)
        if shape is not None and len(shape) > 0:
            array = array.reshape(shape)
        return array


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class TextDataset:
    """Simple text prompt dataset from a file.

    Loads prompts line-by-line from a text file, with optional
    extended prompts from a second file.
    """

    def __init__(
        self,
        prompt_path: str,
        extended_prompt_path: Optional[str] = None,
    ):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self) -> int:
        return len(self.prompt_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        batch = {"prompts": self.prompt_list[idx], "idx": idx}
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class ODERegressionLMDBDataset:
    """LMDB dataset for ODE regression training.

    Each entry contains precomputed ODE trajectories (latents from
    pure noise to clean image) and corresponding text prompts.
    Returns MLX arrays for direct use in training.
    """

    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        import lmdb
        self.env = lmdb.open(
            data_path, readonly=True,
            lock=False, readahead=False, meminit=False,
        )
        self.latents_shape = get_array_shape_from_lmdb(self.env, "latents")
        self.max_pair = max_pair

    def __len__(self) -> int:
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a training sample.

        Returns:
            dict with:
                - prompts: Text prompt string.
                - ode_latent: MLX array of shape
                  (num_denoising_steps, num_frames, C, H, W),
                  ordered from noisy to clean.
        """
        latents = retrieve_row_from_lmdb(
            self.env, "latents", np.float16, idx,
            shape=self.latents_shape[1:],
        )
        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(self.env, "prompts", str, idx)

        return {
            "prompts": prompts,
            "ode_latent": mx.array(latents.astype(np.float32)),
        }


class ShardingLMDBDataset:
    """Sharded LMDB dataset for large-scale training.

    Supports multiple LMDB shards that are indexed contiguously.
    Returns MLX arrays for direct use in training.
    """

    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        import lmdb

        self.envs: List = []
        self.index: List[Tuple[int, int]] = []

        for fname in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, fname)
            env = lmdb.open(
                path, readonly=True,
                lock=False, readahead=False, meminit=False,
            )
            self.envs.append(env)

        self.latents_shape = [None] * len(self.envs)
        for shard_id, env in enumerate(self.envs):
            self.latents_shape[shard_id] = get_array_shape_from_lmdb(
                env, "latents",
            )
            for local_i in range(self.latents_shape[shard_id][0]):
                self.index.append((shard_id, local_i))

        self.max_pair = max_pair

    def __len__(self) -> int:
        return min(len(self.index), self.max_pair)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a training sample from the appropriate shard.

        Returns:
            dict with:
                - prompts: Text prompt string.
                - ode_latent: MLX array of shape
                  (num_denoising_steps, num_frames, C, H, W).
        """
        shard_id, local_idx = self.index[idx]

        latents = retrieve_row_from_lmdb(
            self.envs[shard_id], "latents", np.float16, local_idx,
            shape=self.latents_shape[shard_id][1:],
        )
        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.envs[shard_id], "prompts", str, local_idx,
        )

        return {
            "prompts": prompts,
            "ode_latent": mx.array(latents.astype(np.float32)),
        }


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def cycle(dl: Iterator) -> Iterator:
    """Cycle through a dataloader indefinitely."""
    while True:
        for data in dl:
            yield data


def merge_dict_list(dict_list: List[Dict]) -> Dict:
    """Merge a list of dicts by concatenating array values.

    Args:
        dict_list: List of dicts with same keys.

    Returns:
        Single dict with concatenated values.
    """
    if len(dict_list) == 1:
        return dict_list[0]

    merged = {}
    for k, v in dict_list[0].items():
        if isinstance(v, mx.array):
            if v.ndim == 0:
                merged[k] = mx.stack([d[k] for d in dict_list], axis=0)
            else:
                merged[k] = mx.concatenate([d[k] for d in dict_list], axis=0)
        else:
            merged[k] = v
    return merged
