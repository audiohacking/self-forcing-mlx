"""MLX port of model/ode_regression.py — ODE Regression.

Trains the generator to match ODE solution trajectories.
Uses precomputed ODE sampling trajectories as training targets.
"""

import mlx.core as mx
import mlx.nn as nn

from ..model import WanModel
from sforcing.trainers.diffusion import _get_timestep


class ODERegression(nn.Module):
    """ODE Regression training module.

    Trains the generator by regressing towards precomputed ODE
    solution trajectories at sampled intermediate timesteps.
    """

    def __init__(
        self,
        generator: WanModel,
        denoising_step_list=None,
        num_frame_per_block: int = 3,
        independent_first_frame: bool = False,
        timestep_shift: float = 1.0,
    ):
        super().__init__()
        self.generator = generator
        self.denoising_step_list = denoising_step_list or [1000, 750, 500, 250]
        self.num_frame_per_block = num_frame_per_block
        self.independent_first_frame = independent_first_frame
        self.timestep_shift = timestep_shift

    def loss_fn(
        self,
        ode_latent: mx.array,
        conditional_dict: dict,
        key: mx.array = None,
    ) -> mx.array:
        """Compute ODE regression loss.

        Args:
            ode_latent: ODE trajectories, shape (B, T, F, C, H, W)
                        where T = num_denoising_steps, ordered noisy→clean.
            conditional_dict: Dict with 'context' key.
            key: Random key.

        Returns:
            loss: Scalar regression loss.
        """
        if key is None:
            key = mx.random.key(0)

        key, subkey = mx.random.split(key)

        batch_size, num_steps, num_frames, c, h, w = ode_latent.shape

        # Target is the cleanest latent (last in trajectory)
        target_latent = ode_latent[:, -1]

        # Sample intermediate timestep for each frame
        index = _get_timestep(
            0, len(self.denoising_step_list),
            batch_size, num_frames,
            self.num_frame_per_block,
            independent_first_frame=self.independent_first_frame,
            uniform_timestep=False,
            key=subkey,
        )

        # Gather corresponding noisy latents
        index_expanded = index.reshape(batch_size, 1, num_frames, 1, 1, 1)
        index_expanded = mx.broadcast_to(
            index_expanded, (batch_size, 1, num_frames, c, h, w)
        )
        noisy_input = mx.take_along_axis(
            ode_latent, index_expanded, axis=1
        )[:, 0]

        # Corresponding timestep values
        timestep_vals = mx.array(
            [self.denoising_step_list[i] for i in index.flatten().tolist()],
            dtype=mx.float32,
        ).reshape(batch_size, num_frames)

        # Forward through generator
        context = conditional_dict.get("context")
        seq_len = num_frames * h * w

        _, pred = self.generator(noisy_input, timestep_vals, context, seq_len)
        if isinstance(pred, list):
            pred = mx.stack(pred).transpose(0, 2, 1, 3, 4)

        # Mask out timestep=0 (no noise added)
        mask = timestep_vals != 0

        loss = mx.mean((pred[mask] - target_latent[mask]) ** 2)

        return loss
