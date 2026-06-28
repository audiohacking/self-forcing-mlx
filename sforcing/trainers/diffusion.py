"""MLX port of model/diffusion.py — CausalDiffusion.

Computes the standard diffusion denoising loss for training the
causal video generator with flow matching.
"""

import mlx.core as mx
import mlx.nn as nn

from ..config import DIM, FFN_DIM, NUM_HEADS, NUM_LAYERS, IN_DIM, OUT_DIM, PATCH_SIZE
from ..model import WanModel
from ..scheduler import FlowMatchScheduler
from ..t5 import T5Encoder


def _get_timestep(
    min_timestep: int,
    max_timestep: int,
    batch_size: int,
    num_frame: int,
    num_frame_per_block: int,
    independent_first_frame: bool = False,
    uniform_timestep: bool = False,
    key: mx.array = None,
) -> mx.array:
    """Randomly generate timesteps for training.

    Args:
        min_timestep: Minimum timestep (inclusive).
        max_timestep: Maximum timestep (exclusive).
        batch_size: Batch size.
        num_frame: Number of frames.
        num_frame_per_block: Frames per block.
        independent_first_frame: Whether first frame has independent timestep.
        uniform_timestep: Use same timestep for all frames.
        key: Random key.

    Returns:
        Timestep tensor, shape (batch_size, num_frame).
    """
    if key is None:
        key = mx.random.key(0)

    if uniform_timestep:
        timestep = mx.random.randint(
            min_timestep, max_timestep,
            (batch_size, 1),
            key=key,
        )
        timestep = mx.broadcast_to(timestep, (batch_size, num_frame))
        return timestep
    else:
        timestep = mx.random.randint(
            min_timestep, max_timestep,
            (batch_size, num_frame),
            key=key,
        )
        # Make noise level the same within every block
        if independent_first_frame:
            timestep_from_second = timestep[:, 1:]
            timestep_from_second = timestep_from_second.reshape(
                timestep_from_second.shape[0], -1, num_frame_per_block)
            timestep_from_second[:, :, 1:] = timestep_from_second[:, :, 0:1]
            timestep_from_second = timestep_from_second.reshape(
                timestep_from_second.shape[0], -1)
            timestep = mx.concatenate([timestep[:, 0:1], timestep_from_second], axis=1)
        else:
            timestep = timestep.reshape(
                timestep.shape[0], -1, num_frame_per_block)
            timestep[:, :, 1:] = timestep[:, :, 0:1]
            timestep = timestep.reshape(timestep.shape[0], -1)
        return timestep


class CausalDiffusion(nn.Module):
    """Causal diffusion training with flow matching loss.

    Trains the generator by adding noise to clean latents and
    predicting the flow (velocity) at the given timestep.
    """

    def __init__(
        self,
        generator: WanModel,
        scheduler: FlowMatchScheduler,
        num_frame_per_block: int = 3,
        num_train_timestep: int = 1000,
        min_step: float = 0.02,
        max_step: float = 0.98,
        timestep_shift: float = 1.0,
        independent_first_frame: bool = False,
        teacher_forcing: bool = False,
        noise_augmentation_max_timestep: int = 0,
    ):
        super().__init__()
        self.generator = generator
        self.scheduler = scheduler
        self.num_frame_per_block = num_frame_per_block
        self.num_train_timestep = num_train_timestep
        self.min_step = int(min_step * num_train_timestep)
        self.max_step = int(max_step * num_train_timestep)
        self.timestep_shift = timestep_shift
        self.independent_first_frame = independent_first_frame
        self.teacher_forcing = teacher_forcing
        self.noise_augmentation_max_timestep = noise_augmentation_max_timestep

    def loss_fn(
        self,
        clean_latent: mx.array,
        conditional_dict: dict,
        key: mx.array = None,
    ) -> mx.array:
        """Compute flow matching denoising loss.

        Args:
            clean_latent: Clean latent tensor, shape (B, F, C, H, W).
            conditional_dict: Dict with 'context' key (text embeddings).
            key: Random key.

        Returns:
            Scalar loss.
        """
        if key is None:
            key = mx.random.key(0)

        key, subkey1, subkey2 = mx.random.split(key, 3)

        batch_size, num_frame = clean_latent.shape[:2]
        noise = mx.random.normal(clean_latent.shape, key=subkey1)

        # Sample timestep
        index = _get_timestep(
            0, self.num_train_timestep,
            batch_size, num_frame,
            self.num_frame_per_block,
            independent_first_frame=self.independent_first_frame,
            uniform_timestep=False,
            key=subkey2,
        )

        # Get scheduler timestep values
        timestep_vals = self.scheduler.timesteps[index]

        # Add noise
        noisy_latents = self.scheduler.add_noise(
            clean_latent.reshape(-1, *clean_latent.shape[2:]),
            noise.reshape(-1, *noise.shape[2:]),
            timestep_vals.reshape(-1),
        ).reshape(clean_latent.shape)

        # Training target: noise - clean_latent (flow matching)
        training_target = self.scheduler.training_target(clean_latent, noise, timestep_vals)

        # Noise augmentation for teacher forcing
        if self.noise_augmentation_max_timestep > 0:
            key, subkey3 = mx.random.split(key)
            index_clean_aug = _get_timestep(
                0, self.noise_augmentation_max_timestep,
                batch_size, num_frame,
                self.num_frame_per_block,
                independent_first_frame=self.independent_first_frame,
                uniform_timestep=False,
                key=subkey3,
            )
            timestep_clean_aug = self.scheduler.timesteps[index_clean_aug]
            clean_latent_aug = self.scheduler.add_noise(
                clean_latent.reshape(-1, *clean_latent.shape[2:]),
                noise.reshape(-1, *noise.shape[2:]),
                timestep_clean_aug.reshape(-1),
            ).reshape(clean_latent.shape)
        else:
            clean_latent_aug = clean_latent

        # Forward through generator
        context = conditional_dict.get("context")
        flow_pred = self.generator(
            noisy_latents, timestep_vals, context,
            seq_len=noisy_latents.shape[1] * noisy_latents.shape[3] * noisy_latents.shape[4],
        )

        # Handle list output
        if isinstance(flow_pred, list):
            flow_pred = mx.stack(flow_pred)
            flow_pred = flow_pred.transpose(0, 2, 1, 3, 4)

        # MSE loss
        loss = mx.mean((flow_pred - training_target) ** 2)
        return loss
