"""MLX port of model/sid.py — SiD (Score identity Distillation).

Score identity distillation training using the difference between
real and fake score predictions as the training signal.
"""

import mlx.core as mx
import mlx.nn as nn

from ..model import WanModel
from ..scheduler import FlowMatchScheduler
from sforcing.trainers.diffusion import _get_timestep


class SiD(nn.Module):
    """Score identity Distillation training module.

    Uses the difference between real and fake score model predictions
    as a training signal for the generator.
    """

    def __init__(
        self,
        generator: WanModel,
        real_score: WanModel,
        fake_score: WanModel,
        scheduler: FlowMatchScheduler,
        num_frame_per_block: int = 3,
        num_train_timestep: int = 1000,
        min_step: float = 0.02,
        max_step: float = 0.98,
        real_guidance_scale: float = 5.0,
        timestep_shift: float = 1.0,
        sid_alpha: float = 1.0,
        ts_schedule: bool = True,
        ts_schedule_max: bool = False,
        min_score_timestep: int = 0,
    ):
        super().__init__()
        self.generator = generator
        self.real_score = real_score
        self.fake_score = fake_score
        self.scheduler = scheduler
        self.num_frame_per_block = num_frame_per_block
        self.num_train_timestep = num_train_timestep
        self.min_step = int(min_step * num_train_timestep)
        self.max_step = int(max_step * num_train_timestep)
        self.real_guidance_scale = real_guidance_scale
        self.timestep_shift = timestep_shift
        self.sid_alpha = sid_alpha
        self.ts_schedule = ts_schedule
        self.ts_schedule_max = ts_schedule_max
        self.min_score_timestep = min_score_timestep

    def distribution_matching_loss(
        self,
        image_or_video: mx.array,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: mx.array = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
        key: mx.array = None,
    ):
        """Compute SiD loss.

        SiD loss: (pred_real - pred_fake) * ((pred_real - x) - alpha * (pred_real - pred_fake))

        Args:
            image_or_video: Generated latent, shape (B, F, C, H, W).
            conditional_dict: Conditional text embeddings.
            unconditional_dict: Unconditional text embeddings.
            gradient_mask: Optional boolean mask (unused in SiD).
            denoised_timestep_from: Starting timestep for scheduling.
            denoised_timestep_to: Ending timestep for scheduling.
            key: Random key.

        Returns:
            loss: Scalar SiD loss.
            log_dict: Dict with debug info.
        """
        if key is None:
            key = mx.random.key(0)

        key, subkey1, subkey2 = mx.random.split(key, 3)

        batch_size, num_frame = image_or_video.shape[:2]

        # Sample timestep with scheduling
        min_t = denoised_timestep_to if self.ts_schedule and denoised_timestep_to > 0 else self.min_score_timestep
        max_t = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from > 0 else self.num_train_timestep

        timestep = _get_timestep(
            min_t, max_t,
            batch_size, num_frame,
            self.num_frame_per_block,
            uniform_timestep=True,
            key=subkey1,
        ).astype(mx.float32)

        # Apply shift
        if self.timestep_shift > 1:
            timestep = self.timestep_shift * (timestep / 1000) / \
                (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
        timestep = mx.clip(timestep, self.min_step, self.max_step)

        # Add noise
        noise = mx.random.normal(image_or_video.shape, key=subkey2)
        noisy_latent = self.scheduler.add_noise(
            image_or_video.reshape(-1, *image_or_video.shape[2:]),
            noise.reshape(-1, *noise.shape[2:]),
            timestep.reshape(-1),
        ).reshape(image_or_video.shape)

        context_cond = conditional_dict.get("context")
        context_uncond = unconditional_dict.get("context")
        seq_len = noisy_latent.shape[1] * noisy_latent.shape[3] * noisy_latent.shape[4]

        # Fake score
        _, pred_fake = self.fake_score(noisy_latent, timestep, context_cond, seq_len)
        if isinstance(pred_fake, list):
            pred_fake = mx.stack(pred_fake).transpose(0, 2, 1, 3, 4)

        # Real score with CFG
        _, pred_real_cond = self.real_score(noisy_latent, timestep, context_cond, seq_len)
        if isinstance(pred_real_cond, list):
            pred_real_cond = mx.stack(pred_real_cond).transpose(0, 2, 1, 3, 4)

        _, pred_real_uncond = self.real_score(noisy_latent, timestep, context_uncond, seq_len)
        if isinstance(pred_real_uncond, list):
            pred_real_uncond = mx.stack(pred_real_uncond).transpose(0, 2, 1, 3, 4)

        pred_real = pred_real_cond + (pred_real_cond - pred_real_uncond) * self.real_guidance_scale

        # SiD loss
        diff_real_fake = pred_real - pred_fake
        diff_real_x = pred_real - image_or_video
        sid_loss = diff_real_fake * (diff_real_x - self.sid_alpha * diff_real_fake)

        # Normalizer
        p_real = image_or_video - pred_real
        normalizer = mx.mean(mx.abs(p_real), axis=list(range(1, p_real.ndim)), keepdims=True)
        sid_loss = sid_loss / (normalizer + 1e-8)
        sid_loss = mx.where(mx.isnan(sid_loss), mx.zeros_like(sid_loss), sid_loss)

        sid_loss = mx.mean(sid_loss)

        log_dict = {
            "timestep": timestep,
        }

        return sid_loss, log_dict

    def generator_loss(
        self,
        clean_latent: mx.array,
        conditional_dict: dict,
        unconditional_dict: dict,
        key: mx.array = None,
    ):
        """Compute generator loss: backward simulation + SiD loss."""
        if key is None:
            key = mx.random.key(0)

        # Run generator
        pred_image, _, denoised_from, denoised_to = self._run_generator(
            clean_latent=clean_latent,
            conditional_dict=conditional_dict,
            key=key,
        )

        # SiD loss
        sid_loss, log_dict = self.distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            denoised_timestep_from=denoised_from,
            denoised_timestep_to=denoised_to,
        )

        return sid_loss, log_dict

    def _run_generator(self, clean_latent, conditional_dict, key=None):
        """Backward simulation and generator inference."""
        if key is None:
            key = mx.random.key(0)

        key, subkey = mx.random.split(key)
        batch_size = clean_latent.shape[0]
        context = conditional_dict.get("context")
        seq_len = clean_latent.shape[1] * clean_latent.shape[3] * clean_latent.shape[4]

        noise = mx.random.normal(clean_latent.shape, key=subkey)
        denoising_steps = [1000, 750, 500, 250]
        step_idx = mx.random.randint(0, len(denoising_steps), (1,)).item()
        timestep_val = mx.array([denoising_steps[step_idx]], dtype=mx.float32)

        noisy_input = self.scheduler.add_noise(
            clean_latent.reshape(-1, *clean_latent.shape[2:]),
            noise.reshape(-1, *noise.shape[2:]),
            mx.broadcast_to(timestep_val, (batch_size * clean_latent.shape[1],)),
        ).reshape(clean_latent.shape)

        flow_pred = self.generator(noisy_input, timestep_val, context, seq_len)
        if isinstance(flow_pred, list):
            flow_pred = mx.stack(flow_pred).transpose(0, 2, 1, 3, 4)

        pred_image = noisy_input - timestep_val * flow_pred / self.num_train_timestep

        return pred_image, None, denoising_steps[step_idx], 0

    def critic_loss(
        self,
        clean_latent: mx.array,
        conditional_dict: dict,
        unconditional_dict: dict,
        key: mx.array = None,
    ):
        """Compute critic (fake score) denoising loss."""
        if key is None:
            key = mx.random.key(0)

        key, subkey1, subkey2 = mx.random.split(key, 3)

        generated_image, _, denoised_from, denoised_to = self._run_generator(
            clean_latent=clean_latent,
            conditional_dict=conditional_dict,
            key=subkey1,
        )

        batch_size, num_frame = generated_image.shape[:2]

        min_t = denoised_to if self.ts_schedule and denoised_to > 0 else self.min_score_timestep
        max_t = denoised_from if self.ts_schedule_max and denoised_from > 0 else self.num_train_timestep

        critic_ts = _get_timestep(
            min_t, max_t,
            batch_size, num_frame,
            self.num_frame_per_block,
            uniform_timestep=True,
            key=subkey2,
        ).astype(mx.float32)

        if self.timestep_shift > 1:
            critic_ts = self.timestep_shift * (critic_ts / 1000) / \
                (1 + (self.timestep_shift - 1) * (critic_ts / 1000)) * 1000
        critic_ts = mx.clip(critic_ts, self.min_step, self.max_step)

        critic_noise = mx.random.normal(generated_image.shape, key=subkey2)
        noisy_generated = self.scheduler.add_noise(
            generated_image.reshape(-1, *generated_image.shape[2:]),
            critic_noise.reshape(-1, *critic_noise.shape[2:]),
            critic_ts.reshape(-1),
        ).reshape(generated_image.shape)

        context = conditional_dict.get("context")
        seq_len = generated_image.shape[1] * generated_image.shape[3] * generated_image.shape[4]
        _, pred_fake = self.fake_score(
            noisy_generated, critic_ts, context, seq_len
        )
        if isinstance(pred_fake, list):
            pred_fake = mx.stack(pred_fake).transpose(0, 2, 1, 3, 4)

        loss = mx.mean((pred_fake - generated_image) ** 2)
        return loss, {"critic_timestep": critic_ts}
