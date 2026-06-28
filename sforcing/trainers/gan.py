"""MLX port of model/gan.py — GAN training with discriminator.

Implements GAN-based training with a classifier/discriminator branch
on the fake score model, plus R1/R2 regularization.
"""

import mlx.core as mx
import mlx.nn as nn

from ..model import WanModel
from ..scheduler import FlowMatchScheduler
from sforcing.trainers.diffusion import _get_timestep


class GAN(nn.Module):
    """GAN training module with discriminator and distribution matching."""

    def __init__(
        self,
        generator: WanModel,
        real_score: WanModel,
        fake_score: WanModel,
        scheduler: FlowMatchScheduler,
        num_frame_per_block: int = 3,
        num_training_frames: int = 21,
        num_train_timestep: int = 1000,
        num_class: int = 1,
        min_step: float = 0.02,
        max_step: float = 0.98,
        real_guidance_scale: float = 5.0,
        fake_guidance_scale: float = 0.0,
        timestep_shift: float = 1.0,
        critic_timestep_shift: float = None,
        independent_first_frame: bool = False,
        same_step_across_blocks: bool = True,
        ts_schedule: bool = True,
        ts_schedule_max: bool = False,
        min_score_timestep: int = 0,
        gan_g_weight: float = 1e-2,
        gan_d_weight: float = 1e-2,
        r1_weight: float = 0.0,
        r2_weight: float = 0.0,
        r1_sigma: float = 0.01,
        r2_sigma: float = 0.01,
        relativistic_discriminator: bool = False,
        concat_time_embeddings: bool = False,
    ):
        super().__init__()
        self.generator = generator
        self.real_score = real_score
        self.fake_score = fake_score
        self.scheduler = scheduler
        self.num_frame_per_block = num_frame_per_block
        self.num_training_frames = num_training_frames
        self.num_train_timestep = num_train_timestep
        self.num_class = num_class
        self.min_step = int(min_step * num_train_timestep)
        self.max_step = int(max_step * num_train_timestep)
        self.real_guidance_scale = real_guidance_scale
        self.fake_guidance_scale = fake_guidance_scale
        self.timestep_shift = timestep_shift
        self.critic_timestep_shift = critic_timestep_shift or timestep_shift
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.ts_schedule = ts_schedule
        self.ts_schedule_max = ts_schedule_max
        self.min_score_timestep = min_score_timestep
        self.gan_g_weight = gan_g_weight
        self.gan_d_weight = gan_d_weight
        self.r1_weight = r1_weight
        self.r2_weight = r2_weight
        self.r1_sigma = r1_sigma
        self.r2_sigma = r2_sigma
        self.relativistic_discriminator = relativistic_discriminator
        self.concat_time_embeddings = concat_time_embeddings

    def _sample_critic_timestep(
        self,
        batch_size: int,
        num_frame: int,
        denoised_from: int = 0,
        denoised_to: int = 0,
        key: mx.array = None,
    ):
        """Sample timestep for critic with scheduling."""
        if key is None:
            key = mx.random.key(0)

        min_t = denoised_to if self.ts_schedule and denoised_to > 0 else self.min_score_timestep
        max_t = denoised_from if self.ts_schedule_max and denoised_from > 0 else self.num_train_timestep

        ts = _get_timestep(
            min_t, max_t,
            batch_size, num_frame,
            self.num_frame_per_block,
            independent_first_frame=self.independent_first_frame,
            uniform_timestep=True,
            key=key,
        ).astype(mx.float32)

        if self.critic_timestep_shift > 1:
            ts = self.critic_timestep_shift * (ts / 1000) / \
                (1 + (self.critic_timestep_shift - 1) * (ts / 1000)) * 1000
        return mx.clip(ts, self.min_step, self.max_step)

    def _run_generator(self, clean_latent, conditional_dict, key=None):
        """Backward simulation and single-step generator inference."""
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

    def generator_loss(
        self,
        clean_latent: mx.array,
        conditional_dict: dict,
        unconditional_dict: dict,
        key: mx.array = None,
    ):
        """Compute generator loss with GAN discriminator."""
        if key is None:
            key = mx.random.key(0)

        key, subkey1, subkey2 = mx.random.split(key, 3)

        # Step 1: Run generator
        pred_image, _, denoised_from, denoised_to = self._run_generator(
            clean_latent=clean_latent,
            conditional_dict=conditional_dict,
            key=subkey1,
        )

        # Step 2: Add noise to generated latents for critic
        batch_size, num_frame = pred_image.shape[:2]
        critic_ts = self._sample_critic_timestep(
            batch_size, num_frame, denoised_from, denoised_to, key=subkey2,
        )

        critic_noise = mx.random.normal(pred_image.shape, key=subkey2)
        noisy_fake = self.scheduler.add_noise(
            pred_image.reshape(-1, *pred_image.shape[2:]),
            critic_noise.reshape(-1, *critic_noise.shape[2:]),
            critic_ts.reshape(-1),
        ).reshape(pred_image.shape)

        # Step 3: Add noise to real latents
        real_latent = clean_latent
        critic_noise_real = mx.random.normal(real_latent.shape, key=subkey2)
        noisy_real = self.scheduler.add_noise(
            real_latent.reshape(-1, *real_latent.shape[2:]),
            critic_noise_real.reshape(-1, *critic_noise_real.shape[2:]),
            critic_ts.reshape(-1),
        ).reshape(real_latent.shape)

        # Step 4: Run discriminator on both
        context = conditional_dict.get("context")
        seq_len = pred_image.shape[1] * pred_image.shape[3] * pred_image.shape[4]

        # Concatenate fake and real for batch discriminator
        combined = mx.concatenate([noisy_fake, noisy_real], axis=0)
        combined_ts = mx.concatenate([critic_ts, critic_ts], axis=0)

        # Simplified: use fake_score as discriminator (x0 prediction output)
        _, combined_pred = self.fake_score(combined, combined_ts, context, seq_len)
        if isinstance(combined_pred, list):
            combined_pred = mx.stack(combined_pred).transpose(0, 2, 1, 3, 4)

        fake_pred, real_pred = combined_pred.split(2, axis=0)

        # GAN generator loss
        if not self.relativistic_discriminator:
            gan_loss = mx.mean(nn.softplus(-fake_pred)) * self.gan_g_weight
        else:
            relative_fake = fake_pred - real_pred
            gan_loss = mx.mean(nn.softplus(-relative_fake)) * self.gan_g_weight

        return gan_loss

    def critic_loss(
        self,
        clean_latent: mx.array,
        real_latent: mx.array,
        conditional_dict: dict,
        unconditional_dict: dict,
        key: mx.array = None,
    ):
        """Compute critic loss with GAN discriminator and R1/R2 regularization."""
        if key is None:
            key = mx.random.key(0)

        key, subkey1, subkey2, subkey3 = mx.random.split(key, 4)

        # Step 1: Run generator (no grad)
        generated_image, _, denoised_from, denoised_to = self._run_generator(
            clean_latent=clean_latent,
            conditional_dict=conditional_dict,
            key=subkey1,
        )

        # Step 2: Sample critic timestep
        batch_size, num_frame = generated_image.shape[:2]
        critic_ts = self._sample_critic_timestep(
            batch_size, num_frame, denoised_from, denoised_to, key=subkey2,
        )

        # Step 3: Add noise to fake and real
        critic_noise = mx.random.normal(generated_image.shape, key=subkey2)
        noisy_fake = self.scheduler.add_noise(
            generated_image.reshape(-1, *generated_image.shape[2:]),
            critic_noise.reshape(-1, *critic_noise.shape[2:]),
            critic_ts.reshape(-1),
        ).reshape(generated_image.shape)

        noisy_real = self.scheduler.add_noise(
            real_latent.reshape(-1, *real_latent.shape[2:]),
            critic_noise.reshape(-1, *critic_noise.shape[2:]),
            critic_ts.reshape(-1),
        ).reshape(real_latent.shape)

        # Step 4: Run discriminator
        context = conditional_dict.get("context")
        seq_len = generated_image.shape[1] * generated_image.shape[3] * generated_image.shape[4]

        combined = mx.concatenate([noisy_fake, noisy_real], axis=0)
        combined_ts = mx.concatenate([critic_ts, critic_ts], axis=0)

        _, combined_pred = self.fake_score(combined, combined_ts, context, seq_len)
        if isinstance(combined_pred, list):
            combined_pred = mx.stack(combined_pred).transpose(0, 2, 1, 3, 4)

        fake_pred, real_pred = combined_pred.split(2, axis=0)

        # GAN discriminator loss
        if not self.relativistic_discriminator:
            gan_d_loss = (
                mx.mean(nn.softplus(-real_pred)) +
                mx.mean(nn.softplus(fake_pred))
            )
        else:
            relative_real = real_pred - fake_pred
            gan_d_loss = mx.mean(nn.softplus(-relative_real))
        gan_d_loss = gan_d_loss * self.gan_d_weight

        # R1 regularization
        if self.r1_weight > 0:
            epsilon_r1 = self.r1_sigma * mx.random.normal(noisy_real.shape, key=subkey3)
            noisy_real_perturbed = noisy_real + epsilon_r1
            _, real_pred_perturbed = self.fake_score(
                noisy_real_perturbed, critic_ts, context, seq_len
            )
            if isinstance(real_pred_perturbed, list):
                real_pred_perturbed = mx.stack(real_pred_perturbed).transpose(0, 2, 1, 3, 4)

            r1_grad = (real_pred_perturbed - real_pred) / self.r1_sigma
            r1_loss = self.r1_weight * mx.mean(r1_grad ** 2)
        else:
            r1_loss = mx.zeros((1,))

        # R2 regularization
        if self.r2_weight > 0:
            epsilon_r2 = self.r2_sigma * mx.random.normal(noisy_fake.shape, key=subkey3)
            noisy_fake_perturbed = noisy_fake + epsilon_r2
            _, fake_pred_perturbed = self.fake_score(
                noisy_fake_perturbed, critic_ts, context, seq_len
            )
            if isinstance(fake_pred_perturbed, list):
                fake_pred_perturbed = mx.stack(fake_pred_perturbed).transpose(0, 2, 1, 3, 4)

            r2_grad = (fake_pred_perturbed - fake_pred) / self.r2_sigma
            r2_loss = self.r2_weight * mx.mean(r2_grad ** 2)
        else:
            r2_loss = mx.zeros((1,))

        log_dict = {
            "critic_timestep": critic_ts,
            "real_logit": mx.mean(real_pred),
            "fake_logit": mx.mean(fake_pred),
        }

        return (gan_d_loss, r1_loss, r2_loss), log_dict
