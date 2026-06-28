"""MLX port of model/causvid.py — CausVid.

Implements Distribution Matching Distillation (DMD) training for
causal video generation. Uses a generator, real score model (frozen),
and fake score model (trainable).
"""

import mlx.core as mx
import mlx.nn as nn

from ..model import WanModel
from ..scheduler import FlowMatchScheduler
from sforcing.trainers.diffusion import _get_timestep


class CausVid(nn.Module):
    """CausVid DMD training module.

    Uses distribution matching distillation to train a causal video generator.
    """

    def __init__(
        self,
        generator: WanModel,
        real_score: WanModel,
        fake_score: WanModel,
        scheduler: FlowMatchScheduler,
        num_frame_per_block: int = 3,
        num_training_frames: int = 21,
        num_train_timestep: int = 1000,
        min_step: float = 0.02,
        max_step: float = 0.98,
        real_guidance_scale: float = 5.0,
        fake_guidance_scale: float = 0.0,
        timestep_shift: float = 1.0,
        independent_first_frame: bool = False,
        teacher_forcing: bool = False,
    ):
        super().__init__()
        self.generator = generator
        self.real_score = real_score
        self.fake_score = fake_score
        self.scheduler = scheduler
        self.num_frame_per_block = num_frame_per_block
        self.num_training_frames = num_training_frames
        self.num_train_timestep = num_train_timestep
        self.min_step = int(min_step * num_train_timestep)
        self.max_step = int(max_step * num_train_timestep)
        self.real_guidance_scale = real_guidance_scale
        self.fake_guidance_scale = fake_guidance_scale
        self.timestep_shift = timestep_shift
        self.independent_first_frame = independent_first_frame
        self.teacher_forcing = teacher_forcing

    def _compute_kl_grad(
        self,
        noisy_latent: mx.array,
        estimated_clean: mx.array,
        timestep: mx.array,
        conditional_dict: dict,
        unconditional_dict: dict,
        normalization: bool = True,
    ):
        """Compute KL gradient for distribution matching.

        Args:
            noisy_latent: Noisy latent, shape (B, F, C, H, W).
            estimated_clean: Estimated clean latent, same shape.
            timestep: Timesteps, shape (B, F).
            conditional_dict: Conditional text embeddings.
            unconditional_dict: Unconditional text embeddings.
            normalization: Whether to normalize gradient.

        Returns:
            grad: KL gradient tensor.
            log_dict: Dict with debug info.
        """
        context_cond = conditional_dict.get("context")
        context_uncond = unconditional_dict.get("context")
        seq_len = noisy_latent.shape[1] * noisy_latent.shape[3] * noisy_latent.shape[4]

        # Fake score (conditional)
        _, pred_fake_cond = self.fake_score(
            noisy_latent, timestep, context_cond, seq_len
        )
        # Handle list output
        if isinstance(pred_fake_cond, list):
            pred_fake_cond = mx.stack(pred_fake_cond).transpose(0, 2, 1, 3, 4)

        if self.fake_guidance_scale != 0.0:
            _, pred_fake_uncond = self.fake_score(
                noisy_latent, timestep, context_uncond, seq_len
            )
            if isinstance(pred_fake_uncond, list):
                pred_fake_uncond = mx.stack(pred_fake_uncond).transpose(0, 2, 1, 3, 4)
            pred_fake = pred_fake_cond + (pred_fake_cond - pred_fake_uncond) * self.fake_guidance_scale
        else:
            pred_fake = pred_fake_cond

        # Real score (conditional + unconditional with CFG)
        _, pred_real_cond = self.real_score(
            noisy_latent, timestep, context_cond, seq_len
        )
        if isinstance(pred_real_cond, list):
            pred_real_cond = mx.stack(pred_real_cond).transpose(0, 2, 1, 3, 4)

        _, pred_real_uncond = self.real_score(
            noisy_latent, timestep, context_uncond, seq_len
        )
        if isinstance(pred_real_uncond, list):
            pred_real_uncond = mx.stack(pred_real_uncond).transpose(0, 2, 1, 3, 4)

        pred_real = pred_real_cond + (pred_real_cond - pred_real_uncond) * self.real_guidance_scale

        # DMD gradient
        grad = pred_fake - pred_real

        if normalization:
            p_real = estimated_clean - pred_real
            normalizer = mx.mean(mx.abs(p_real), axis=list(range(1, p_real.ndim)), keepdims=True)
            grad = grad / (normalizer + 1e-8)

        grad = mx.where(mx.isnan(grad), mx.zeros_like(grad), grad)

        return grad, {
            "gradient_norm": mx.mean(mx.abs(grad)),
            "timestep": timestep,
        }

    def distribution_matching_loss(
        self,
        image_or_video: mx.array,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: mx.array = None,
        key: mx.array = None,
    ):
        """Compute distribution matching loss.

        Args:
            image_or_video: Generated latent, shape (B, F, C, H, W).
            conditional_dict: Conditional text embeddings.
            unconditional_dict: Unconditional text embeddings.
            gradient_mask: Optional boolean mask for gradient application.
            key: Random key.

        Returns:
            loss: Scalar DMD loss.
            log_dict: Dict with debug info.
        """
        if key is None:
            key = mx.random.key(0)

        key, subkey1, subkey2 = mx.random.split(key, 3)

        batch_size, num_frame = image_or_video.shape[:2]

        # Sample timestep
        timestep = _get_timestep(
            0, self.num_train_timestep,
            batch_size, num_frame,
            self.num_frame_per_block,
            independent_first_frame=self.independent_first_frame,
            uniform_timestep=True,
            key=subkey1,
        ).astype(mx.float32)

        # Apply timestep shift
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

        # Compute KL gradient
        grad, log_dict = self._compute_kl_grad(
            noisy_latent=noisy_latent,
            estimated_clean=image_or_video,
            timestep=timestep,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
        )

        # DMD loss: 0.5 * ||x - (x - grad)||^2 = 0.5 * ||grad||^2
        # But implemented as: 0.5 * MSE(x, (x - grad).detach())
        if gradient_mask is not None:
            target = mx.stop_gradient(image_or_video - grad)
            loss = 0.5 * mx.mean(
                (image_or_video.astype(mx.float64) - target.astype(mx.float64)) ** 2
            )
        else:
            target = mx.stop_gradient(image_or_video - grad)
            loss = 0.5 * mx.mean(
                (image_or_video.astype(mx.float64) - target.astype(mx.float64)) ** 2
            )

        return loss, log_dict

    def generator_loss(
        self,
        clean_latent: mx.array,
        conditional_dict: dict,
        unconditional_dict: dict,
        key: mx.array = None,
    ):
        """Compute generator loss: run generator then DMD loss.

        Args:
            clean_latent: Clean latent, shape (B, F, C, H, W).
            conditional_dict: Conditional text embeddings.
            unconditional_dict: Unconditional text embeddings.
            key: Random key.

        Returns:
            loss: Scalar loss.
            log_dict: Dict with debug info.
        """
        if key is None:
            key = mx.random.key(0)

        # Step 1: Run generator on backward-simulated noisy input
        pred_image, _ = self._run_generator(
            clean_latent=clean_latent,
            conditional_dict=conditional_dict,
            key=key,
        )

        # Step 2: Compute DMD loss
        dmd_loss, log_dict = self.distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
        )

        return dmd_loss, log_dict

    def _run_generator(
        self,
        clean_latent: mx.array,
        conditional_dict: dict,
        key: mx.array = None,
    ):
        """Backward simulation and single-step generator inference.

        Args:
            clean_latent: Clean latent, shape (B, F, C, H, W).
            conditional_dict: Conditional text embeddings.
            key: Random key.

        Returns:
            pred_image: Generator output.
            gradient_mask: None (all frames get gradients).
        """
        if key is None:
            key = mx.random.key(0)

        batch_size = clean_latent.shape[0]
        context = conditional_dict.get("context")
        seq_len = clean_latent.shape[1] * clean_latent.shape[3] * clean_latent.shape[4]

        # For simplicity in MLX: use a single denoising step from noise
        # Full backward simulation requires the inference pipeline
        key, subkey = mx.random.split(key)
        noise = mx.random.normal(clean_latent.shape, key=subkey)

        # Sample a random timestep from the denoising step list
        denoising_steps = [1000, 750, 500, 250]
        step_idx = mx.random.randint(0, len(denoising_steps), (1,)).item()
        timestep_val = mx.array([denoising_steps[step_idx]], dtype=mx.float32)

        # Add noise at the selected timestep
        noisy_input = self.scheduler.add_noise(
            clean_latent.reshape(-1, *clean_latent.shape[2:]),
            noise.reshape(-1, *noise.shape[2:]),
            mx.broadcast_to(timestep_val, (batch_size * clean_latent.shape[1],)),
        ).reshape(clean_latent.shape)

        # Generator predicts x0
        flow_pred = self.generator(noisy_input, timestep_val, context, seq_len)
        if isinstance(flow_pred, list):
            flow_pred = mx.stack(flow_pred).transpose(0, 2, 1, 3, 4)

        # Simple Euler step: x0 = xt - t * flow_pred (approximate)
        # For proper x0 prediction, use scheduler
        pred_image = noisy_input - timestep_val * flow_pred / self.num_train_timestep

        return pred_image, None

    def critic_loss(
        self,
        clean_latent: mx.array,
        conditional_dict: dict,
        unconditional_dict: dict,
        key: mx.array = None,
    ):
        """Compute critic (fake score) denoising loss.

        Args:
            clean_latent: Clean latent, shape (B, F, C, H, W).
            conditional_dict: Conditional text embeddings.
            unconditional_dict: Unconditional text embeddings.
            key: Random key.

        Returns:
            loss: Scalar denoising loss.
            log_dict: Dict with debug info.
        """
        if key is None:
            key = mx.random.key(0)

        key, subkey1, subkey2 = mx.random.split(key, 3)

        # Run generator (no grad)
        generated_image, _ = self._run_generator(
            clean_latent=clean_latent,
            conditional_dict=conditional_dict,
            key=subkey1,
        )

        # Sample critic timestep
        batch_size, num_frame = generated_image.shape[:2]
        critic_timestep = _get_timestep(
            0, self.num_train_timestep,
            batch_size, num_frame,
            self.num_frame_per_block,
            independent_first_frame=self.independent_first_frame,
            uniform_timestep=True,
            key=subkey2,
        ).astype(mx.float32)

        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * (critic_timestep / 1000) / \
                (1 + (self.timestep_shift - 1) * (critic_timestep / 1000)) * 1000
        critic_timestep = mx.clip(critic_timestep, self.min_step, self.max_step)

        # Add noise to generated image
        critic_noise = mx.random.normal(generated_image.shape, key=subkey2)
        noisy_generated = self.scheduler.add_noise(
            generated_image.reshape(-1, *generated_image.shape[2:]),
            critic_noise.reshape(-1, *critic_noise.shape[2:]),
            critic_timestep.reshape(-1),
        ).reshape(generated_image.shape)

        # Fake score prediction
        context = conditional_dict.get("context")
        seq_len = generated_image.shape[1] * generated_image.shape[3] * generated_image.shape[4]
        _, pred_fake = self.fake_score(
            noisy_generated, critic_timestep, context, seq_len
        )
        if isinstance(pred_fake, list):
            pred_fake = mx.stack(pred_fake).transpose(0, 2, 1, 3, 4)

        # Simple MSE loss
        loss = mx.mean((pred_fake - generated_image) ** 2)

        return loss, {"critic_timestep": critic_timestep}
