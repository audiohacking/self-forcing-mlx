"""Flow matching scheduler wrapping mlx-arsenal's FlowMatchEulerDiscreteScheduler.

Ported from utils/scheduler.py.
Uses mlx_arsenal.diffusion.FlowMatchEulerDiscreteScheduler.
"""
import mlx.core as mx
import numpy as np
from mlx_arsenal.diffusion import FlowMatchEulerDiscreteScheduler


class FlowMatchScheduler:
    """Flow matching scheduler wrapping mlx-arsenal.

    Delegates to FlowMatchEulerDiscreteScheduler for step logic,
    timestep scheduling, and noise addition.
    """

    def __init__(
        self,
        num_train_timesteps=1000,
        shift=1.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        extra_one_step=False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.extra_one_step = extra_one_step
        self._scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
        )
        self.sigmas = None
        self.timesteps = None

    def set_timesteps(self, num_inference_steps, denoising_strength=1.0):
        """Set scheduler timesteps for sampling.

        Args:
            num_inference_steps: Number of sampling steps.
            denoising_strength: How much to denoise (1.0 = full denoise).
        """
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            sigmas = np.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            sigmas = np.linspace(sigma_start, self.sigma_min, num_inference_steps)
        # Sigmoid warping
        sigmas = self.shift * sigmas / (1.0 + (self.shift - 1.0) * sigmas)
        self._scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)
        self.sigmas = mx.array(self._scheduler.sigmas)
        self.timesteps = mx.array(sigmas * self.num_train_timesteps)

    def step(self, model_output, timestep, sample):
        """Euler integration step via mlx-arsenal.

        Args:
            model_output: The model's flow prediction (velocity).
            timestep: Current timestep(s).
            sample: Current latent sample.

        Returns:
            Next sample after one Euler step.
        """
        return self._scheduler.step(model_output, timestep, sample)

    def add_noise(self, original_samples, noise, timestep):
        """Apply noise to clean samples.

        x_t = (1 - sigma) * x0 + sigma * noise

        Args:
            original_samples: Clean samples.
            noise: Noise tensor, same shape.
            timestep: Timestep(s) to determine noise level.

        Returns:
            Noisy samples.
        """
        timestep_flat = timestep.flatten() if timestep.ndim >= 2 else timestep
        indices = self._find_sigma_index(timestep_flat)
        sigma = self.sigmas[indices]
        sigma = sigma.reshape(-1, *([1] * (original_samples.ndim - 1)))
        return self._scheduler.add_noise(original_samples, noise, sigma)

    def _find_sigma_index(self, timestep):
        """Find the sigma index for a given timestep."""
        ts = self.timesteps.astype(timestep.dtype)
        diff = mx.abs(ts.reshape(1, -1) - timestep.reshape(-1, 1))
        return mx.argmin(diff, axis=1)

    def training_target(self, sample, noise, timestep):
        """Compute flow matching target: noise - sample."""
        return noise - sample
