"""MLX model variants for Self-Forcing training.

Port of model/*.py to MLX. Each variant implements a training strategy
(generator loss, critic loss) using MLX operations.
"""

from sforcing.trainers.diffusion import CausalDiffusion
from sforcing.trainers.causvid import CausVid
from sforcing.trainers.dmd import DMD
from sforcing.trainers.gan import GAN
from sforcing.trainers.sid import SiD
from sforcing.trainers.ode_regression import ODERegression

__all__ = [
    "CausalDiffusion",
    "CausVid",
    "DMD",
    "GAN",
    "SiD",
    "ODERegression",
]
