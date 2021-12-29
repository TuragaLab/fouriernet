import jax.numpy as jnp
from jax.lax import stop_gradient
import flax.linen as nn

from .layers import FourierConv2D

from typing import Tuple
from einops import rearrange


class FourierNet2D(nn.Module):

    fourier_features: int = 8
    fourier_strides: Tuple[int, int] = (2, 2)
    leaky_relu_slope: float = 0.01

    @nn.compact
    def __call__(self, image: jnp.ndarray) -> jnp.ndarray:
        image, scale = self.input_scaling(image, scale=0.01)

        # First layer
        image = FourierConv2D(self.fourier_features, self.fourier_strides)(image)
        image = nn.leaky_relu(image, negative_slope=self.leaky_relu_slope)
        image = nn.GroupNorm(num_groups=1)(image)

        # Output layer
        image = nn.Conv(features=1, kernel_size=(11, 11), strides=(1, 1))(image)
        image = nn.relu(image)

        return image * scale

    @staticmethod
    def input_scaling(image: jnp.ndarray, scale: float) -> Tuple[jnp.ndarray, float]:
        """Scales image, returns image / (median(image) * scale).
        ** Stops gradients on scale**."""
        # We dont need the gradient as we rescale later, and since its a big number itll be more stable
        normed_scale = stop_gradient(jnp.median(image) * scale)
        return image / normed_scale, normed_scale


class FourierNet3D(nn.Module):
    n_planes: int = 12
    n_features_per_plane: int = 5
    fourier_strides: Tuple[int, int] = (2, 2)
    leaky_relu_slope: float = 0.01

    @nn.compact
    def __call__(self, image: jnp.ndarray) -> jnp.ndarray:
        image, scale = self.input_scaling(image, scale=0.01)

        # First layer
        image = FourierConv2D(
            self.n_features_per_plane * self.n_planes, self.fourier_strides,
        )(image)
        image = nn.leaky_relu(image, negative_slope=self.leaky_relu_slope)
        image = nn.GroupNorm(num_groups=1)(image)

        # 2D to 3D
        image = rearrange(image, "b h w (c d) -> b d h w c", d=self.n_planes)

        # Second layer
        image = nn.Conv(features=5, kernel_size=(11, 7, 7), strides=(1, 1, 1))(image)
        image = nn.leaky_relu(image, negative_slope=self.leaky_relu_slope)
        image = nn.GroupNorm(num_groups=1)(image)

        # Output layer
        image = nn.Conv(features=1, kernel_size=(11, 7, 7), strides=(1, 1, 1))(image)
        image = nn.relu(image)
        return image * scale

    @staticmethod
    def input_scaling(image: jnp.ndarray, scale: float) -> Tuple[jnp.ndarray, float]:
        """Scales image, returns image / (median(image) * scale).
        ** Stops gradients on scale**."""
        # We dont need the gradient as we rescale later, and since its a big number itll be more stable
        normed_scale = stop_gradient(jnp.median(image) * scale)
        return image / normed_scale, normed_scale
