import jax
import jax.numpy as jnp
from jax.core import NamedShape
from jax.lax import stop_gradient
import flax.linen as nn

from .layers import FourierConv, SmallFourierConv
from .utils import he_uniform, fan_in_bias, _compute_fans

from typing import Tuple
from einops import rearrange, repeat


class FourierNet2D(nn.Module):

    fourier_features: int = 5
    fourier_strides: Tuple[int, int] = (2, 2)
    quantile_scale: float = 0.01
    leaky_relu_slope: float = 0.01
    precision: str = 'fastest'

    @nn.compact
    def __call__(self, image: jnp.ndarray) -> jnp.ndarray:
        image, scale = self.input_scaling(image, scale=self.quantile_scale)

        image = FourierConv(self.fourier_features, self.fourier_strides)(image)
        image = nn.leaky_relu(image, negative_slope=self.leaky_relu_slope)
        image = nn.GroupNorm(num_groups=None, group_size=1, epsilon=1e-5)(image)

        fan_in, _ = _compute_fans(NamedShape(*image.shape[1:-1], self.fourier_features, 1))
        image = nn.Conv(features=1, kernel_size=(11, 11), strides=(1, 1), padding='SAME', kernel_init=he_uniform(), bias_init=fan_in_bias(fan_in), precision=self.precision)(image)
        image = nn.relu(image)

        return image * scale

    @staticmethod
    def input_scaling(image: jnp.ndarray, scale: float) -> Tuple[jnp.ndarray, float]:
        """Scales image, returns image / (median(image) * scale).
        ** Stops gradients on scale**."""
        normed_scale = stop_gradient(jnp.median(image)) * scale
        return image / normed_scale, normed_scale


# multiple single-plane fouriernets
VmapFourierNet2D = nn.vmap(
    FourierNet2D,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": 0},
    split_rngs={"params": True},
)


class FourierNet3D(nn.Module):
    num_planes: int = 32
    features_per_plane: int = 5
    fourier_strides: Tuple[int, int] = (2, 2)
    quantile_scale: float = 0.01
    leaky_relu_slope: float = 0.01
    precision: str = 'fastest'

    @nn.compact
    def __call__(self, image: jnp.ndarray) -> jnp.ndarray:
        image, scale = self.input_scaling(image, scale=self.quantile_scale)

        image = FourierConv(
            self.features_per_plane * self.num_planes, self.fourier_strides,
        )(image)
        image = nn.leaky_relu(image, negative_slope=self.leaky_relu_slope)
        image = nn.GroupNorm(num_groups=None, group_size=1, epsilon=1e-5)(image)

        image = rearrange(image, "b h w (c d) -> b d h w c", d=self.num_planes)

        image = SmallFourierConv(self.features_per_plane, (11, 7, 7))(image)
        image = nn.leaky_relu(image, negative_slope=self.leaky_relu_slope)
        image = nn.GroupNorm(num_groups=None, group_size=1, epsilon=1e-5)(image)

        image = SmallFourierConv(1, (11, 7, 7))(image)
        image = nn.relu(image)

        return image * scale

    @staticmethod
    def input_scaling(image: jnp.ndarray, scale: float) -> Tuple[jnp.ndarray, float]:
        """
        Scales image, returns image / (median(image) * scale).
        ** Stops gradients on scale**.
        """
        normed_scale = stop_gradient(jnp.median(image)) * scale
        return image / normed_scale, normed_scale


class ManuallyChunkedFourierNet3D(nn.Module):
    num_chunks: int = 8
    num_planes: int = 32
    features_per_plane: int = 5
    fourier_strides: Tuple[int, int] = (2, 2)
    quantile_scale: float = 0.01
    leaky_relu_slope: float = 0.01
    precision: str = 'fastest'

    @nn.compact
    def __call__(self, image: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([FourierNet3D(num_planes=self.num_planes)(image) for i in range(self.num_chunks)], axis=1)


class FourierNetRGB(nn.Module):

    fourier_features: int = 20
    fourier_strides: Tuple[int, int] = (2, 2)
    leaky_relu_slope: float = 0.01
    use_running_average: bool = False
    precision: str = 'fastest'

    @nn.compact
    def __call__(self, image: jnp.ndarray) -> jnp.ndarray:
        image = FourierConv(self.fourier_features, self.fourier_strides)(image)
        image = nn.leaky_relu(image, negative_slope=self.leaky_relu_slope)
        image = nn.BatchNorm(use_running_average=self.use_running_average)(image)

        fan_in, _ = _compute_fans(NamedShape(*image.shape[1:-1], self.fourier_features, 1))
        image = nn.Conv(features=64, kernel_size=(11, 11), strides=(1, 1), padding='SAME', kernel_init=he_uniform(), bias_init=fan_in_bias(fan_in), precision=self.precision)(image)
        image = nn.leaky_relu(image, negative_slope=self.leaky_relu_slope)
        image = nn.BatchNorm(use_running_average=self.use_running_average)(image)

        fan_in, _ = _compute_fans(NamedShape(11, 11, 64, 64))
        image = nn.Conv(features=64, kernel_size=(11, 11), strides=(1, 1), padding='SAME', kernel_init=he_uniform(), bias_init=fan_in_bias(fan_in), precision=self.precision)(image)
        image = nn.leaky_relu(image, negative_slope=self.leaky_relu_slope)
        image = nn.BatchNorm(use_running_average=self.use_running_average)(image)

        fan_in, _ = _compute_fans(NamedShape(11, 11, 64, 3))
        image = nn.Conv(features=3, kernel_size=(11, 11), strides=(1, 1), padding='SAME', kernel_init=he_uniform(), bias_init=fan_in_bias(fan_in), precision=self.precision)(image)
        image = nn.relu(image)

        return image
