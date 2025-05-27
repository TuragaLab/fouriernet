import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
import flax.linen as nn

from .layers import FourierConv, SmallFourierConv
from .utils import he_uniform, fan_in_bias, _compute_fans

from typing import Iterable, Tuple, Union
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

        fan_in, _ = _compute_fans((*image.shape[1:-1], self.fourier_features, 1))
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


class LargeFourierNet3D(nn.Module):
    num_planes: int = 32
    global_features_per_plane: int = 1
    local_features_per_plane: Iterable[int] = (3, 3)
    fourier_strides: Tuple[int, int] = (2, 2)
    quantile_scale: float = 0.01
    leaky_relu_slope: float = 0.01
    precision: str = 'fastest'

    @nn.compact
    def __call__(self, image: jnp.ndarray) -> jnp.ndarray:
        image, scale = self.input_scaling(image, scale=self.quantile_scale)

        image = FourierConv(
            self.global_features_per_plane * self.num_planes, self.fourier_strides,
        )(image)
        image = nn.leaky_relu(image, negative_slope=self.leaky_relu_slope)
        image = nn.GroupNorm(num_groups=None, group_size=1, epsilon=1e-5)(image)

        image = rearrange(image, "b h w (d c) -> b d h w c", d=self.num_planes)

        for features in self.local_features_per_plane:
            image = SmallFourierConv(features, (11, 7, 7))(image)
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


class FourierNet3DFlaxConvs(nn.Module):
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

        fan_in, _ = _compute_fans((*image.shape[1:-1], self.features_per_plane, self.features_per_plane))
        image = nn.Conv(self.features_per_plane, (11, 7, 7), padding="SAME", kernel_init=he_uniform(), bias_init=fan_in_bias(fan_in), precision=self.precision)(image)
        image = nn.leaky_relu(image, negative_slope=self.leaky_relu_slope)
        image = nn.GroupNorm(num_groups=None, group_size=1, epsilon=1e-5)(image)

        fan_in, _ = _compute_fans((*image.shape[1:-1], self.features_per_plane, 1))
        image = nn.Conv(1, (11, 7, 7), padding="SAME", kernel_init=he_uniform(), bias_init=fan_in_bias(fan_in), precision=self.precision)(image)
        image = nn.relu(image)

        return image * scale
        # image, scale = self.input_scaling(image, scale=self.quantile_scale)

        # image = FourierConv(
        #     self.features_per_plane * self.num_planes, self.fourier_strides,
        # )(image)
        # image = nn.leaky_relu(image, negative_slope=self.leaky_relu_slope)
        # image = nn.GroupNorm(num_groups=None, group_size=1, epsilon=1e-5)(image)

        # image = rearrange(image, "b h w (c d) -> b c d h w", d=self.num_planes)

        # # image = nn.Conv(self.features_per_plane, (11, 7, 7), padding="SAME", precision=self.precision)(image)
        # params_conv1 = self.param("conv1", nn.initializers.he_normal(in_axis=1, out_axis=0), (self.features_per_plane, self.features_per_plane, 11, 7, 7))
        # image = jax.lax.conv_general_dilated(image, params_conv1, (1, 1, 1), "SAME", precision=self.precision, dimension_numbers=("NCDHW", "OIDHW", "NDHWC"))
        # image = nn.leaky_relu(image, negative_slope=self.leaky_relu_slope)
        # image = nn.GroupNorm(num_groups=None, group_size=1, epsilon=1e-5)(image)

        # image = rearrange(image, "b d h w c -> b c d h w", d=self.num_planes)
        # params_conv2 = self.param("conv2", nn.initializers.he_normal(in_axis=1, out_axis=0), (1, self.features_per_plane, 11, 7, 7))
        # image = jax.lax.conv_general_dilated(image, params_conv2, (1, 1, 1), "SAME", precision=self.precision, dimension_numbers=("NCDHW", "OIDHW", "NDHWC"))
        # # image = nn.Conv(1, (11, 7, 7), padding="SAME", precision=self.precision)(image)
        # image = nn.relu(image)

        # return image * scale

    @staticmethod
    def input_scaling(image: jnp.ndarray, scale: float) -> Tuple[jnp.ndarray, float]:
        """
        Scales image, returns image / (median(image) * scale).
        ** Stops gradients on scale**.
        """
        normed_scale = stop_gradient(jnp.median(image)) * scale
        return image / normed_scale, normed_scale


class DenoisingUNet3D(nn.Module):
    """
    Denoising UNet3D adapted from
    Yanny et al. 2022 (https://github.com/Waller-Lab/MultiWienerNet).
    """
    num_features: int = 1
    base_num_features: int = 32
    precision: str = 'fastest'
    use_fft_conv_block: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # encoding path
        c1 = self.conv_block(x, num_features=self.base_num_features)
        y = self.downsample(c1, num_features=self.base_num_features)
        c2 = self.conv_block(y, num_features=self.base_num_features * 2)
        y = self.downsample(c2, num_features=self.base_num_features * 2)
        c3 = self.conv_block(y, num_features=self.base_num_features * 4)
        y = self.downsample(c3, num_features=self.base_num_features * 4)
        c4 = self.conv_block(y, num_features=self.base_num_features * 8)
        y = self.downsample(c4, num_features=self.base_num_features * 8)

        y = self.conv_block(y, num_features=self.base_num_features * 8)

        # decoding path
        y = self.upsample(y, num_features=self.base_num_features * 8)
        y = self.conv_block(jnp.concatenate([y, c4], axis=-1), num_features=self.base_num_features * 4)
        y = self.upsample(y, num_features=self.base_num_features * 4)
        y = self.conv_block(jnp.concatenate([y, c3], axis=-1), num_features=self.base_num_features * 2)
        y = self.upsample(y, num_features=self.base_num_features * 2)
        y = self.conv_block(jnp.concatenate([y, c2], axis=-1), num_features=self.base_num_features)
        y = self.upsample(y, num_features=self.base_num_features)
        y = self.conv_block(jnp.concatenate([y, c1], axis=-1), num_features=self.num_features)
        y = nn.relu(y)
        return y

    def conv_block(self, x: jnp.ndarray, num_features: int) -> jnp.ndarray:
        y = x
        for i in range(2):
            if self.use_fft_conv_block:
                y = SmallFourierConv(num_features, (3, 3, 3))(y)
            else:
                fan_in, _ = _compute_fans((*y.shape[1:-1], y.shape[-1], num_features))
                y = nn.Conv(features=num_features, kernel_size=(3, 3, 3), padding='SAME', kernel_init=he_uniform(), bias_init=fan_in_bias(fan_in), precision=self.precision)(y)
            y = nn.GroupNorm(num_groups=None, group_size=1, epsilon=1e-5)(y)
            y = nn.leaky_relu(y, negative_slope=0.01)
        return y

    def downsample(self, x: jnp.ndarray, num_features: int) -> jnp.ndarray:
        fan_in, _ = _compute_fans((*x.shape[1:-1], x.shape[-1], num_features))
        y = nn.Conv(features=num_features, kernel_size=(2, 2, 2), strides=(2, 2, 2), feature_group_count=num_features, kernel_init=he_uniform(scale=0.01, shift=0.025), bias_init=fan_in_bias(fan_in, scale=0.01), precision=self.precision)(x)
        return y

    def upsample(self, x: jnp.ndarray, num_features: int) -> jnp.ndarray:
        fan_in, _ = _compute_fans((*x.shape[1:-1], x.shape[-1], num_features))
        y = nn.ConvTranspose(features=num_features, kernel_size=(2, 2, 2), strides=(2, 2, 2), kernel_init=he_uniform(scale=0.01, shift=0.025), bias_init=fan_in_bias(fan_in, scale=0.01), precision=self.precision)(x)
        return y


class FourierNetDenoisingUNet3D(nn.Module):
    num_planes: int = 32
    features_per_plane: int = 5
    base_num_features: int = 32
    fourier_strides: Tuple[int, int] = (2, 2)
    use_fft_conv_block: bool = False
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

        image = DenoisingUNet3D(num_features=1, base_num_features=self.base_num_features, precision=self.precision, use_fft_conv_block=self.use_fft_conv_block)(image)

        return image * scale

    @staticmethod
    def input_scaling(image: jnp.ndarray, scale: float) -> Tuple[jnp.ndarray, float]:
        """
        Scales image, returns image / (median(image) * scale).
        ** Stops gradients on scale**.
        """
        normed_scale = stop_gradient(jnp.median(image)) * scale
        return image / normed_scale, normed_scale


class DirectConvFourierNet3D(nn.Module):
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

        fan_in, _ = _compute_fans((*image.shape[1:-1], self.features_per_plane, self.features_per_plane))
        image = nn.Conv(features=self.features_per_plane, kernel_size=(11, 7, 7), strides=(1, 1, 1), padding='SAME', kernel_init=he_uniform(), bias_init=fan_in_bias(fan_in), precision=self.precision)(image)
        image = nn.leaky_relu(image, negative_slope=self.leaky_relu_slope)
        image = nn.GroupNorm(num_groups=None, group_size=1, epsilon=1e-5)(image)

        fan_in, _ = _compute_fans((*image.shape[1:-1], self.features_per_plane, 1))
        image = nn.Conv(features=1, kernel_size=(11, 7, 7), strides=(1, 1, 1), padding='SAME', kernel_init=he_uniform(), bias_init=fan_in_bias(fan_in), precision=self.precision)(image)
        image = nn.relu(image)

        return image * scale

    @staticmethod
    def input_scaling(image: jnp.ndarray, scale: float) -> Tuple[jnp.ndarray, float]:
        """
        Scales image, returns image / (median(image) * scale).
        ** Stops gradients on scale**.
        """
        normed_scale = stop_gradient(jnp.median(image)) * scale
        # normed_scale = jnp.median(image) * scale
        return image / normed_scale, normed_scale


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

        fan_in, _ = _compute_fans((*image.shape[1:-1], self.fourier_features, 1))
        image = nn.Conv(features=64, kernel_size=(11, 11), strides=(1, 1), padding='SAME', kernel_init=he_uniform(), bias_init=fan_in_bias(fan_in), precision=self.precision)(image)
        image = nn.leaky_relu(image, negative_slope=self.leaky_relu_slope)
        image = nn.BatchNorm(use_running_average=self.use_running_average)(image)

        fan_in, _ = _compute_fans((11, 11, 64, 64))
        image = nn.Conv(features=64, kernel_size=(11, 11), strides=(1, 1), padding='SAME', kernel_init=he_uniform(), bias_init=fan_in_bias(fan_in), precision=self.precision)(image)
        image = nn.leaky_relu(image, negative_slope=self.leaky_relu_slope)
        image = nn.BatchNorm(use_running_average=self.use_running_average)(image)

        fan_in, _ = _compute_fans((11, 11, 64, 3))
        image = nn.Conv(features=3, kernel_size=(11, 11), strides=(1, 1), padding='SAME', kernel_init=he_uniform(), bias_init=fan_in_bias(fan_in), precision=self.precision)(image)
        image = nn.relu(image)

        return image


class FourierNetSpectral(nn.Module):
    num_wavelengths: int = 3
    fourier_features: int = 20
    fourier_strides: Tuple[int, int] = (2, 2)
    conv_features: int = 64
    num_conv_layers: int = 2
    leaky_relu_slope: float = 0.01
    precision: str = 'fastest'

    @nn.compact
    def __call__(self, image: jnp.ndarray) -> jnp.ndarray:
        image = FourierConv(self.fourier_features, self.fourier_strides)(image)
        image = nn.leaky_relu(image, negative_slope=self.leaky_relu_slope)
        image = nn.GroupNorm(num_groups=None, group_size=1, epsilon=1e-5)(image)

        fan_in, _ = _compute_fans((*image.shape[1:-1], self.fourier_features, 1))
        image = nn.Conv(features=64, kernel_size=(11, 11), strides=(1, 1), padding='SAME', kernel_init=he_uniform(), bias_init=fan_in_bias(fan_in), precision=self.precision)(image)
        image = nn.leaky_relu(image, negative_slope=self.leaky_relu_slope)
        image = nn.GroupNorm(num_groups=None, group_size=1, epsilon=1e-5)(image)

        for layer in range(self.num_conv_layers - 1):
            fan_in, _ = _compute_fans((11, 11, 64, 64))
            image = nn.Conv(features=64, kernel_size=(11, 11), strides=(1, 1), padding='SAME', kernel_init=he_uniform(), bias_init=fan_in_bias(fan_in), precision=self.precision)(image)
            image = nn.leaky_relu(image, negative_slope=self.leaky_relu_slope)
            image = nn.GroupNorm(num_groups=None, group_size=1, epsilon=1e-5)(image)

        fan_in, _ = _compute_fans((11, 11, 64, self.num_wavelengths))
        image = nn.Conv(features=self.num_wavelengths, kernel_size=(11, 11), strides=(1, 1), padding='SAME', kernel_init=he_uniform(), bias_init=fan_in_bias(fan_in), precision=self.precision)(image)
        image = nn.relu(image)

        return image
