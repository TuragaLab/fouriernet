import jax.numpy as jnp
from jax import lax
from flax import linen as nn
from jax import vmap


from .utils import complex_lecun_normal, next_order

from typing import Tuple, Callable


class FourierConv2D(nn.Module):
    """
    Applies fourier convolution over h and w axis of input signal [b x h x w x c], where
    b is a batch axis and c number of channels. This operation also supports efficient
    multiscale feature output by simply cropping the calculated Fourier
    features to the desired resolutions through stride. Pads to next power of 2 for efficient DFTs
    and returns "same" mode (i.e. same shape as input image).
    """

    features: int
    stride: Tuple[int, int] = (1, 1)
    kernel_init: Callable = complex_lecun_normal()
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, image: jnp.ndarray) -> jnp.ndarray:
        kernel = self.param(
            "kernel",
            self.kernel_init,
            (
                int(2 * image.shape[1] / self.stride[0]),
                int(image.shape[2] / self.stride[1]) + 1,
                self.features,
                image.shape[-1],
            ),
        )
        bias = self.param("bias", self.bias_init, (self.features,))
        return vmap(self.fourier_convolution, in_axes=(0, None, None, None))(
            image, kernel, bias, self.stride
        )

    @staticmethod
    def fourier_convolution(
        image: jnp.ndarray,
        kernel: jnp.ndarray,
        bias: jnp.ndarray,
        stride: Tuple[int, int],
    ) -> jnp.ndarray:
        """Fourier convolution in fourier space."""

        # padding to prevent circular convolution
        padded_shape = [2 * k - 1 for k in image.shape[:2]]
        fast_shape = [next_order(k) for k in padded_shape]
        fast_strided_shape = [next_order(k) for k in kernel.shape[:2]]

        # Convolution and bias
        F_image = jnp.fft.rfft2(image, s=fast_shape, axes=[0, 1])
        F_image = F_image[:: stride[0], :: stride[1]]
        F_image = jnp.sum(
            vmap(lambda im, kernel: im * kernel, in_axes=(-2, -1))(
                F_image[..., None], kernel
            ),
            axis=0,
        )

        conv_image = jnp.fft.irfft2(F_image, s=fast_strided_shape, axes=[0, 1],)
        conv_image += bias

        # Everything below also works for strided / padded images
        # Cropping
        conv_image = conv_image[
            tuple([slice(sz) for sz in [*padded_shape, *conv_image.shape[2:]]])
        ]

        # Returning same mode
        start_idx = [
            (k1 - k2) // 2 if idx < 2 else 0
            for idx, (k1, k2) in enumerate(zip(conv_image.shape, image.shape))
        ]
        stop_idx = [
            k1 + k2 if idx < 2 else k3
            for idx, (k1, k2, k3) in enumerate(
                zip(start_idx, image.shape, conv_image.shape)
            )
        ]
        conv_image = lax.slice(conv_image, start_idx, stop_idx)

        return conv_image
