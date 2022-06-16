import jax.numpy as jnp
from jax import lax
from flax import linen as nn
from jax import vmap
from jax.core import NamedShape
from jax.random import PRNGKey


from .utils import complex_he_uniform, complex_fan_in_bias, real_fan_in_bias, next_order

from typing import Tuple, Callable, Optional, Union, Sequence, Any
Dtype = Any
Shape = Any

from functools import partial

class FourierConv(nn.Module):
    """
    Convolution Module implemented using the Fourier convolution theorem (for n
    dimensional input).  Weights are parameterized in Fourier space, and their
    shape must match the shape of the input.

    Args:
      features: number of convolution filters.
      strides: a sequence of `n` integers, representing the inter-window
        strides.
      padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
        of `n` `(low, high)` integer pairs that give the padding to apply before
        and after each spatial dimension.
      use_bias: whether to add a bias to the output (default: True).
      dtype: the dtype of the computation (default: float32).
      kernel_init: initializer for the convolutional kernel.
      bias_init: initializer for the bias.
    """

    features: int
    strides: Optional[Tuple[int, ...]] = None
    padding: Union[str, Sequence[Tuple[int, ...]]] = "SAME"
    use_bias: bool = True
    dtype: Dtype = jnp.float64
    kernel_init: Callable[
        [PRNGKey, Shape, Dtype], jnp.ndarray
    ] = complex_he_uniform()
    bias_init: Callable[
        [PRNGKey, Shape, Dtype], jnp.ndarray
    ] = real_fan_in_bias()

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Applies a convolution to the inputs using the Fourier convolution
        theorem.

        Args:
          inputs: input data with dimensions (batch, spatial_dims..., features).
        Returns:
          The convolved data.
        """
        inputs = jnp.asarray(inputs, self.dtype)
        # Extract shapes used to make kernel from input (only matters on first
        # call)
        # TODO(dd): use explicit setup()
        in_features = inputs.shape[-1]
        kernel_size = inputs.shape[1:-1]
        ndim = len(kernel_size)
        if self.strides is None:
            strides = (1,) * ndim
        else:
            strides = self.strides

        # Kernel size must be modified to match padded and strided input size
        if self.padding == "SAME" and strides == ((1,) * ndim):
            kernel_size = tuple(2 * k - 1 for k in kernel_size)
        elif strides != ((1,) * ndim):
            kernel_size = tuple(
                k // (s - 1) for k, s in zip(kernel_size, strides)
            )
        # Kernel size must be modified to make last dimension small due to
        # Hermitian symmetry
        # TODO(dd): reimplement rfft kernel
        # kernel_size = kernel_size[:-1] + (
        #     kernel_size[-1] // 2 + 1,
        # )
        kernel_shape = kernel_size + (
            in_features,
            self.features,
        )

        kernel = self.param("kernel", self.kernel_init, kernel_shape)

        # Get padded shape to prevent circular convolution
        # TODO(dd): reimplement computing fast shape for FFTs
        fft_shape = [2 * k1 - 1 for k1 in inputs.shape[1 : ndim + 1]]
        fft_axes = tuple(range(1, ndim + 1))
        fft = partial(jnp.fft.fftn, axes=fft_axes, s=fft_shape)
        ifft = partial(jnp.fft.ifftn, axes=fft_axes)

        # Transform signals and perform Fourier convolution
        y = fft(inputs)
        if strides != ((1,) * ndim):
            # Apply stride by subsampling Fourier transformed input
            total_dims = len(y.shape)
            y = lax.slice(
                y,
                start_indices=tuple(0 for s in y.shape),
                limit_indices=tuple(s for s in y.shape),
                strides=(1,) + strides + (1,),
            )
        y = ifft(
            y[..., jnp.newaxis] * kernel)

        # Get real features
        y = y.real

        if self.padding == "SAME":
            # Crop image back to size of input
            start_idx = (
                (0,)
                + tuple(
                    (k2 - k1) // 2
                    for k1, k2 in zip(
                        inputs.shape[1 : ndim + 1],
                        y.shape[1 : ndim + 1],
                    )
                )
                + (0, 0)
            )
            stop_idx = (
                (inputs.shape[0],)
                + tuple(
                    si + k1
                    for si, k1 in zip(
                        start_idx[1 : ndim + 1],
                        inputs.shape[1 : ndim + 1],
                    )
                )
                + (inputs.shape[-1], self.features)
            )
            y = lax.slice(y, start_idx, stop_idx)
        else:
            return NotImplemented

        # Sum over input feature dimension
        y = jnp.sum(y, axis=-1)

        if self.use_bias:
            bias = self.param("bias", self.bias_init, NamedShape(1, *[1 for d in range(ndim)], self.features), NamedShape(*kernel.shape))
            y = y + bias
        return y
