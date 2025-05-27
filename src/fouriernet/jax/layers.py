import jax.numpy as jnp
from jax import lax
from flax import linen as nn
from jax import vmap
from jax.random import PRNGKey

from typing import Tuple, Callable, Optional, Union, Sequence, Any

from functools import partial

from .utils import next_order, _compute_fans, he_uniform, complex_he_uniform, fan_in_bias

DType = Any


def looped_small_fourier_convolution(y: jnp.ndarray, kernel: jnp.ndarray, ndim: int, ifft: Callable) -> jnp.ndarray:
    # y shape (N, D, H, W, C_in)
    # kernel shape (D, H, W, C_in, C_out)
    # convolved shape (N, D, H, W, C_in, C_out)
    convolved = jnp.zeros(y.shape + (kernel.shape[-1],))
    indices = tuple(slice(0, s) for s in convolved.shape)
    # transform kernel, padding to correct shape 
    kernel_fft_axes = tuple(range(0, ndim))
    kernel_fft_shape = tuple(k2 for k2 in y.shape[1:ndim + 1])
    kernel_fft = partial(jnp.fft.fftn, s=kernel_fft_shape, axes=kernel_fft_axes)
    # looping over number of output features
    for i in range(kernel.shape[-1]):
        ki = jnp.expand_dims(kernel.take(i, axis=-1), axis=-1)
        ki = kernel_fft(ki)
        axis_indices = tuple(i if si == 5 else s
                             for si, s in enumerate(indices))
        convolved = convolved.at[axis_indices].set(jnp.squeeze(ifft(y[..., jnp.newaxis] * ki).real, axis=-1))
    return convolved


def looped_fft(inputs: jnp.ndarray, signal_fft_shape: Tuple[int], signal_fft_axes: Tuple[int]) -> jnp.ndarray:
    # y shape (N, D, H, W, C_in)
    y = jnp.zeros((inputs.shape[0],) + signal_fft_shape + (inputs.shape[-1],), dtype=jnp.complex64)
    indices = tuple(slice(0, s) for s in y.shape)
    signal_fft = partial(jnp.fft.fftn, s=signal_fft_shape, axes=signal_fft_axes)
    # looping over number of input features
    for i in range(y.shape[-1]):
        yi = jnp.expand_dims(inputs.take(i, axis=-1), axis=-1)
        yi = jnp.squeeze(signal_fft(yi), axis=-1)
        axis_indices = tuple(i if si == 4 else s
                             for si, s in enumerate(indices))
        y = y.at[axis_indices].set(yi)
    return y


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
    dtype: DType = jnp.float32
    kernel_init: Callable = complex_he_uniform
    bias_init: Callable = fan_in_bias

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
        # extract shapes used to make kernel from input (only matters on first
        # call)
        in_features = inputs.shape[-1]
        kernel_size = inputs.shape[1:-1]
        ndim = len(kernel_size)
        if self.strides is None:
            strides = (1,) * ndim
        else:
            strides = self.strides

        # kernel size must be modified to match padded and strided input size
        if self.padding == "SAME" and strides == ((1,) * ndim):
            kernel_size = tuple(2 * k - 1 for k in kernel_size)
        elif strides != ((1,) * ndim):
            kernel_size = tuple(
                k // (s - 1) for k, s in zip(kernel_size, strides)
            )
        # TODO(dd): reimplement rfft kernel
        # # kernel size must be modified to make last dimension small due to
        # # Hermitian symmetry
        # kernel_size = kernel_size[:-1] + (
        #     kernel_size[-1] // 2 + 1,
        # )
        kernel_shape = kernel_size + (
            in_features,
            self.features,
        )

        # NOTE(dd): the complex kernel must be stored as two real
        # arrays for the gradients to be correct --- do not change the
        # parameterization to be a complex number!
        kernel = self.param("kernel", self.kernel_init(dtype=self.dtype), kernel_shape)
        complex_kernel = lax.complex(kernel[0], kernel[1])

        # Get padded shape to prevent circular convolution
        # TODO(dd): reimplement computing fast shape for FFTs
        fft_shape = tuple(2 * k1 - 1 for k1 in inputs.shape[1 : ndim + 1])
        fft_axes = tuple(range(1, ndim + 1))
        fft = partial(jnp.fft.fftn, axes=fft_axes)
        ifft = partial(jnp.fft.ifftn, axes=fft_axes)

        # transform signal, padding to correct shape
        y = fft(jnp.pad(inputs, tuple((0, ffts - ins) for ins, ffts
                                      in zip(inputs.shape, (inputs.shape[0],) + fft_shape + (inputs.shape[-1],)))))
        if strides != ((1,) * ndim):
            # apply stride by subsampling Fourier transformed input
            total_dims = len(y.shape)
            y = lax.slice(
                y,
                start_indices=tuple(0 for s in y.shape),
                limit_indices=tuple(s for s in y.shape),
                strides=(1,) + strides + (1,),
            )
        # perform Fourier convolution
        y = ifft(y[..., jnp.newaxis] * complex_kernel)

        # get real features
        y = y.real

        if self.padding == "SAME":
            # crop image back to size of input
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

        # sum over input feature dimension
        y = jnp.sum(y, axis=-2)

        if self.use_bias:
            fan_in, _ = _compute_fans(kernel_shape)
            bias = self.param("bias", self.bias_init(fan_in, dtype=self.dtype), (self.features,))
            bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)
            y = y + bias
        return y


class SmallFourierConv(nn.Module):
    """
    Convolution Module implemented using the Fourier convolution
    theorem (for n dimensional input).  Weights are not parameterized
    in Fourier space, so their shape does not need to match the shape
    of the input. Instead, the kernel is Fourier transformed for each
    forward pass and padded to the shape of the input. This is
    intended for performing convolutions that are too large to be
    efficient in direct convolution, but that do not need to have a
    fully global view of the input. Further, this means that the
    kernel parameters themselves will not take up too much memory.

    Args:
      features: number of convolution filters.
      kernel_size: shape of the convolutional kernel, specified as a
        sequence of integers.
      padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
        of `n` `(low, high)` integer pairs that give the padding to apply before
        and after each spatial dimension.
      use_bias: whether to add a bias to the output (default: True).
      dtype: the dtype of the computation (default: float32).
      kernel_init: initializer for the convolutional kernel.
      bias_init: initializer for the bias.
    """

    features: int
    kernel_size: Tuple[int, ...]
    padding: Union[str, Sequence[Tuple[int, ...]]] = "SAME"
    use_bias: bool = True
    dtype: DType = jnp.float32
    kernel_init: Callable = he_uniform
    bias_init: Callable = fan_in_bias

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
        # extract shapes used to make kernel from input (only matters on first
        # call)
        in_features = inputs.shape[-1]
        # spatial shape of kernel is already specified
        kernel_size = self.kernel_size
        ndim = len(kernel_size)
        kernel_shape = kernel_size + (
            in_features,
            self.features,
        )

        kernel = self.param("kernel", self.kernel_init(dtype=self.dtype), kernel_shape)

        # get padded shape to prevent circular convolution
        # TODO(dd): reimplement computing fast shape for FFTs
        signal_fft_shape = tuple(k1 + k2 - 1 for k1, k2 in zip(inputs.shape[1:ndim + 1], kernel_size))
        signal_fft_axes = tuple(range(1, ndim + 1))
        signal_fft = partial(jnp.fft.rfftn, s=signal_fft_shape, axes=signal_fft_axes)
        ifft = partial(jnp.fft.irfftn, s=signal_fft_shape, axes=signal_fft_axes)

        # # transform signal, padding to correct shape
        # y = signal_fft(jnp.pad(inputs, tuple((0, ffts - ins) for ins, ffts
        #                               in zip(inputs.shape, (inputs.shape[0],) + signal_fft_shape + (inputs.shape[-1],)))))
        y = signal_fft(inputs)

        # transform kernel, padding to correct shape 
        kernel_fft_axes = tuple(range(0, ndim))
        # kernel_fft_shape = tuple(k2 for k2 in y.shape[1:ndim + 1])
        kernel_fft = partial(jnp.fft.rfftn, s=signal_fft_shape, axes=kernel_fft_axes)
        complex_kernel = kernel_fft(kernel)

        # perform fourier convolution
        y = ifft(y[..., jnp.newaxis] * complex_kernel)
        # get real features
        # y = y.real

        if self.padding == "SAME":
            # crop image back to size of input
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

        # sum over input feature dimension
        y = jnp.sum(y, axis=-2)

        if self.use_bias:
            fan_in, _ = _compute_fans(kernel_shape)
            bias = self.param("bias", self.bias_init(fan_in, dtype=self.dtype), (self.features,))
            bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)
            y = y + bias
        return y
