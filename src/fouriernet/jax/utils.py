import jax.numpy as jnp
from jax.lax import complex
from jax import random
import numpy as np

import math
from functools import partial
from typing import Callable, Iterable, Any

DType = Any

def next_order(val: int) -> int:
    return int(2 ** np.ceil(np.log2(val)))


def _compute_fans(shape: tuple[int, ...], in_axis=-2, out_axis=-1,
                  batch_axis=()):
  """
  From jax._src.nn.initializers
  Compute effective input and output sizes for a linear or convolutional layer.

  Axes not in in_axis, out_axis, or batch_axis are assumed to constitute the
  "receptive field" of a convolution (kernel spatial dimensions).
  """
  if isinstance(in_axis, int):
    in_size = shape[in_axis]
  else:
    in_size = int(np.prod([shape[i] for i in in_axis]))
  if isinstance(out_axis, int):
    out_size = shape[out_axis]
  else:
    out_size = int(np.prod([shape[i] for i in out_axis]))
  if isinstance(batch_axis, int):
    batch_size = shape[batch_axis]
  else:
    batch_size = int(np.prod([shape[i] for i in batch_axis]))
  receptive_field_size = math.prod(shape) / in_size / out_size / batch_size
  fan_in = in_size * receptive_field_size
  fan_out = out_size * receptive_field_size
  return fan_in, fan_out


def he_uniform(dtype: DType = jnp.float32, scale: float = 1.0, shift: float = 0.0) -> Callable:
    """
    He/Kaiming uniform initialization that matches PyTorch for linear layers.
    """
    def _init(key, shape: tuple[int, ...], dtype=dtype) -> jnp.ndarray:
        fan_in, _ = _compute_fans(shape)
        a = jnp.sqrt(5)
        gain = jnp.sqrt(2.0 / (1 + a ** 2))
        std = gain / jnp.sqrt(fan_in)
        bound = jnp.sqrt(3.0) * std
        return scale * random.uniform(key, shape, dtype, minval=-bound, maxval=bound) + shift
    return _init


def complex_he_uniform(dtype: DType = jnp.float32) -> Callable:
    """
    Version of He/Kaiming uniform linear layer initialization for a
    complex number. Note that this doesn't actually return a complex
    number data type! This is intended for use with complex number
    parameters that are actually stored as two real arrays. Complex
    parameters are stored in this way to ensure gradients are in the
    correct direction.
    """
    def _init(key, shape: tuple[int, ...], dtype=dtype) -> jnp.ndarray:
        fan_in, fan_out = _compute_fans(shape)
        a = jnp.sqrt(5)
        gain = jnp.sqrt(2.0 / (1 + a ** 2))
        std = gain / jnp.sqrt(fan_in)
        bound = jnp.sqrt(3.0) * std
        return random.uniform(key, (2, *shape), dtype, minval=-bound, maxval=bound)
    return _init


def fan_in_bias(fan_in: int, dtype: DType = jnp.float32, scale: float = 1.0, shift: float = 0.0) -> Callable:
    def _init(key, shape: tuple[int, ...], dtype=dtype) -> jnp.ndarray:
        bound = 1.0 / jnp.sqrt(fan_in)
        return scale * random.uniform(key, shape, dtype=dtype, minval=-bound, maxval=bound) + shift
    return _init

