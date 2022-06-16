import jax.numpy as jnp
from jax.nn.initializers import lecun_normal, he_uniform, variance_scaling
from jax.lax import complex
from jax.core import NamedShape
from jax import random
import numpy as np

from functools import partial
from typing import Callable, Iterable


def next_order(val: int) -> int:
    return int(2 ** np.ceil(np.log2(val)))


def _compute_fans(shape: NamedShape, in_axis=-2, out_axis=-1,
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
  receptive_field_size = shape.total / in_size / out_size / batch_size
  fan_in = in_size * receptive_field_size
  fan_out = out_size * receptive_field_size
  return fan_in, fan_out


def complex_lecun_normal(*args, **kwargs) -> Callable:
    """Thin wrapper so lecun_normal() returns complex number."""

    def _init(key, shape: Iterable[int]) -> jnp.ndarray:
        return complex(*normal(key, (2, *shape)))

    normal = lecun_normal(*args, **kwargs)
    return _init


def complex_he_uniform(*args, **kwargs) -> Callable:
    """Thin wrapper so he_uniform() returns complex number."""

    def _init(key, shape: Iterable[int]) -> jnp.ndarray:
        return complex(*uniform(key, (2, *shape)))

    uniform = variance_scaling(2.0 / 3.0, "fan_in", "uniform", *args, **kwargs)
    return _init


def complex_fan_in_bias(*args, **kwargs) -> Callable:

    def _init(key, shape: NamedShape) -> jnp.ndarray:
        fan_in, _ = _compute_fans(shape, in_axis=-1)
        bound = 1.0 / jnp.sqrt(fan_in)
        return complex(*random.uniform(key, (2, *shape), minval=-bound, maxval=bound))

    return _init

def real_fan_in_bias(*args, **kwargs) -> Callable:

    def _init(key, shape: NamedShape) -> jnp.ndarray:
        fan_in, _ = _compute_fans(shape, in_axis=-1)
        bound = 1.0 / jnp.sqrt(fan_in)
        return random.uniform(key, shape, minval=-bound, maxval=bound)

    return _init
