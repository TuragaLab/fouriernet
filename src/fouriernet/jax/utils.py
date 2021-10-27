import jax.numpy as jnp
from jax.nn.initializers import lecun_normal
from jax.lax import complex
import numpy as np

from typing import Callable, Iterable


def next_order(val: int) -> int:
    return int(2 ** np.ceil(np.log2(val)))


def complex_lecun_normal(*args, **kwargs) -> Callable:
    """Thin wrapper so lecun normal returns complex number."""

    def _init(key, shape: Iterable[int]) -> jnp.ndarray:
        return complex(*normal(key, (2, *shape)))

    normal = lecun_normal(*args, **kwargs)
    return _init

