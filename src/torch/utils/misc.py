import math
from itertools import repeat

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from learnedLFM.utils.complex import fftconvn


def _list(x, repetitions=1):
    if hasattr(x, "__iter__") and not isinstance(x, str):
        return x
    else:
        return [
            x,
        ] * repetitions


def _ntuple(n):
    """Creates a function enforcing ``x`` to be a tuple of ``n`` elements."""

    def parse(x):
        if isinstance(x, tuple):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
