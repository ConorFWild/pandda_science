from typing import NamedTuple, Tuple

import numpy as np

class XMap:
    pass


class Grid(NamedTuple):
    origin: np.ndarray
    rotation: np.ndarray
    shape: np.ndarray
    scale: np.ndarray

