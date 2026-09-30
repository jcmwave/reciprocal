from enum import Enum

import numpy as np
from numpy.typing import NDArray

from .brillouin_zone import BrillouinZone
from .numerics import Tolerances
from .sampling import SamplingResult
from .symmetry import PointOperation

class GridCentering(str, Enum):
    GAMMA: GridCentering
    MONKHORST_PACK: GridCentering

class MonkhorstPackGrid:
    shape: tuple[int, int]
    centering: GridCentering
    shift: tuple[float, float]
    def __init__(
        self,
        shape: tuple[int, int],
        centering: GridCentering | str = ...,
        shift: tuple[float, float] = ...,
    ) -> None: ...

class MonkhorstPackMetadata:
    grid: MonkhorstPackGrid
    mesh_indices: NDArray[np.int64]
    fractional_points: NDArray[np.float64]
    canonical_keys: NDArray[np.float64]
    degeneracies: NDArray[np.int64] | None

class MeshReduction:
    full: SamplingResult
    irreducible: SamplingResult
    full_to_irreducible: NDArray[np.int64]
    representative_indices: NDArray[np.int64]
    degeneracies: NDArray[np.int64]
    operations: tuple[PointOperation, ...]
    placement: object | None
    @property
    def reduced(self) -> SamplingResult: ...
    @property
    def full_to_reduced(self) -> NDArray[np.int64]: ...

def reduce_monkhorst_pack(
    zone: BrillouinZone,
    grid: MonkhorstPackGrid,
    *,
    tolerances: Tolerances = ...,
) -> MeshReduction: ...
def sample_monkhorst_pack(
    zone: BrillouinZone,
    grid: MonkhorstPackGrid,
    *,
    irreducible: bool = ...,
    tolerances: Tolerances = ...,
) -> SamplingResult: ...
