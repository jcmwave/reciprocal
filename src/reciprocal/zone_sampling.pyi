from enum import Enum
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .brillouin_zone import BrillouinZone
from .numerics import Tolerances
from .reciprocal_mesh import MeshReduction, MonkhorstPackGrid
from .sampling import MaxSpacing, PointCounts, SamplingResult
from .symmetry import PointOperation

class ZoneSamplingError(ValueError): ...
class IncompatibleGridSymmetryError(ZoneSamplingError): ...
class CanonicalPlacementError(ZoneSamplingError): ...

class ZoneRegion(str, Enum):
    BZ: ZoneRegion
    IRREDUCIBLE: ZoneRegion

class RepresentativePlacement(str, Enum):
    COMPUTATIONAL: RepresentativePlacement
    CANONICAL_IBZ: RepresentativePlacement

class BoundaryGrid:
    constraint: PointCounts | MaxSpacing
    center: tuple[float, float]
    def __init__(
        self,
        constraint: PointCounts | MaxSpacing = ...,
        center: tuple[float, float] = ...,
    ) -> None: ...

ZoneGrid: TypeAlias = BoundaryGrid | MonkhorstPackGrid
ZoneReduction: TypeAlias = MeshReduction

class RepresentativeMap:
    source_indices: NDArray[np.int64]
    operations: tuple[PointOperation, ...]
    reciprocal_shifts: NDArray[np.int64]
    source_points: NDArray[np.float64]
    placed_points: NDArray[np.float64]
    def __init__(
        self,
        source_indices: NDArray[np.int64],
        operations: tuple[PointOperation, ...],
        reciprocal_shifts: NDArray[np.int64],
        source_points: NDArray[np.float64],
        placed_points: NDArray[np.float64],
    ) -> None: ...

class GridSymmetry:
    lattice_operations: tuple[PointOperation, ...]
    preserving_operations: tuple[PointOperation, ...]
    def __init__(
        self,
        lattice_operations: tuple[PointOperation, ...],
        preserving_operations: tuple[PointOperation, ...],
    ) -> None: ...
    @property
    def preserves_full_group(self) -> bool: ...

def grid_symmetry(
    zone: BrillouinZone,
    grid: ZoneGrid,
    *,
    tolerances: Tolerances = ...,
) -> GridSymmetry: ...
def reduce_brillouin_zone(
    zone: BrillouinZone,
    grid: ZoneGrid,
    *,
    placement: RepresentativePlacement | str = ...,
    require_full_symmetry: bool = ...,
    tolerances: Tolerances = ...,
) -> ZoneReduction: ...
def sample_brillouin_zone(
    zone: BrillouinZone,
    grid: ZoneGrid,
    *,
    region: ZoneRegion | str = ...,
    placement: RepresentativePlacement | str | None = ...,
    require_full_symmetry: bool = ...,
    tolerances: Tolerances = ...,
) -> SamplingResult: ...
