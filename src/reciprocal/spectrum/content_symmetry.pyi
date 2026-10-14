from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reciprocal.sampling import SamplingResult

from .model import FieldSamples

class CartesianPointOperation:
    cartesian: NDArray[np.float64]
    def __init__(self, cartesian: ArrayLike) -> None: ...
    @property
    def determinant(self) -> int: ...
    def apply(self, points: ArrayLike) -> NDArray[np.float64]: ...

class SymmetryCone:
    start_degrees: float
    end_degrees: float
    def __init__(self, start_degrees: float, end_degrees: float) -> None: ...
    def contains(self, points: ArrayLike) -> NDArray[np.bool_]: ...

class ContentSymmetry:
    name: str
    orientation_degrees: float
    rotations: int
    operations: tuple[CartesianPointOperation, ...]
    def __init__(self, name: str, orientation_degrees: float = ...) -> None: ...
    @property
    def cone(self) -> SymmetryCone: ...

class SymmetryConeDomain:
    domain: Any
    cone: SymmetryCone
    def __init__(self, domain: Any, cone: SymmetryCone) -> None: ...
    @property
    def area(self) -> float: ...
    def contains(self, points: ArrayLike) -> NDArray[np.bool_]: ...
    def bounds(self) -> tuple[float, float, float, float]: ...

class NonPeriodicSymmetryReduction:
    target: SamplingResult
    source_points: NDArray[np.float64]
    source_ids: NDArray[np.int64]
    full_to_reduced: NDArray[np.int64]
    operations: tuple[tuple[CartesianPointOperation, ...], ...]
    content_symmetry: ContentSymmetry
    def __init__(self, target: SamplingResult, source_points: ArrayLike, source_ids: ArrayLike, full_to_reduced: ArrayLike, operations: tuple[tuple[CartesianPointOperation, ...], ...], content_symmetry: ContentSymmetry) -> None: ...
    @property
    def speedup(self) -> float: ...
    def expand(self, data: FieldSamples[Any], *, verify: bool = ...) -> FieldSamples[Any]: ...

def prepare_symmetric_grid(domain: Any, grid: Any, symmetry: ContentSymmetry, *, adapt: bool) -> Any: ...
def reduce_sampling(full: SamplingResult, symmetry: ContentSymmetry, cone: SymmetryCone | tuple[float, float] | str) -> SamplingResult: ...
