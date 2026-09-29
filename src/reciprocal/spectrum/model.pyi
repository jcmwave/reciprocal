from enum import Enum
from typing import Any, Generic, Protocol, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..sampling import MaxSpacing as MaxSpacing
from ..sampling import PointCounts as PointCounts
from ..sampling import SamplingConstraint as SamplingConstraint
from ..sampling import SamplingResult
from ..sampling import legacy_constraint as legacy_constraint

T = TypeVar("T")
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

class SourceRegion(str, Enum):
    BZ: SourceRegion
    IBZ: SourceRegion
    PHYSICAL_DOMAIN: SourceRegion

class TargetRegion(str, Enum):
    BZ: TargetRegion
    PROPAGATING_SPECTRUM: TargetRegion
    EXTENDED_SPECTRUM: TargetRegion

class CartesianGrid:
    shape: tuple[int, int]
    def __init__(self, shape: tuple[int, int]) -> None: ...

class PolarGrid:
    radial: int
    azimuthal: int
    def __init__(self, radial: int, azimuthal: int) -> None: ...

class ValueRepresentation(Protocol[T]):
    def transform(self, value: T, operation: object, source_k: FloatArray, target_k: FloatArray, reciprocal_order: tuple[int, int]) -> T: ...

KSampling = SamplingResult

class FieldSamples(Generic[T]):
    sampling: KSampling
    values: NDArray[Any]
    representation: ValueRepresentation[T]
    valid: BoolArray
    units: object | None
    def __init__(self, sampling: KSampling, values: NDArray[Any], representation: ValueRepresentation[T], valid: BoolArray | None = ..., units: object | None = ...) -> None: ...
    def integrate(self, measure: object | None = ...) -> NDArray[Any]: ...
    def average(self) -> NDArray[Any]: ...
    def interpolate(self, target: KSampling | ArrayLike, **kwargs: object) -> Any: ...
