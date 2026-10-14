from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reciprocal.lattice import Lattice
from reciprocal.sampling import SamplingConstraint

from .domains import PropagationDisk

class CircleArc:
    center: tuple[float, float]
    radius: float
    theta_start: float
    theta_end: float
    def __init__(self, center: tuple[float, float], radius: float, theta_start: float, theta_end: float) -> None: ...
    @property
    def length(self) -> float: ...
    @property
    def closed(self) -> bool: ...
    def points_at(self, angles: ArrayLike) -> NDArray[np.float64]: ...

class WoodAnomalyCircle:
    center: tuple[float, float]
    radius: float
    reciprocal_order: tuple[int, int]
    shell_index: int
    arc: CircleArc | None
    tangent_point: tuple[float, float] | None
    def __init__(self, center: tuple[float, float], radius: float, reciprocal_order: tuple[int, int], shell_index: int, arc: CircleArc | None, tangent_point: tuple[float, float] | None = ...) -> None: ...

class CurveSamplingResult:
    points: NDArray[np.float64]
    physical_weights: NDArray[np.float64]
    circle_indices: NDArray[np.int64]
    point_ids: NDArray[np.int64]
    segment_boundaries: tuple[tuple[int, int], ...]
    circles: tuple[WoodAnomalyCircle, ...]
    tangent_points: NDArray[np.float64]
    tangent_circle_indices: NDArray[np.int64]
    def __init__(self, points: ArrayLike, physical_weights: ArrayLike, circle_indices: ArrayLike, segment_boundaries: tuple[tuple[int, int], ...], circles: tuple[WoodAnomalyCircle, ...], tangent_points: ArrayLike = ..., tangent_circle_indices: ArrayLike = ...) -> None: ...
    @property
    def integration_element(self) -> float: ...
    @property
    def normalized_weights(self) -> NDArray[np.float64]: ...
    @property
    def weights(self) -> NDArray[np.float64]: ...
    @property
    def reciprocal_orders(self) -> NDArray[np.int64]: ...
    def integrate(self, values: ArrayLike) -> NDArray[Any]: ...
    def average(self, values: ArrayLike) -> NDArray[Any]: ...

class WoodAnomalySet:
    circles: tuple[WoodAnomalyCircle, ...]
    propagating_domain: PropagationDisk
    def __init__(self, circles: tuple[WoodAnomalyCircle, ...], propagating_domain: PropagationDisk) -> None: ...
    @property
    def circles_by_shell(self) -> Mapping[int, tuple[WoodAnomalyCircle, ...]]: ...
    def sample_edges(self, constraint: SamplingConstraint = ...) -> CurveSamplingResult: ...

def circle_disk_intersection(center: ArrayLike, radius: float, domain: PropagationDisk) -> tuple[CircleArc | None, tuple[float, float] | None]: ...
def make_wood_anomalies(reciprocal_lattice: Lattice, wavevector_magnitude: float, *, shells: Iterable[int] = ...) -> WoodAnomalySet: ...
