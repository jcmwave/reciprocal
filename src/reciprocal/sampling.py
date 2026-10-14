"""Shared immutable sampling contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .integration_cells import (
    CartesianIntegrationCell,
    CircularIntegrationCell,
    IntegrationCell,
    PolarIntegrationCell,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@runtime_checkable
class SamplingDomain(Protocol):
    """A bounded two-dimensional integration domain."""

    @property
    def area(self) -> float: ...

    def contains(self, points: ArrayLike) -> BoolArray: ...

    def bounds(self) -> tuple[float, float, float, float]: ...




@dataclass(frozen=True, slots=True)
class PointCounts:
    first: int
    second: int | None = None
    dimensions: int = field(init=False)

    def __post_init__(self) -> None:
        dimensions = 1 if self.second is None else 2
        second = self.first if self.second is None else self.second
        if isinstance(self.first, (bool, np.bool_)) or isinstance(second, (bool, np.bool_)):
            raise TypeError("point counts must be integers")
        if (
            not np.all(np.isfinite([self.first, second]))
            or int(self.first) != self.first
            or int(second) != second
        ):
            raise TypeError("point counts must be integers")
        if int(self.first) <= 0 or int(second) <= 0:
            raise ValueError("point counts must be positive")
        object.__setattr__(self, "first", int(self.first))
        object.__setattr__(self, "second", int(second))
        object.__setattr__(self, "dimensions", dimensions)

    @property
    def values(self) -> tuple[int, ...]:
        if self.dimensions == 1:
            return (self.first,)
        if self.second is None:
            raise ValueError("second dimension count is missing")
        return (self.first, self.second)

    def as_legacy(self) -> Mapping[str, object]:
        return {
            "type": "n_points",
            "value": self.first if self.dimensions == 1 else (self.first, self.second),
        }


@dataclass(frozen=True, slots=True)
class MaxSpacing:
    value: float | tuple[float, float]

    def __post_init__(self) -> None:
        array = np.asarray(self.value)
        if array.shape not in ((), (2,)) or array.dtype.kind == "b":
            raise ValueError("maximum spacing must be a scalar or a pair")
        array = array.astype(float)
        if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
            raise ValueError("maximum spacing must be finite and positive")
        value = float(array) if array.ndim == 0 else (float(array[0]), float(array[1]))
        object.__setattr__(self, "value", value)

    @property
    def values(self) -> tuple[float, ...]:
        return self.value if isinstance(self.value, tuple) else (self.value,)

    def as_legacy(self) -> Mapping[str, object]:
        return {"type": "max_length", "value": self.value}


SamplingConstraint: TypeAlias = PointCounts | MaxSpacing | Mapping[str, object] | None


def normalize_constraint(
    constraint: SamplingConstraint, *, default: int = 16
) -> PointCounts | MaxSpacing:
    """Normalize typed or legacy constraints, preserving their dimensionality."""
    if constraint is None:
        return PointCounts(default)
    if isinstance(constraint, (PointCounts, MaxSpacing)):
        return constraint
    if not isinstance(constraint, Mapping):
        raise TypeError("constraint must be PointCounts, MaxSpacing, or a legacy mapping")
    value = constraint.get("value")
    if constraint.get("type") == "n_points":
        array = np.asarray(value)
        if array.shape == ():
            return PointCounts(array.item())
        if array.shape == (2,):
            return PointCounts(array[0].item(), array[1].item())
        raise ValueError("point counts must be a scalar or a pair")
    if constraint.get("type") == "max_length":
        return MaxSpacing(value)
    raise ValueError("constraint type must be 'n_points' or 'max_length'")


def resolve_point_counts(
    lengths: ArrayLike,
    constraint: SamplingConstraint = None,
    *,
    placement: str = "centers",
    default: int = 16,
) -> tuple[int, ...]:
    """Resolve 1D/2D counts from physical extents and a common constraint.

    Centers/closed curves use N intervals; endpoint-inclusive grids use N-1.
    A scalar constraint broadcasts over two dimensions.
    """
    extents = np.atleast_1d(np.asarray(lengths, dtype=float))
    if (
        extents.ndim != 1
        or len(extents) not in (1, 2)
        or not np.all(np.isfinite(extents))
        or np.any(extents <= 0)
    ):
        raise ValueError("lengths must contain one or two finite positive extents")
    if placement not in ("centers", "closed", "endpoints"):
        raise ValueError("placement must be centers, closed, or endpoints")
    typed = normalize_constraint(constraint, default=default)
    values = typed.values
    if len(values) != 1 and len(values) != len(extents):
        raise ValueError("constraint dimensions do not match the sampling dimensions")
    values = values * len(extents) if len(values) == 1 else values
    if isinstance(typed, PointCounts):
        return tuple(int(value) for value in values)
    extra = 1 if placement == "endpoints" else 0
    return tuple(
        max(1, int(np.ceil(length / maximum))) + extra for length, maximum in zip(extents, values)
    )


def legacy_constraint(constraint: SamplingConstraint) -> Mapping[str, object] | None:
    if constraint is None:
        return None
    if isinstance(constraint, (PointCounts, MaxSpacing)):
        return constraint.as_legacy()
    return constraint


def _readonly(value: ArrayLike, dtype: Any | None = None) -> NDArray[Any]:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _immutable_metadata(value: object) -> object:
    if isinstance(value, np.ndarray):
        return _readonly(value)
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _immutable_metadata(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_immutable_metadata(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_immutable_metadata(item) for item in value)
    return value


@dataclass(frozen=True, slots=True, eq=False)
class SamplingResult:
    """Points, quadrature weights, integration domain, and stable identities.

    ``normalized_weights`` sum to one. ``physical_weights`` therefore sum to
    ``integration_element``, which is the physical area of ``domain``.
    Optional ``integration_cells`` retain cell geometry and effective
    quadrature areas in point order, without storing plotting artists.
    """

    points: FloatArray
    normalized_weights: FloatArray
    integration_element: float
    domain: SamplingDomain
    point_ids: IntArray | None = None
    kz: ComplexArray | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    integration_cells: tuple[IntegrationCell, ...] | None = None

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float)
        if points.ndim != 2 or points.shape[1] not in (2, 3) or len(points) == 0:
            raise ValueError("points must have shape (N, 2) or (N, 3) with N > 0")
        if not np.all(np.isfinite(points)):
            raise ValueError("points must be finite")
        if points.shape[1] == 3 and np.any(points[:, 2] != 0.0):
            raise ValueError("sampling points must lie in the x-y plane")
        points = points[:, :2]
        if not isinstance(self.domain, SamplingDomain):
            raise TypeError("domain must implement SamplingDomain")

        coordinate_domain = self.metadata.get("coordinate_domain", self.domain)
        if not isinstance(coordinate_domain, SamplingDomain):
            raise TypeError("coordinate_domain metadata must implement SamplingDomain")
        if not np.all(coordinate_domain.contains(points)):
            raise ValueError("all sampling points must lie in the coordinate domain")

        weights = np.asarray(self.normalized_weights, dtype=float)
        if weights.shape != (len(points),):
            raise ValueError("normalized_weights must have shape (N,)")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("normalized_weights must be finite and non-negative")
        if not np.isclose(np.sum(weights), 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("normalized_weights must sum to one")

        integration_element = float(self.integration_element)
        if not np.isfinite(integration_element) or integration_element <= 0.0:
            raise ValueError("integration_element must be finite and positive")
        if not np.isclose(integration_element, self.domain.area, rtol=1e-9, atol=1e-14):
            raise ValueError("integration_element must equal the declared domain area")

        ids = (
            np.arange(len(points), dtype=np.int64)
            if self.point_ids is None
            else np.asarray(self.point_ids)
        )
        if ids.shape != (len(points),) or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("point_ids must have shape (N,) and integer dtype")
        ids = ids.astype(np.int64, copy=False)
        if len(np.unique(ids)) != len(ids):
            raise ValueError("point_ids must be unique")

        kz = None if self.kz is None else np.asarray(self.kz, dtype=complex)
        if kz is not None and (kz.shape != (len(points),) or not np.all(np.isfinite(kz))):
            raise ValueError("kz must contain one finite value per point")

        object.__setattr__(self, "points", _readonly(points, float))
        object.__setattr__(self, "normalized_weights", _readonly(weights, float))
        object.__setattr__(self, "integration_element", integration_element)
        object.__setattr__(self, "point_ids", _readonly(ids, np.int64))
        object.__setattr__(self, "kz", None if kz is None else _readonly(kz, complex))
        if self.integration_cells is not None:
            cells = tuple(self.integration_cells)
            if len(cells) != len(points) or not all(
                isinstance(
                    cell,
                    (
                        CartesianIntegrationCell,
                        CircularIntegrationCell,
                        PolarIntegrationCell,
                                        ),
                )
                for cell in cells
            ):
                raise ValueError("integration_cells must contain one cell per point")
            if not np.allclose(
                [cell.area for cell in cells],
                weights * integration_element,
                rtol=1e-9,
                atol=np.finfo(float).eps * integration_element * 256,
            ):
                raise ValueError("integration cell areas must match physical_weights")
            object.__setattr__(self, "integration_cells", cells)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                {str(key): _immutable_metadata(value) for key, value in self.metadata.items()}
            ),
        )

    @property
    def weights(self) -> FloatArray:
        """Compatibility alias for normalized quadrature weights."""

        return self.normalized_weights

    @property
    def physical_weights(self) -> FloatArray:
        result = self.normalized_weights * self.integration_element
        result.setflags(write=False)
        return result

    @property
    def symmetry_operations(self) -> tuple[object, ...] | None:
        """Compatibility view of global operations stored in metadata."""

        value = self.metadata.get("symmetry_operations")
        return None if value is None else tuple(value)  # type: ignore[arg-type]

    def integrate(self, values: ArrayLike, *, valid: ArrayLike | None = None) -> NDArray[Any]:
        array = np.asarray(values)
        if array.ndim == 0 or array.shape[0] != len(self.points):
            raise ValueError("values must have the sampling length as their first dimension")
        if valid is not None:
            mask = np.asarray(valid, dtype=bool)
            if mask.shape != (len(self.points),):
                raise ValueError("valid must have shape (N,)")
            if not np.all(mask):
                raise ValueError("cannot integrate invalid samples without an imputation policy")
        return np.tensordot(self.physical_weights, array, axes=(0, 0))

    def average(self, values: ArrayLike, *, valid: ArrayLike | None = None) -> NDArray[Any]:
        return self.integrate(values, valid=valid) / self.integration_element

    def with_kz(self, magnitude: float, direction: int = 1) -> SamplingResult:
        magnitude = float(magnitude)
        if not np.isfinite(magnitude) or magnitude <= 0.0:
            raise ValueError("wavevector magnitude must be finite and positive")
        if direction not in (-1, 1):
            raise ValueError("direction must be +1 or -1")
        transverse_squared = np.sum(self.points**2, axis=1)
        kz = direction * np.sqrt((magnitude**2 - transverse_squared).astype(complex))
        return SamplingResult(
            self.points,
            self.normalized_weights,
            self.integration_element,
            self.domain,
            self.point_ids,
            kz,
            self.metadata,
            self.integration_cells,
        )

    def to_kvectors(self, wavelength: float, refractive_index: float, direction: int = 1):
        from reciprocal.kvector import KVectorGroup

        return KVectorGroup.from_transverse(
            wavelength,
            n=np.full(len(self.points), refractive_index),
            kx=self.points[:, 0],
            ky=self.points[:, 1],
            normal=np.full(len(self.points), direction),
            weighting=self.normalized_weights,
        )


__all__ = [
    "MaxSpacing",
    "PointCounts",
    "SamplingConstraint",
    "SamplingDomain",
    "SamplingResult",
    "legacy_constraint",
    "normalize_constraint",
    "resolve_point_counts",
]
