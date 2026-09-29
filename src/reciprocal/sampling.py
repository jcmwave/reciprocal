"""Shared immutable sampling contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

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

    def __post_init__(self) -> None:
        second = self.first if self.second is None else self.second
        if isinstance(self.first, bool) or isinstance(second, bool):
            raise TypeError("point counts must be integers")
        if int(self.first) != self.first or int(second) != second:
            raise TypeError("point counts must be integers")
        if int(self.first) <= 0 or int(second) <= 0:
            raise ValueError("point counts must be positive")
        object.__setattr__(self, "first", int(self.first))
        object.__setattr__(self, "second", int(second))

    def as_legacy(self) -> Mapping[str, object]:
        return {"type": "n_points", "value": (self.first, self.second)}


@dataclass(frozen=True, slots=True)
class MaxSpacing:
    value: float

    def __post_init__(self) -> None:
        value = float(self.value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("maximum spacing must be finite and positive")
        object.__setattr__(self, "value", value)

    def as_legacy(self) -> Mapping[str, object]:
        return {"type": "max_length", "value": self.value}


SamplingConstraint: TypeAlias = PointCounts | MaxSpacing | Mapping[str, object] | None


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
    """

    points: FloatArray
    normalized_weights: FloatArray
    integration_element: float
    domain: SamplingDomain
    point_ids: IntArray | None = None
    kz: ComplexArray | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

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

        ids = np.arange(len(points), dtype=np.int64) if self.point_ids is None else np.asarray(
            self.point_ids
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
]
