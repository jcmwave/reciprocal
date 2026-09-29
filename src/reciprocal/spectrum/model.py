"""Immutable reciprocal-spectrum sampling and data contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Protocol, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reciprocal.sampling import (
    MaxSpacing,
    PointCounts,
    SamplingConstraint,
    SamplingResult,
    legacy_constraint,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
T = TypeVar("T")


def _readonly(value: ArrayLike, dtype: Any | None = None) -> NDArray[Any]:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


class SourceRegion(str, Enum):
    BZ = "bz"
    IBZ = "ibz"
    PHYSICAL_DOMAIN = "physical_domain"


class TargetRegion(str, Enum):
    BZ = "bz"
    PROPAGATING_SPECTRUM = "propagating_spectrum"
    EXTENDED_SPECTRUM = "extended_spectrum"


@dataclass(frozen=True, slots=True)
class CartesianGrid:
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or any(isinstance(value, bool) for value in self.shape):
            raise ValueError("CartesianGrid shape must contain two positive integers")
        shape = tuple(int(value) for value in self.shape)
        if tuple(self.shape) != shape or any(value <= 0 for value in shape):
            raise ValueError("CartesianGrid shape must contain two positive integers")
        object.__setattr__(self, "shape", shape)


@dataclass(frozen=True, slots=True)
class PolarGrid:
    radial: int
    azimuthal: int

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) for value in (self.radial, self.azimuthal)):
            raise TypeError("polar grid counts must be integers")
        if int(self.radial) != self.radial or int(self.azimuthal) != self.azimuthal:
            raise TypeError("polar grid counts must be integers")
        if int(self.radial) <= 0 or int(self.azimuthal) <= 0:
            raise ValueError("polar grid counts must be positive")
        object.__setattr__(self, "radial", int(self.radial))
        object.__setattr__(self, "azimuthal", int(self.azimuthal))


class ValueRepresentation(Protocol[T]):
    def transform(
        self,
        value: T,
        operation: object,
        source_k: FloatArray,
        target_k: FloatArray,
        reciprocal_order: tuple[int, int],
    ) -> T: ...


KSampling = SamplingResult


@dataclass(frozen=True, slots=True, eq=False)
class FieldSamples(Generic[T]):
    sampling: KSampling
    values: NDArray[Any]
    representation: ValueRepresentation[T]
    valid: BoolArray | None = None
    units: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sampling, KSampling):
            raise TypeError("sampling must be a KSampling")
        values = np.asarray(self.values)
        if values.ndim == 0 or values.shape[0] != len(self.sampling.points):
            raise ValueError("values must have the sampling length as their first dimension")
        if not hasattr(self.representation, "transform"):
            raise TypeError("representation must implement transform()")
        valid = (
            np.ones(len(self.sampling.points), dtype=bool)
            if self.valid is None
            else np.asarray(self.valid, dtype=bool)
        )
        if valid.shape != (len(self.sampling.points),):
            raise ValueError("valid must have shape (N,)")
        object.__setattr__(self, "values", _readonly(values))
        object.__setattr__(self, "valid", _readonly(valid, bool))

    def integrate(self, measure: object | None = None) -> NDArray[Any]:
        if not np.all(self.valid):
            raise ValueError("cannot integrate invalid field samples")
        if measure is None:
            return self.sampling.integrate(self.values, valid=self.valid)
        from .integration import integrate

        return integrate(self.sampling, self.values, measure)  # type: ignore[arg-type]

    def average(self) -> NDArray[Any]:
        return self.sampling.average(self.values, valid=self.valid)

    def interpolate(self, target: KSampling | ArrayLike, **kwargs: object):
        from .interpolation import interpolate

        return interpolate(self, target, **kwargs)  # type: ignore[arg-type]


__all__ = [
    "CartesianGrid",
    "FieldSamples",
    "KSampling",
    "MaxSpacing",
    "PointCounts",
    "PolarGrid",
    "SamplingConstraint",
    "SourceRegion",
    "TargetRegion",
    "ValueRepresentation",
    "legacy_constraint",
]
