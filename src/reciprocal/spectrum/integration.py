"""Integration measures for reciprocal-spectrum samplings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .model import FieldSamples, KSampling


class IntegrationMeasure(Protocol):
    def weights(self, sampling: KSampling) -> NDArray[np.float64]: ...


@dataclass(frozen=True, slots=True)
class TransverseWavevectorMeasure:
    """The geometric measure ``d kx d ky``."""

    def weights(self, sampling: KSampling) -> NDArray[np.float64]:
        return sampling.physical_weights


@dataclass(frozen=True, slots=True)
class NormalizedAreaMeasure:
    """A domain average rather than a physical-area integral."""

    def weights(self, sampling: KSampling) -> NDArray[np.float64]:
        return sampling.normalized_weights


@dataclass(frozen=True, slots=True)
class SolidAngleMeasure:
    """Solid-angle measure induced by a propagating k-space plane."""

    wavevector_magnitude: float

    def __post_init__(self) -> None:
        value = float(self.wavevector_magnitude)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("wavevector_magnitude must be finite and positive")
        object.__setattr__(self, "wavevector_magnitude", value)

    def weights(self, sampling: KSampling) -> NDArray[np.float64]:
        if sampling.kz is None:
            transverse_squared = np.sum(sampling.points**2, axis=1)
            kz = np.sqrt((self.wavevector_magnitude**2 - transverse_squared).astype(complex))
        else:
            kz = sampling.kz
        if np.any(np.abs(np.imag(kz)) > 1e-12):
            raise ValueError("solid-angle measure is only defined for propagating samples")
        denominator = self.wavevector_magnitude * np.abs(np.real(kz))
        if np.any(denominator == 0.0):
            raise ValueError("solid-angle measure is singular at grazing incidence")
        return sampling.physical_weights / denominator


@dataclass(frozen=True, slots=True)
class JacobianMeasure:
    """Multiply transverse-area weights by a user-provided Jacobian."""

    jacobian: object

    def weights(self, sampling: KSampling) -> NDArray[np.float64]:
        values = np.asarray(self.jacobian(sampling.points), dtype=float)  # type: ignore[operator]
        if values.shape != (len(sampling.points),) or not np.all(np.isfinite(values)):
            raise ValueError("jacobian must return one finite value per point")
        return sampling.physical_weights * values


def integrate(
    sampling: KSampling,
    values: ArrayLike,
    measure: IntegrationMeasure | None = None,
) -> NDArray:
    array = np.asarray(values)
    if array.ndim == 0 or array.shape[0] != len(sampling.points):
        raise ValueError("values must have the sampling length as their first dimension")
    selected = TransverseWavevectorMeasure() if measure is None else measure
    weights = np.asarray(selected.weights(sampling), dtype=float)
    if weights.shape != (len(sampling.points),):
        raise ValueError("integration measure returned invalid weights")
    return np.tensordot(weights, array, axes=(0, 0))


def integrate_field(
    field: FieldSamples,
    measure: IntegrationMeasure | None = None,
) -> NDArray:
    if not np.all(field.valid):
        raise ValueError("cannot integrate invalid field samples")
    return integrate(field.sampling, field.values, measure)


__all__ = [
    "IntegrationMeasure",
    "JacobianMeasure",
    "NormalizedAreaMeasure",
    "SolidAngleMeasure",
    "TransverseWavevectorMeasure",
    "integrate",
    "integrate_field",
]
