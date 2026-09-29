"""Structured and scattered interpolation of sampled spectrum data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    NearestNDInterpolator,
)
from scipy.spatial import Delaunay, QhullError

from .model import FieldSamples, KSampling
from .periodic import ExpansionMap


@dataclass(frozen=True, slots=True, eq=False)
class InterpolationResult:
    points: NDArray[np.float64]
    values: NDArray
    valid: NDArray[np.bool_]

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float)
        values = np.asarray(self.values)
        valid = np.asarray(self.valid, dtype=bool)
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError("points must have shape (N, 2) or (N, 3)")
        if values.ndim == 0 or values.shape[0] != len(points):
            raise ValueError("values must have one row per point")
        if valid.shape != (len(points),):
            raise ValueError("valid must have shape (N,)")
        points = np.array(points[:, :2], copy=True)
        values = np.array(values, copy=True)
        valid = np.array(valid, copy=True)
        points.setflags(write=False)
        values.setflags(write=False)
        valid.setflags(write=False)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid", valid)


def _target_points(target: KSampling | ArrayLike) -> np.ndarray:
    values = target.points if isinstance(target, KSampling) else np.asarray(target, dtype=float)
    if values.ndim != 2 or values.shape[1] not in (2, 3) or not np.all(np.isfinite(values)):
        raise ValueError("target must contain finite points with shape (N, 2) or (N, 3)")
    return np.asarray(values[:, :2], dtype=float)


def _periodic_copies(
    points: np.ndarray,
    values: np.ndarray,
    basis: ArrayLike | None,
) -> tuple[np.ndarray, np.ndarray]:
    if basis is None:
        return points, values
    matrix = np.asarray(basis, dtype=float)
    if matrix.shape == (2, 3):
        matrix = matrix[:, :2]
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError("reciprocal_basis must have shape (2, 2) or (2, 3)")
    orders = np.array([(first, second) for first in (-1, 0, 1) for second in (-1, 0, 1)])
    shifts = orders @ matrix
    augmented_points = np.vstack([points + shift for shift in shifts])
    augmented_values = np.concatenate([values] * len(shifts), axis=0)
    return augmented_points, augmented_values


def interpolate(
    field: FieldSamples,
    target: KSampling | ArrayLike,
    *,
    method: str = "linear",
    extrapolation: str = "mask",
    reciprocal_basis: ArrayLike | None = None,
    expansion: ExpansionMap | None = None,
) -> InterpolationResult:
    """Interpolate real, complex, scalar, vector, or tensor sample values.

    Supplying ``expansion`` performs symmetry-aware interpolation by expanding
    representatives once before triangulation. Supplying ``reciprocal_basis``
    adds the nearest reciprocal copies and is appropriate only when the values
    are already in a translation-invariant representation.
    """

    if expansion is not None:
        field = expansion.expand(field)
    query = _target_points(target)
    source_mask = np.asarray(field.valid, dtype=bool)
    points = field.sampling.points[source_mask]
    values = field.values[source_mask]
    if len(points) == 0:
        raise ValueError("no valid source samples are available")
    points, values = _periodic_copies(points, values, reciprocal_basis)
    trailing_shape = values.shape[1:]
    flat = values.reshape(len(values), -1)
    method = method.lower()
    if method not in {"linear", "nearest", "clough_tocher"}:
        raise ValueError("method must be 'linear', 'nearest', or 'clough_tocher'")
    if extrapolation not in {"mask", "nearest"}:
        raise ValueError("extrapolation must be 'mask' or 'nearest'")

    if method == "nearest":
        interpolator = NearestNDInterpolator(points, flat)
    elif method == "linear":
        interpolator = LinearNDInterpolator(points, flat, fill_value=np.nan)
    else:
        interpolator = CloughTocher2DInterpolator(points, flat, fill_value=np.nan)
    result = np.asarray(interpolator(query))
    if result.ndim == 1:
        result = result[:, None]
    valid = np.all(np.isfinite(result), axis=1)
    if extrapolation == "nearest" and not np.all(valid):
        nearest = NearestNDInterpolator(points, flat)
        result[~valid] = nearest(query[~valid])
        valid[:] = True
    elif method == "nearest" and extrapolation == "mask":
        try:
            valid = Delaunay(points).find_simplex(query) >= 0
        except QhullError:
            valid = np.zeros(len(query), dtype=bool)
        result[~valid] = np.nan
    result = result.reshape((len(query),) + trailing_shape)
    return InterpolationResult(query, result, valid)


__all__ = ["InterpolationResult", "interpolate"]
