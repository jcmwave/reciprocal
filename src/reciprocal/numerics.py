"""Numerical policies shared by geometry and sampling algorithms.

Relative tolerances compare dimensionless ratios or values at the scale of the
input.  Absolute tolerances are reserved for quantities with fixed units, such
as angles in degrees.  Callers may pass a different :class:`Tolerances` object
when their data resolution requires it.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True, slots=True)
class Tolerances:
    """Scale-aware tolerances used by the public geometry API.

    Attributes
    ----------
    relative:
        Relative tolerance for lengths and Cartesian coordinates.
    absolute:
        Absolute floor in the coordinate unit. Keep this at zero when inputs
        span very different physical scales.
    angle_degrees:
        Absolute tolerance for angles expressed in degrees.
    degeneracy:
        Minimum dimensionless ``|a x b| / (|a| |b|)`` for a valid basis.
    boundary:
        Relative allowance used when testing a radial domain boundary.
    """

    relative: float = 1e-9
    absolute: float = 0.0
    angle_degrees: float = 1e-8
    degeneracy: float = 1e-12
    boundary: float = 1e-9

    def __post_init__(self) -> None:
        values = (
            self.relative,
            self.absolute,
            self.angle_degrees,
            self.degeneracy,
            self.boundary,
        )
        if not all(np.isfinite(value) and value >= 0 for value in values):
            raise ValueError("numerical tolerances must be finite and non-negative")


DEFAULT_TOLERANCES = Tolerances()


def contains_close_point(
    points: ArrayLike,
    point: ArrayLike,
    *,
    relative_tolerance: float = DEFAULT_TOLERANCES.relative,
    absolute_tolerance: float = DEFAULT_TOLERANCES.absolute,
) -> bool:
    """Return whether an ``(N, D)`` array contains a coordinate-wise match.

    The implementation performs one vectorized comparison instead of scanning
    Python rows. Empty inputs return ``False``. Shape mismatches are rejected
    rather than accidentally broadcasting.
    """

    existing = np.asarray(points, dtype=float)
    candidate = np.asarray(point, dtype=float)
    if existing.size == 0:
        return False
    existing = np.atleast_2d(existing)
    if candidate.ndim != 1 or existing.shape[1] != candidate.shape[0]:
        raise ValueError("points must have shape (N, D) and point shape (D,)")
    close = np.isclose(
        existing,
        candidate,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
    return bool(np.any(np.all(close, axis=1)))
