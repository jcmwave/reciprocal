"""Immutable integration-cell descriptions, independent of Matplotlib."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

if TYPE_CHECKING:
    from .sampling import SamplingDomain


@dataclass(frozen=True, slots=True)
class CartesianIntegrationCell:
    """A full grid rectangle centered on a Cartesian sampling point.

    ``bounds`` are ``(x_min, x_max, y_min, y_max)``. ``area`` is the
    effective quadrature area (domain area divided by the retained point
    count), while ``geometric_area`` is the rectangle's width times height.
    The rectangle is not clipped to ``domain`` at sampling boundaries.
    """

    bounds: tuple[float, float, float, float]
    domain: SamplingDomain
    area: float

    def __post_init__(self) -> None:
        bounds = tuple(float(value) for value in self.bounds)
        if len(bounds) != 4 or not np.all(np.isfinite(bounds)):
            raise ValueError("cell bounds must contain four finite values")
        if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
            raise ValueError("cell bounds must have positive width and height")
        area = float(self.area)
        if not np.isfinite(area) or area <= 0.0:
            raise ValueError("cell area must be finite and positive")
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "area", area)

    @property
    def geometric_area(self) -> float:
        x0, x1, y0, y1 = self.bounds
        return (x1 - x0) * (y1 - y0)


@dataclass(frozen=True, slots=True)
class CircularIntegrationCell:
    """A circular integration region around the origin sample."""

    radius: float
    center: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        radius = float(self.radius)
        center = tuple(float(value) for value in self.center)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("circle radius must be finite and positive")
        if len(center) != 2 or not np.all(np.isfinite(center)):
            raise ValueError("circle center must be a finite pair")
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "center", center)

    @property
    def area(self) -> float:
        return float(np.pi * self.radius**2)


@dataclass(frozen=True, slots=True)
class PolarIntegrationCell:
    """An annular sector, with counterclockwise angles in radians."""

    center: tuple[float, float]
    inner_radius: float
    outer_radius: float
    theta_start: float
    theta_end: float

    def __post_init__(self) -> None:
        center = tuple(float(value) for value in self.center)
        values = tuple(
            float(value)
            for value in (self.inner_radius, self.outer_radius, self.theta_start, self.theta_end)
        )
        if len(center) != 2 or not np.all(np.isfinite((*center, *values))):
            raise ValueError("sector geometry must be finite")
        inner, outer, start, end = values
        if (
            not 0.0 <= inner < outer
            or not 0.0 < end - start <= 2.0 * np.pi + np.finfo(float).eps * 16
        ):
            raise ValueError("sector requires ordered radii and an angle in (0, 2*pi]")
        object.__setattr__(self, "center", center)
        for name, value in zip(
            ("inner_radius", "outer_radius", "theta_start", "theta_end"), values
        ):
            object.__setattr__(self, name, value)

    @property
    def area(self) -> float:
        return (
            0.5
            * (self.outer_radius**2 - self.inner_radius**2)
            * (self.theta_end - self.theta_start)
        )


@dataclass(frozen=True, slots=True)
class OrbitIntegrationCell:
    """A representative footprint with its full-orbit quadrature allocation."""

    footprint: CartesianIntegrationCell | CircularIntegrationCell | PolarIntegrationCell
    area: float

    def __post_init__(self) -> None:
        if not isinstance(
            self.footprint,
            (CartesianIntegrationCell, CircularIntegrationCell, PolarIntegrationCell),
        ):
            raise TypeError("footprint must be an integration cell")
        if not np.isfinite(self.area) or self.area <= 0:
            raise ValueError("orbit area must be finite and positive")

    @property
    def geometric_area(self) -> float:
        if isinstance(self.footprint, CartesianIntegrationCell):
            return self.footprint.geometric_area
        return self.footprint.area


IntegrationCell: TypeAlias = (
    CartesianIntegrationCell | CircularIntegrationCell | PolarIntegrationCell | OrbitIntegrationCell
)

__all__ = [
    "CartesianIntegrationCell",
    "CircularIntegrationCell",
    "IntegrationCell",
    "PolarIntegrationCell",
    "OrbitIntegrationCell",
]
