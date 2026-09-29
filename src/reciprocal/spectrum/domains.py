"""Bounded transverse-wavevector domains.

The classes in this module describe geometry only.  They deliberately do not
know about samplers, optical fields, or plotting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reciprocal.cells.geometry import contains_points, intersect_convex_polygons
from reciprocal.cells.model import PolygonDomain

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _points(value: ArrayLike) -> FloatArray:
    points = np.asarray(value, dtype=float)
    if points.ndim == 1:
        points = points[None, :]
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError("points must have shape (N, 2) or (N, 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must be finite")
    return points[:, :2]


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@runtime_checkable
class KDomain(Protocol):
    """A finite two-dimensional integration domain."""

    @property
    def area(self) -> float: ...

    def contains(self, points: ArrayLike) -> BoolArray: ...

    def bounds(self) -> tuple[float, float, float, float]: ...


@dataclass(frozen=True, slots=True)
class PropagationDisk:
    """The propagating transverse spectrum ``|k_parallel-center| <= radius``."""

    radius: float
    center: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "radius", _positive(self.radius, "radius"))
        center = np.asarray(self.center, dtype=float)
        if center.shape != (2,) or not np.all(np.isfinite(center)):
            raise ValueError("center must be a finite pair")
        object.__setattr__(self, "center", (float(center[0]), float(center[1])))

    @property
    def area(self) -> float:
        return float(np.pi * self.radius**2)

    def contains(self, points: ArrayLike) -> BoolArray:
        relative = _points(points) - np.asarray(self.center)
        tolerance = np.finfo(float).eps * self.radius * 64
        return np.linalg.norm(relative, axis=1) <= self.radius + tolerance

    def bounds(self) -> tuple[float, float, float, float]:
        x, y = self.center
        return x - self.radius, x + self.radius, y - self.radius, y + self.radius


@dataclass(frozen=True, slots=True)
class EvanescentDisk:
    """A finite spectrum containing propagating and evanescent wave vectors."""

    propagating_radius: float
    max_parallel_wavevector: float
    center: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        propagating = _positive(self.propagating_radius, "propagating_radius")
        maximum = _positive(self.max_parallel_wavevector, "max_parallel_wavevector")
        if maximum <= propagating:
            raise ValueError("max_parallel_wavevector must exceed propagating_radius")
        object.__setattr__(self, "propagating_radius", propagating)
        object.__setattr__(self, "max_parallel_wavevector", maximum)
        center = np.asarray(self.center, dtype=float)
        if center.shape != (2,) or not np.all(np.isfinite(center)):
            raise ValueError("center must be a finite pair")
        object.__setattr__(self, "center", (float(center[0]), float(center[1])))

    @property
    def area(self) -> float:
        return float(np.pi * self.max_parallel_wavevector**2)

    def contains(self, points: ArrayLike) -> BoolArray:
        relative = _points(points) - np.asarray(self.center)
        tolerance = np.finfo(float).eps * self.max_parallel_wavevector * 64
        return np.linalg.norm(relative, axis=1) <= self.max_parallel_wavevector + tolerance

    def is_evanescent(self, points: ArrayLike) -> BoolArray:
        relative = _points(points) - np.asarray(self.center)
        return np.linalg.norm(relative, axis=1) > self.propagating_radius

    def bounds(self) -> tuple[float, float, float, float]:
        x, y = self.center
        radius = self.max_parallel_wavevector
        return x - radius, x + radius, y - radius, y + radius


@dataclass(frozen=True, slots=True)
class PupilDomain:
    """A circular or annular pupil in transverse-wavevector coordinates."""

    outer_radius: float
    inner_radius: float = 0.0
    center: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        outer = _positive(self.outer_radius, "outer_radius")
        inner = float(self.inner_radius)
        if not np.isfinite(inner) or inner < 0.0 or inner >= outer:
            raise ValueError(
                "inner_radius must be finite, non-negative, and smaller than outer_radius"
            )
        center = np.asarray(self.center, dtype=float)
        if center.shape != (2,) or not np.all(np.isfinite(center)):
            raise ValueError("center must be a finite pair")
        object.__setattr__(self, "outer_radius", outer)
        object.__setattr__(self, "inner_radius", inner)
        object.__setattr__(self, "center", (float(center[0]), float(center[1])))

    @property
    def area(self) -> float:
        return float(np.pi * (self.outer_radius**2 - self.inner_radius**2))

    def contains(self, points: ArrayLike) -> BoolArray:
        radius = np.linalg.norm(_points(points) - np.asarray(self.center), axis=1)
        tolerance = np.finfo(float).eps * self.outer_radius * 64
        return (radius >= self.inner_radius - tolerance) & (
            radius <= self.outer_radius + tolerance
        )

    def bounds(self) -> tuple[float, float, float, float]:
        x, y = self.center
        return (
            x - self.outer_radius,
            x + self.outer_radius,
            y - self.outer_radius,
            y + self.outer_radius,
        )


@dataclass(frozen=True, slots=True)
class PolygonKDomain:
    """Adapt a :class:`PolygonDomain` to the spectrum-domain protocol."""

    polygon: PolygonDomain

    def __post_init__(self) -> None:
        if not isinstance(self.polygon, PolygonDomain):
            raise TypeError("polygon must be a PolygonDomain")

    @property
    def area(self) -> float:
        return self.polygon.area

    def contains(self, points: ArrayLike) -> BoolArray:
        return contains_points(self.polygon, points)

    def bounds(self) -> tuple[float, float, float, float]:
        vertices = self.polygon.vertices[:, :2]
        return (
            float(np.min(vertices[:, 0])),
            float(np.max(vertices[:, 0])),
            float(np.min(vertices[:, 1])),
            float(np.max(vertices[:, 1])),
        )


def _cross(first: FloatArray, second: FloatArray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _segment_circle_parameters(first: FloatArray, second: FloatArray, radius: float) -> list[float]:
    direction = second - first
    a = float(np.dot(direction, direction))
    b = 2.0 * float(np.dot(first, direction))
    c = float(np.dot(first, first) - radius * radius)
    discriminant = b * b - 4.0 * a * c
    parameters = [0.0, 1.0]
    if discriminant > 0.0 and a > 0.0:
        root = np.sqrt(discriminant)
        for value in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)):
            if 0.0 < value < 1.0:
                parameters.append(float(value))
    return sorted(parameters)


def _disk_polygon_area(disk: PropagationDisk, polygon: PolygonDomain) -> float:
    """Return the exact line/sector area of a circle-convex-polygon intersection."""

    vertices = polygon.vertices[:, :2] - np.asarray(disk.center)
    radius = disk.radius
    signed_area = 0.0
    for index, first in enumerate(vertices):
        second = vertices[(index + 1) % len(vertices)]
        direction = second - first
        parameters = _segment_circle_parameters(first, second, radius)
        for lower, upper in zip(parameters[:-1], parameters[1:]):
            start = first + lower * direction
            end = first + upper * direction
            midpoint = 0.5 * (start + end)
            if np.linalg.norm(midpoint) <= radius:
                signed_area += 0.5 * _cross(start, end)
            else:
                angle = float(np.arctan2(_cross(start, end), np.dot(start, end)))
                signed_area += 0.5 * radius * radius * angle
    return abs(signed_area)


def _as_disk(domain: KDomain) -> PropagationDisk | None:
    if isinstance(domain, PropagationDisk):
        return domain
    if isinstance(domain, EvanescentDisk):
        return PropagationDisk(domain.max_parallel_wavevector, domain.center)
    return None


def intersection_area(domain: KDomain, polygon: PolygonDomain) -> float:
    """Return the area of ``domain`` intersected with a convex polygon.

    This helper is used to construct quadrature weights.  All built-in
    primitive domains have an analytic or polygon-clipping implementation.
    """

    disk = _as_disk(domain)
    if disk is not None:
        return _disk_polygon_area(disk, polygon)
    if isinstance(domain, PupilDomain):
        outer = _disk_polygon_area(
            PropagationDisk(domain.outer_radius, domain.center), polygon
        )
        if domain.inner_radius == 0.0:
            return outer
        inner = _disk_polygon_area(
            PropagationDisk(domain.inner_radius, domain.center), polygon
        )
        return max(0.0, outer - inner)
    if isinstance(domain, PolygonKDomain):
        try:
            return intersect_convex_polygons(domain.polygon, polygon).area
        except ValueError:
            return 0.0
    if isinstance(domain, IntersectionDomain):
        clipped = polygon
        remaining: list[KDomain] = []
        for member in domain.domains:
            if isinstance(member, PolygonKDomain):
                try:
                    clipped = intersect_convex_polygons(clipped, member.polygon)
                except ValueError:
                    return 0.0
            else:
                remaining.append(member)
        if not remaining:
            return clipped.area
        # The public constructor currently permits at most one non-polygonal
        # primitive, making the remaining calculation exact.
        return intersection_area(remaining[0], clipped)
    raise TypeError(f"intersection area is not implemented for {type(domain).__name__}")


@dataclass(frozen=True, slots=True)
class IntersectionDomain:
    """Intersection of bounded domains.

    The initial implementation supports any number of convex polygons and at
    most one disk or annular pupil, which covers BZ/spectrum intersections.
    """

    domains: tuple[KDomain, ...]

    def __post_init__(self) -> None:
        domains = tuple(self.domains)
        if len(domains) < 2 or not all(isinstance(domain, KDomain) for domain in domains):
            raise ValueError("an intersection requires at least two KDomain values")
        non_polygons = sum(not isinstance(domain, PolygonKDomain) for domain in domains)
        if non_polygons > 1:
            raise ValueError("intersections currently support at most one non-polygon domain")
        object.__setattr__(self, "domains", domains)

    @property
    def area(self) -> float:
        polygons = [domain.polygon for domain in self.domains if isinstance(domain, PolygonKDomain)]
        if not polygons:
            return min(domain.area for domain in self.domains)
        clipped = polygons[0]
        for polygon in polygons[1:]:
            try:
                clipped = intersect_convex_polygons(clipped, polygon)
            except ValueError:
                return 0.0
        others = [domain for domain in self.domains if not isinstance(domain, PolygonKDomain)]
        return clipped.area if not others else intersection_area(others[0], clipped)

    def contains(self, points: ArrayLike) -> BoolArray:
        query = _points(points)
        masks = [domain.contains(query) for domain in self.domains]
        return np.logical_and.reduce(masks)

    def bounds(self) -> tuple[float, float, float, float]:
        bounds = [domain.bounds() for domain in self.domains]
        result = (
            max(item[0] for item in bounds),
            min(item[1] for item in bounds),
            max(item[2] for item in bounds),
            min(item[3] for item in bounds),
        )
        if result[0] > result[1] or result[2] > result[3]:
            raise ValueError("intersection domain is empty")
        return result


def brillouin_zone_domain(zone: object) -> PolygonKDomain:
    """Return a domain adapter for a Brillouin zone without importing it here."""

    cell = getattr(zone, "cell", None)
    polygon = getattr(cell, "domain", None)
    if not isinstance(polygon, PolygonDomain):
        raise TypeError("zone must expose a cell with a PolygonDomain")
    return PolygonKDomain(polygon)


def irreducible_zone_domain(zone: object) -> PolygonKDomain:
    polygon = getattr(zone, "irreducible_domain", None)
    if not isinstance(polygon, PolygonDomain):
        raise TypeError("zone must expose an irreducible PolygonDomain")
    return PolygonKDomain(polygon)


__all__ = [
    "EvanescentDisk",
    "IntersectionDomain",
    "KDomain",
    "PolygonKDomain",
    "PropagationDisk",
    "PupilDomain",
    "brillouin_zone_domain",
    "intersection_area",
    "irreducible_zone_domain",
]
