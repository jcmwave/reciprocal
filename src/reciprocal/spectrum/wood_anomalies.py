"""Diffraction-threshold circles and one-dimensional arc quadrature."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike

from reciprocal.lattice import Lattice
from reciprocal.sampling import SamplingConstraint, _readonly, resolve_point_counts

from .domains import PropagationDisk


@dataclass(frozen=True, slots=True)
class CircleArc:
    center: tuple[float, float]
    radius: float
    theta_start: float
    theta_end: float

    def __post_init__(self) -> None:
        center = tuple(float(value) for value in self.center)
        radius, start, end = float(self.radius), float(self.theta_start), float(self.theta_end)
        if (
            len(center) != 2
            or not np.all(np.isfinite((*center, radius, start, end)))
            or radius <= 0
            or not 0 < end - start <= 2 * np.pi + 1e-12
        ):
            raise ValueError(
                "arc requires finite geometry, positive radius and angular length in (0, 2*pi]"
            )
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "theta_start", start)
        object.__setattr__(self, "theta_end", end)

    @property
    def length(self) -> float:
        return self.radius * (self.theta_end - self.theta_start)

    @property
    def closed(self) -> bool:
        return bool(np.isclose(self.theta_end - self.theta_start, 2 * np.pi, rtol=0, atol=1e-12))

    def points_at(self, angles: ArrayLike) -> np.ndarray:
        angles = np.atleast_1d(np.asarray(angles, dtype=float))
        return np.asarray(self.center) + self.radius * np.column_stack(
            (np.cos(angles), np.sin(angles))
        )


@dataclass(frozen=True, slots=True)
class WoodAnomalyCircle:
    center: tuple[float, float]
    radius: float
    reciprocal_order: tuple[int, int]
    shell_index: int
    arc: CircleArc | None
    tangent_point: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        center = tuple(float(value) for value in self.center)
        order = tuple(self.reciprocal_order)
        if (
            len(center) != 2
            or not np.all(np.isfinite(center))
            or not np.isfinite(self.radius)
            or self.radius <= 0
        ):
            raise ValueError("circle requires a finite center and positive radius")
        if len(order) != 2 or any(
            isinstance(value, bool) or int(value) != value for value in order
        ):
            raise ValueError("reciprocal_order must contain two integers")
        if (
            isinstance(self.shell_index, bool)
            or int(self.shell_index) != self.shell_index
            or self.shell_index <= 0
        ):
            raise ValueError("shell index must be a positive integer")
        if (self.arc is None) == (self.tangent_point is None):
            raise ValueError(
                "circle must have either a positive-length visible arc or a tangent contact"
            )
        if self.arc is not None and (
            self.arc.center != center or not np.isclose(self.arc.radius, self.radius)
        ):
            raise ValueError("visible arc must belong to its circle")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "radius", float(self.radius))
        object.__setattr__(self, "reciprocal_order", tuple(int(value) for value in order))
        object.__setattr__(self, "shell_index", int(self.shell_index))
        if self.tangent_point is not None:
            contact = tuple(float(value) for value in self.tangent_point)
            if len(contact) != 2 or not np.all(np.isfinite(contact)):
                raise ValueError("tangent point must be finite")
            object.__setattr__(self, "tangent_point", contact)


def circle_disk_intersection(center: ArrayLike, radius: float, domain: PropagationDisk):
    """Return the visible edge arc and/or tangent contact, not filled-disk overlap."""
    center = np.asarray(center, dtype=float)
    radius = float(radius)
    if (
        center.shape != (2,)
        or not np.all(np.isfinite(center))
        or not np.isfinite(radius)
        or radius <= 0
    ):
        raise ValueError("circle geometry must be finite with positive radius")
    delta = center - np.asarray(domain.center)
    distance = float(np.linalg.norm(delta))
    if distance == 0:
        if radius <= domain.radius:
            return CircleArc(tuple(center), radius, 0.0, 2 * np.pi), None
        return None, None
    # Normalize lengths before squaring, including for microscopic/large units.
    scale = max(distance, radius, domain.radius)
    d, r, bound = distance / scale, radius / scale, domain.radius / scale
    cosine = (d * d + r * r - bound * bound) / (2 * d * r)
    tolerance = np.finfo(float).eps * 128
    if cosine > 1 + tolerance:
        return None, None
    direction = float(np.arctan2(-delta[1], -delta[0]))
    if cosine >= 1 - tolerance:
        contact = center + radius * np.array([np.cos(direction), np.sin(direction)])
        return None, tuple(contact)
    if cosine <= -1:
        return CircleArc(tuple(center), radius, 0.0, 2 * np.pi), None
    half_width = float(np.arccos(np.clip(cosine, -1, 1)))
    return CircleArc(tuple(center), radius, direction - half_width, direction + half_width), None


@dataclass(frozen=True, slots=True, eq=False)
class CurveSamplingResult:
    """Arc-length quadrature, distinct from two-dimensional area sampling.

    Tangent contacts have zero length and are reported separately. An empty
    result has an empty normalized-weight array and integrates to zero.
    """

    points: np.ndarray
    physical_weights: np.ndarray
    circle_indices: np.ndarray
    segment_boundaries: tuple[tuple[int, int], ...]
    circles: tuple[WoodAnomalyCircle, ...]
    tangent_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    tangent_circle_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    point_ids: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float)
        weights = np.asarray(self.physical_weights, dtype=float)
        indices = np.asarray(self.circle_indices)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or weights.shape != (len(points),)
            or not np.all(np.isfinite(points))
            or not np.all(np.isfinite(weights))
            or np.any(weights <= 0)
        ):
            raise ValueError("curve points and positive arc-length weights must align")
        if (
            indices.shape != weights.shape
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            or np.any(indices >= len(self.circles))
        ):
            raise ValueError("circle indices must identify every curve point")
        segments = tuple(tuple(segment) for segment in self.segment_boundaries)
        stop = 0
        for start, end in segments:
            if start != stop or not start < end <= len(points):
                raise ValueError("segments must partition curve points in order")
            stop = end
        if stop != len(points):
            raise ValueError("segments must cover all curve points")
        contacts = np.asarray(self.tangent_points, dtype=float)
        contact_indices = np.asarray(self.tangent_circle_indices)
        if (
            contacts.shape != (len(contact_indices), 2)
            or not np.all(np.isfinite(contacts))
            or not np.issubdtype(contact_indices.dtype, np.integer)
            or np.any(contact_indices < 0)
            or np.any(contact_indices >= len(self.circles))
        ):
            raise ValueError("tangent contacts and circle indices must align")
        for name, value, dtype in [
            ("points", points, float),
            ("physical_weights", weights, float),
            ("circle_indices", indices, np.int64),
            ("tangent_points", contacts, float),
            ("tangent_circle_indices", contact_indices, np.int64),
        ]:
            object.__setattr__(self, name, _readonly(value, dtype))
        object.__setattr__(self, "circles", tuple(self.circles))
        object.__setattr__(self, "segment_boundaries", segments)
        object.__setattr__(self, "point_ids", _readonly(np.arange(len(points)), np.int64))

    @property
    def integration_element(self) -> float:
        return float(np.sum(self.physical_weights))

    @property
    def normalized_weights(self) -> np.ndarray:
        return _readonly(
            self.physical_weights / self.integration_element if len(self.points) else np.empty(0)
        )

    @property
    def weights(self) -> np.ndarray:
        return self.normalized_weights

    @property
    def reciprocal_orders(self) -> np.ndarray:
        return _readonly(
            np.asarray(
                [self.circles[index].reciprocal_order for index in self.circle_indices],
                dtype=np.int64,
            ).reshape(-1, 2)
        )

    def integrate(self, values: ArrayLike):
        values = np.asarray(values)
        if values.ndim == 0 or values.shape[0] != len(self.points):
            raise ValueError("values must align with curve points")
        return np.tensordot(self.physical_weights, values, axes=(0, 0))

    def average(self, values: ArrayLike):
        if self.integration_element == 0:
            raise ValueError("a zero-length curve has no average")
        return self.integrate(values) / self.integration_element


@dataclass(frozen=True, slots=True)
class WoodAnomalySet:
    circles: tuple[WoodAnomalyCircle, ...]
    propagating_domain: PropagationDisk

    def __post_init__(self) -> None:
        if not isinstance(self.propagating_domain, PropagationDisk):
            raise TypeError("propagating_domain must be a PropagationDisk")
        circles = tuple(self.circles)
        if not all(isinstance(circle, WoodAnomalyCircle) for circle in circles):
            raise TypeError("circles must be WoodAnomalyCircle objects")
        object.__setattr__(self, "circles", circles)

    @property
    def circles_by_shell(self):
        return MappingProxyType(
            {
                shell: tuple(circle for circle in self.circles if circle.shell_index == shell)
                for shell in sorted({circle.shell_index for circle in self.circles})
            }
        )

    def sample_edges(self, constraint: SamplingConstraint = None) -> CurveSamplingResult:
        points, weights, indices, boundaries, contacts, contact_indices = [], [], [], [], [], []
        start = 0
        for index, circle in enumerate(self.circles):
            arc = circle.arc
            if arc is None:
                contacts.append(circle.tangent_point)
                contact_indices.append(index)
                continue
            count = resolve_point_counts(
                arc.length,
                constraint,
                placement="closed" if arc.closed else "endpoints",
                default=64,
            )[0]
            if count == 1:
                angles = np.array([(arc.theta_start + arc.theta_end) / 2])
                local_weights = np.array([arc.length])
            else:
                angles = np.linspace(arc.theta_start, arc.theta_end, count, endpoint=not arc.closed)
                local_weights = np.full(count, arc.length / (count if arc.closed else count - 1))
                if not arc.closed:
                    local_weights[[0, -1]] *= 0.5
            points.extend(arc.points_at(angles))
            weights.extend(local_weights)
            indices.extend([index] * count)
            boundaries.append((start, start + count))
            start += count
        return CurveSamplingResult(
            np.asarray(points).reshape(-1, 2),
            np.asarray(weights),
            np.asarray(indices, dtype=np.int64),
            tuple(boundaries),
            self.circles,
            np.asarray(contacts).reshape(-1, 2),
            np.asarray(contact_indices, dtype=np.int64),
        )


def make_wood_anomalies(
    reciprocal_lattice: Lattice, wavevector_magnitude: float, *, shells=(1, 2, 3, 4)
) -> WoodAnomalySet:
    """Circles for k_diffracted = k_incident + G: centers are -G."""
    if (
        not isinstance(reciprocal_lattice, Lattice)
        or reciprocal_lattice.lattice_type != "reciprocal"
    ):
        raise ValueError("Wood anomalies require a reciprocal lattice")
    selected = tuple(shells)
    if any(
        isinstance(value, bool) or not np.isfinite(value) or int(value) != value or value <= 0
        for value in selected
    ):
        raise ValueError("shell indices must be positive integers")
    if len(set(selected)) != len(selected):
        raise ValueError("shell indices must be unique")
    domain = PropagationDisk(wavevector_magnitude)
    groups, _ = reciprocal_lattice.translation_shells(max(selected, default=0))
    circles = []
    for shell in sorted(selected):
        for order in groups[int(shell) - 1]:
            center = -order @ reciprocal_lattice.vectors.basis[:, :2]
            arc, contact = circle_disk_intersection(center, wavevector_magnitude, domain)
            if arc is not None or contact is not None:
                circles.append(
                    WoodAnomalyCircle(
                        tuple(center), wavevector_magnitude, tuple(order), int(shell), arc, contact
                    )
                )
    return WoodAnomalySet(tuple(circles), domain)


__all__ = [
    "CircleArc",
    "CurveSamplingResult",
    "WoodAnomalyCircle",
    "WoodAnomalySet",
    "circle_disk_intersection",
    "make_wood_anomalies",
]
