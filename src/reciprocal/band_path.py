"""Conventional and custom paths through reciprocal-space special points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from reciprocal.bravais import BravaisLattice
from reciprocal.brillouin_zone import BrillouinZone

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

DEFAULT_PATHS: dict[BravaisLattice, tuple[str, ...]] = {
    BravaisLattice.SQUARE: ("Γ", "X", "M", "Γ"),
    BravaisLattice.RECTANGULAR: ("Γ", "X", "S", "Y", "Γ"),
    BravaisLattice.CENTERED_RECTANGULAR: ("Γ", "X", "S", "Y", "Γ"),
    BravaisLattice.HEXAGONAL: ("Γ", "M", "K", "Γ"),
    BravaisLattice.OBLIQUE: ("Γ", "X", "C", "Y", "Γ"),
}


def _readonly(value: object, dtype: object) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, eq=False)
class HighSymmetryPath:
    """An ordered reciprocal-space polyline with band-axis metadata."""

    points: FloatArray
    fractional_points: FloatArray
    distance: FloatArray
    node_indices: IntArray
    labels: tuple[str, ...]
    segments: tuple[tuple[str, str], ...]
    reciprocal_basis: FloatArray
    break_indices: IntArray

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float)
        fractional = np.asarray(self.fractional_points, dtype=float)
        distance = np.asarray(self.distance, dtype=float)
        nodes = np.asarray(self.node_indices)
        basis = np.asarray(self.reciprocal_basis, dtype=float)
        breaks = np.asarray(self.break_indices)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("path points must have shape (N, 2), N >= 2")
        if fractional.shape != points.shape or not np.all(np.isfinite(fractional)):
            raise ValueError("fractional_points must match points and be finite")
        if distance.shape != (len(points),) or not np.all(np.isfinite(distance)):
            raise ValueError("distance must contain one finite value per point")
        if distance[0] != 0.0 or np.any(np.diff(distance) < 0.0):
            raise ValueError("distance must start at zero and be non-decreasing")
        if nodes.shape != (len(self.labels),) or not np.issubdtype(nodes.dtype, np.integer):
            raise ValueError("node_indices must contain one integer per label")
        if np.any(nodes < 0) or np.any(nodes >= len(points)):
            raise ValueError("node_indices contains an invalid point index")
        if basis.shape not in ((2, 2), (2, 3)) or not np.all(np.isfinite(basis)):
            raise ValueError("reciprocal_basis must have shape (2, 2) or (2, 3)")
        basis2 = basis[:, :2]
        if abs(np.linalg.det(basis2)) == 0.0:
            raise ValueError("reciprocal_basis must be independent")
        if not np.allclose(fractional @ basis2, points, rtol=1e-10, atol=1e-12):
            raise ValueError("fractional and Cartesian path coordinates disagree")
        if breaks.ndim != 1 or not np.issubdtype(breaks.dtype, np.integer):
            raise ValueError("break_indices must be a one-dimensional integer array")
        if np.any(breaks <= 0) or np.any(breaks >= len(points)):
            raise ValueError("break_indices contains an invalid point index")
        if len(self.segments) != len(self.labels) - 1 - len(breaks):
            raise ValueError("segments, labels, and path breaks are inconsistent")
        object.__setattr__(self, "points", _readonly(points, float))
        object.__setattr__(self, "fractional_points", _readonly(fractional, float))
        object.__setattr__(self, "distance", _readonly(distance, float))
        object.__setattr__(self, "node_indices", _readonly(nodes, np.int64))
        object.__setattr__(self, "labels", tuple(self.labels))
        object.__setattr__(self, "segments", tuple(tuple(item) for item in self.segments))
        object.__setattr__(self, "reciprocal_basis", _readonly(basis, float))
        object.__setattr__(self, "break_indices", _readonly(breaks, np.int64))

    @property
    def tick_positions(self) -> FloatArray:
        result = self.distance[self.node_indices]
        result.setflags(write=False)
        return result

    @property
    def tick_labels(self) -> tuple[str, ...]:
        return self.labels

    def to_kvectors(self, wavelength: float, refractive_index: float, direction: int = 1):
        from reciprocal.kvector import KVectorGroup

        return KVectorGroup.from_transverse(
            wavelength,
            n=np.full(len(self.points), refractive_index),
            kx=self.points[:, 0],
            ky=self.points[:, 1],
            normal=np.full(len(self.points), direction),
        )


def _branches(
    zone: BrillouinZone,
    labels: Sequence[str] | Sequence[Sequence[str]] | None,
) -> tuple[tuple[str, ...], ...]:
    if labels is None:
        return (DEFAULT_PATHS[zone.bravais],)
    values = tuple(labels)
    if not values:
        raise ValueError("a path requires at least one branch")
    if all(isinstance(item, str) for item in values):
        branches = (tuple(values),)
    elif all(not isinstance(item, str) for item in values):
        branches = tuple(tuple(branch) for branch in values)  # type: ignore[arg-type]
    else:
        raise TypeError("labels must be one label sequence or a sequence of branches")
    if any(len(branch) < 2 for branch in branches):
        raise ValueError("every path branch requires at least two labels")
    for branch in branches:
        for label in branch:
            if not isinstance(label, str) or label not in zone.special_points:
                raise ValueError(f"unknown special-point label: {label!r}")
    return branches


def make_high_symmetry_path(
    zone: BrillouinZone,
    labels: Sequence[str] | Sequence[Sequence[str]] | None = None,
    *,
    points_per_segment: int | Sequence[int] | None = None,
    max_spacing: float | None = None,
) -> HighSymmetryPath:
    """Construct a conventional or custom piecewise-linear k path."""

    if not isinstance(zone, BrillouinZone):
        raise TypeError("zone must be a BrillouinZone")
    if points_per_segment is not None and max_spacing is not None:
        raise ValueError("specify either points_per_segment or max_spacing, not both")
    branches = _branches(zone, labels)
    segment_total = sum(len(branch) - 1 for branch in branches)
    if max_spacing is not None:
        spacing = float(max_spacing)
        if not np.isfinite(spacing) or spacing <= 0.0:
            raise ValueError("max_spacing must be finite and positive")
        counts = None
    elif points_per_segment is None:
        counts = [51] * segment_total
    elif isinstance(points_per_segment, (int, np.integer)) and not isinstance(
        points_per_segment, bool
    ):
        counts = [int(points_per_segment)] * segment_total
    else:
        counts = [int(item) for item in points_per_segment]  # type: ignore[union-attr]
        if len(counts) != segment_total:
            raise ValueError("points_per_segment must contain one count per segment")
    if counts is not None and any(count < 2 for count in counts):
        raise ValueError("every segment requires at least two points")

    all_points: list[np.ndarray] = []
    all_distances: list[float] = []
    node_indices: list[int] = []
    flat_labels: list[str] = []
    segments: list[tuple[str, str]] = []
    break_indices: list[int] = []
    count_index = 0
    cumulative = 0.0
    for branch_index, branch in enumerate(branches):
        if branch_index:
            break_indices.append(len(all_points))
        for label_index, label in enumerate(branch):
            start = zone.special_points[label].cartesian[:2]
            flat_labels.append(label)
            if label_index == 0:
                all_points.append(np.array(start, copy=True))
                all_distances.append(cumulative)
                node_indices.append(len(all_points) - 1)
                continue
            previous_label = branch[label_index - 1]
            end = start
            start = zone.special_points[previous_label].cartesian[:2]
            length = float(np.linalg.norm(end - start))
            count = (
                int(np.ceil(length / spacing)) + 1
                if max_spacing is not None
                else counts[count_index]  # type: ignore[index]
            )
            count_index += 1
            fractions = np.linspace(0.0, 1.0, count)[1:]
            for fraction in fractions:
                all_points.append(start + fraction * (end - start))
                all_distances.append(cumulative + fraction * length)
            cumulative += length
            node_indices.append(len(all_points) - 1)
            segments.append((previous_label, label))

    points = np.vstack(all_points)
    basis = zone.cell.basis
    fractional = points @ np.linalg.inv(basis[:, :2])
    return HighSymmetryPath(
        points,
        fractional,
        np.asarray(all_distances),
        np.asarray(node_indices),
        tuple(flat_labels),
        tuple(segments),
        basis,
        np.asarray(break_indices, dtype=np.int64),
    )


def set_band_path_axis(ax: object, path: HighSymmetryPath) -> None:
    """Apply k-path ticks, limits, and segment separators to an axes-like object."""

    positions = path.tick_positions
    ax.set_xlim(float(path.distance[0]), float(path.distance[-1]))  # type: ignore[attr-defined]
    ax.set_xticks(positions, path.tick_labels)  # type: ignore[attr-defined]
    for position in np.unique(positions[1:-1]):
        ax.axvline(float(position), color="0.8", linewidth=0.8, zorder=0)  # type: ignore[attr-defined]


__all__ = [
    "DEFAULT_PATHS",
    "HighSymmetryPath",
    "make_high_symmetry_path",
    "set_band_path_axis",
]
