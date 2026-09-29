"""Immutable value objects used by the cell subsystem."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import numpy as np
from numpy.typing import ArrayLike, NDArray

from reciprocal.sampling import SamplingResult

FloatArray = NDArray[np.float64]


def _planar_array(value: ArrayLike, *, rows: int | None = None, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] not in (2, 3):
        raise ValueError(f"{name} must have shape (N, 2) or (N, 3)")
    if rows is not None and array.shape[0] != rows:
        raise ValueError(f"{name} must have {rows} rows")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if array.shape[1] == 2:
        array = np.column_stack((array, np.zeros(len(array))))
    elif np.any(array[:, 2] != 0.0):
        raise ValueError(f"{name} must lie in the x-y plane")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _signed_area(vertices: FloatArray) -> float:
    relative = vertices[:, :2] - vertices[0, :2]
    x = relative[:, 0]
    y = relative[:, 1]
    return float(0.5 * (x @ np.roll(y, -1) - y @ np.roll(x, -1)))


def _segments_intersect(a: FloatArray, b: FloatArray, c: FloatArray, d: FloatArray) -> bool:
    def orientation(p: FloatArray, q: FloatArray, r: FloatArray) -> float:
        first = q[:2] - p[:2]
        second = r[:2] - p[:2]
        return float(first[0] * second[1] - first[1] * second[0])

    o1, o2 = orientation(a, b, c), orientation(a, b, d)
    o3, o4 = orientation(c, d, a), orientation(c, d, b)
    scale = max(
        np.linalg.norm(b[:2] - a[:2]),
        np.linalg.norm(d[:2] - c[:2]),
        np.finfo(float).tiny,
    )
    tolerance = np.finfo(float).eps * scale * scale * 64
    return o1 * o2 < -tolerance and o3 * o4 < -tolerance


@dataclass(frozen=True, slots=True, eq=False)
class PolygonDomain:
    """A bounded, simple planar polygon with counter-clockwise vertices."""

    vertices: FloatArray

    def __post_init__(self) -> None:
        vertices = _planar_array(self.vertices, name="vertices")
        if len(vertices) < 3:
            raise ValueError("a polygon requires at least three vertices")
        coordinate_scale = max(float(np.max(np.abs(vertices))), np.finfo(float).tiny)
        vertex_tolerance = np.finfo(float).eps * coordinate_scale * 64
        if np.linalg.norm(vertices[0] - vertices[-1]) <= vertex_tolerance:
            vertices = vertices[:-1].copy()
            vertices.setflags(write=False)
        if len(vertices) < 3:
            raise ValueError("a polygon requires at least three distinct vertices")
        distances = np.linalg.norm(vertices[:, None, :2] - vertices[None, :, :2], axis=2)
        np.fill_diagonal(distances, np.inf)
        if np.any(distances <= vertex_tolerance):
            raise ValueError("polygon vertices must be distinct")
        signed_area = _signed_area(vertices)
        scale = max(
            float(np.ptp(vertices[:, 0])),
            float(np.ptp(vertices[:, 1])),
            np.finfo(float).tiny,
        )
        if abs(signed_area) <= np.finfo(float).eps * scale * scale * 64:
            raise ValueError("polygon area must be nonzero")
        for first in range(len(vertices)):
            for second in range(first + 2, len(vertices)):
                if first == 0 and second == len(vertices) - 1:
                    continue
                if _segments_intersect(
                    vertices[first],
                    vertices[(first + 1) % len(vertices)],
                    vertices[second],
                    vertices[(second + 1) % len(vertices)],
                ):
                    raise ValueError("polygon boundary must not self-intersect")
        if signed_area < 0:
            vertices = np.array(vertices[::-1], copy=True)
            vertices.setflags(write=False)
        object.__setattr__(self, "vertices", vertices)

    @property
    def area(self) -> float:
        return abs(_signed_area(self.vertices))

    @property
    def max_extent(self) -> float:
        return float(np.max(np.linalg.norm(self.vertices[:, :2], axis=1)))

    def contains(self, points: ArrayLike) -> NDArray[np.bool_]:
        from .geometry import contains_points

        return contains_points(self, points)

    def bounds(self) -> tuple[float, float, float, float]:
        vertices = self.vertices[:, :2]
        return (
            float(np.min(vertices[:, 0])),
            float(np.max(vertices[:, 0])),
            float(np.min(vertices[:, 1])),
            float(np.max(vertices[:, 1])),
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PolygonDomain) and np.array_equal(self.vertices, other.vertices)

    def __hash__(self) -> int:
        return hash((self.vertices.shape, self.vertices.tobytes()))


class UnitCellKind(Enum):
    PRIMITIVE = "primitive"
    CONVENTIONAL = "conventional"
    WIGNER_SEITZ = "wigner_seitz"


@dataclass(frozen=True, slots=True, eq=False)
class UnitCell:
    """Immutable lattice-cell data, independent of any parent lattice."""

    basis: FloatArray
    domain: PolygonDomain
    fractional_lattice_points: FloatArray
    kind: UnitCellKind

    def __post_init__(self) -> None:
        if not isinstance(self.kind, UnitCellKind):
            raise TypeError("kind must be a UnitCellKind")
        basis = _planar_array(self.basis, rows=2, name="basis")
        determinant = float(np.linalg.det(basis[:, :2]))
        scale = np.linalg.norm(basis[0, :2]) * np.linalg.norm(basis[1, :2])
        if scale == 0.0 or abs(determinant) <= np.finfo(float).eps * scale * 64:
            raise ValueError("basis translations must be independent and nonzero")
        points = np.asarray(self.fractional_lattice_points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
            raise ValueError("fractional_lattice_points must have shape (N, 2)")
        if not np.all(np.isfinite(points)):
            raise ValueError("fractional_lattice_points must be finite")
        canonical = np.mod(points, 1.0)
        canonical[np.isclose(canonical, 1.0)] = 0.0
        rounded = np.round(canonical, decimals=12)
        if len(np.unique(rounded, axis=0)) != len(points):
            raise ValueError("fractional lattice points must be unique modulo translations")
        if self.kind in (UnitCellKind.PRIMITIVE, UnitCellKind.WIGNER_SEITZ) and len(points) != 1:
            raise ValueError("primitive and Wigner-Seitz cells have multiplicity one")
        if self.kind in (UnitCellKind.PRIMITIVE, UnitCellKind.WIGNER_SEITZ) and not np.allclose(
            canonical[0], 0.0, rtol=0.0, atol=1e-12
        ):
            raise ValueError("primitive and Wigner-Seitz lattice point must be the origin")
        if not np.isclose(self.domain.area, abs(determinant), rtol=1e-9, atol=0.0):
            raise ValueError("cell domain area must equal the basis parallelogram area")
        points = np.array(canonical, copy=True)
        points.setflags(write=False)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "fractional_lattice_points", points)

    @property
    def area(self) -> float:
        return self.domain.area

    @property
    def vertices(self) -> FloatArray:
        return self.domain.vertices

    @property
    def max_extent(self) -> float:
        return self.domain.max_extent

    @property
    def multiplicity(self) -> int:
        return len(self.fractional_lattice_points)

    @property
    def primitive_area(self) -> float:
        return self.area / self.multiplicity

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, UnitCell)
            and self.kind is other.kind
            and self.domain == other.domain
            and np.array_equal(self.basis, other.basis)
            and np.array_equal(self.fractional_lattice_points, other.fractional_lattice_points)
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.kind,
                self.domain,
                self.basis.tobytes(),
                self.fractional_lattice_points.tobytes(),
            )
        )
