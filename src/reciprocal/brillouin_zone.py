"""Reciprocal-space aggregates built from immutable cell and symmetry data."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reciprocal.bravais import BravaisLattice
from reciprocal.cells.construction import make_wigner_seitz_cell
from reciprocal.cells.geometry import contains_points
from reciprocal.cells.model import (
    FloatArray,
    PolygonDomain,
    SamplingResult,
    UnitCell,
    UnitCellKind,
)
from reciprocal.cells.sampling import CellSampler, SamplingConstraint
from reciprocal.numerics import DEFAULT_TOLERANCES, Tolerances
from reciprocal.symmetry import (
    PointGroup,
    PointOperation,
    equivalent_mod_lattice,
    little_group,
    point_group,
    point_orbit,
)


def _readonly_vector(value: ArrayLike, length: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector of length {length}")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, eq=False)
class SpecialKPoint:
    label: str
    fractional: FloatArray
    cartesian: FloatArray
    little_group: tuple[PointOperation, ...]
    orbit: tuple[FloatArray, ...]

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("a special k-point requires a label")
        if not self.little_group or not self.orbit:
            raise ValueError("a special k-point requires a little group and orbit")
        orbit = tuple(_readonly_vector(member, 3, "orbit member") for member in self.orbit)
        object.__setattr__(self, "fractional", _readonly_vector(self.fractional, 2, "fractional"))
        object.__setattr__(self, "cartesian", _readonly_vector(self.cartesian, 3, "cartesian"))
        object.__setattr__(self, "little_group", tuple(self.little_group))
        object.__setattr__(self, "orbit", orbit)


@dataclass(frozen=True, slots=True, eq=False)
class BrillouinZone:
    cell: UnitCell
    irreducible_domain: PolygonDomain
    special_points: Mapping[str, SpecialKPoint]
    point_group: tuple[PointOperation, ...]
    bravais: BravaisLattice

    def __post_init__(self) -> None:
        if self.cell.kind is not UnitCellKind.WIGNER_SEITZ:
            raise ValueError("a Brillouin zone cell must be a Wigner-Seitz cell")
        if not self.point_group:
            raise ValueError("a Brillouin zone requires a nonempty point group")
        if not np.all(contains_points(self.cell.domain, self.irreducible_domain.vertices)):
            raise ValueError("irreducible domain must lie inside the Brillouin-zone cell")
        if not np.isclose(
            self.irreducible_domain.area * len(self.point_group),
            self.cell.area,
            rtol=1e-8,
            atol=0.0,
        ):
            raise ValueError("irreducible-domain area must equal full area divided by group order")
        for label, point in self.special_points.items():
            if label != point.label:
                raise ValueError("special-point mapping keys must equal their labels")
            if not contains_points(self.cell.domain, [point.cartesian])[0]:
                raise ValueError("special points must lie inside the Brillouin-zone cell")
        object.__setattr__(self, "special_points", MappingProxyType(dict(self.special_points)))
        object.__setattr__(self, "point_group", tuple(self.point_group))
        if not isinstance(self.bravais, BravaisLattice):
            raise TypeError("bravais must be a BravaisLattice")

    @property
    def vertices(self) -> FloatArray:
        """Compatibility convenience exposing the Wigner-Seitz vertices."""
        return self.cell.vertices

    @property
    def area(self) -> float:
        return self.cell.area

    @property
    def max_extent(self) -> float:
        return self.cell.max_extent


class ReciprocalLattice(Protocol):
    lattice_type: str
    bravais: BravaisLattice
    vectors: object


def _clip_origin_half_plane(vertices: FloatArray, normal: FloatArray) -> FloatArray:
    """Clip a polygon to ``normal dot x >= 0``."""
    output: list[FloatArray] = []
    tolerance = (
        np.finfo(float).eps * max(float(np.max(np.abs(vertices))), np.finfo(float).tiny) * 256
    )
    for index, start in enumerate(vertices):
        end = vertices[(index + 1) % len(vertices)]
        first, second = float(np.dot(normal, start[:2])), float(np.dot(normal, end[:2]))
        first_inside, second_inside = first >= -tolerance, second >= -tolerance
        if first_inside != second_inside:
            fraction = first / (first - second)
            output.append(start + fraction * (end - start))
        if second_inside:
            output.append(end)
    if len(output) < 3:
        raise RuntimeError("point-group chamber construction produced an empty polygon")
    array = np.asarray(output)
    keep = np.linalg.norm(array - np.roll(array, 1, axis=0), axis=1) > tolerance
    return array[keep]


def _irreducible_domain(cell: UnitCell, operations: tuple[PointOperation, ...]) -> PolygonDomain:
    # A generic direction has a trivial stabilizer. Its Dirichlet chamber under
    # the finite point group occupies exactly one group-order fraction.
    basis = cell.basis[:, :2].T
    direction = basis @ np.array([1.0, np.sqrt(2.0) / 5.0])
    vertices = cell.vertices.copy()
    identity = np.eye(2)
    for operation in operations:
        if np.array_equal(operation.fractional, identity):
            continue
        transformed = operation.cartesian[:2, :2] @ direction
        normal = direction - transformed
        if np.linalg.norm(normal) > np.finfo(float).eps * np.linalg.norm(direction):
            vertices = _clip_origin_half_plane(vertices, normal)
    return PolygonDomain(vertices)


def _label_coordinates(bravais: BravaisLattice, basis: FloatArray) -> dict[str, FloatArray]:
    if bravais is BravaisLattice.HEXAGONAL:
        # For equal vectors, the conventional corner changes fractional form
        # between the 60- and 120-degree primitive representations.
        k = (
            np.array([1.0 / 3, 1.0 / 3])
            if np.dot(basis[0], basis[1]) > 0
            else np.array([2.0 / 3, 1.0 / 3])
        )
        return {"Γ": np.zeros(2), "M": np.array([0.5, 0.0]), "K": k}
    if bravais is BravaisLattice.SQUARE:
        return {"Γ": np.zeros(2), "X": np.array([0.5, 0.0]), "M": np.array([0.5, 0.5])}
    if bravais in (BravaisLattice.RECTANGULAR, BravaisLattice.CENTERED_RECTANGULAR):
        return {
            "Γ": np.zeros(2),
            "X": np.array([0.5, 0.0]),
            "Y": np.array([0.0, 0.5]),
            "S": np.array([0.5, 0.5]),
        }
    return {
        "Γ": np.zeros(2),
        "X": np.array([0.5, 0.0]),
        "Y": np.array([0.0, 0.5]),
        "C": np.array([0.5, 0.5]),
    }


def _representative_in_cell(fractional: FloatArray, cell: UnitCell) -> FloatArray:
    """Choose a deterministic lattice-equivalent representative in the WS cell."""
    if contains_points(cell.domain, [fractional @ cell.basis[:, :2]])[0]:
        return np.array(fractional, copy=True)
    candidates = []
    for first_shift in range(-2, 3):
        for second_shift in range(-2, 3):
            candidate = fractional + np.array([first_shift, second_shift])
            cartesian = candidate @ cell.basis[:, :2]
            if contains_points(cell.domain, [cartesian])[0]:
                candidates.append(candidate)
    if not candidates:
        raise RuntimeError("could not place a special point in the Brillouin zone")
    candidates.sort(key=lambda item: (np.linalg.norm(item @ cell.basis[:, :2]), item[0], item[1]))
    return candidates[0]


def _special_point(
    label: str,
    fractional: FloatArray,
    cell: UnitCell,
    operations: tuple[PointOperation, ...],
    tolerance: float,
) -> SpecialKPoint:
    representative = _representative_in_cell(fractional, cell)
    cartesian2 = representative @ cell.basis[:, :2]
    cartesian = np.array([cartesian2[0], cartesian2[1], 0.0])
    stabilizer = little_group(cartesian, cell.basis, operations, tolerance=tolerance)
    orbit = point_orbit(cartesian, cell.basis, operations, tolerance=tolerance)
    if len(orbit) * len(stabilizer) != len(operations):
        raise RuntimeError(f"orbit-stabilizer identity failed for {label}")
    return SpecialKPoint(label, representative, cartesian, stabilizer, orbit)


def make_brillouin_zone(
    lattice: ReciprocalLattice, *, tolerances: Tolerances = DEFAULT_TOLERANCES
) -> BrillouinZone:
    if not isinstance(tolerances, Tolerances):
        raise TypeError("tolerances must be a Tolerances instance")
    if lattice.lattice_type != "reciprocal":
        raise ValueError("a Brillouin zone requires a reciprocal lattice")
    cell = make_wigner_seitz_cell(lattice.vectors, tolerances=tolerances)  # type: ignore[arg-type]
    group: PointGroup = point_group(
        cell.basis, lattice.bravais, tolerance=tolerances.relative
    )
    operations = group.operations
    points: dict[str, SpecialKPoint] = {}
    for label, fractional in _label_coordinates(lattice.bravais, cell.basis).items():
        points[label] = _special_point(label, fractional, cell, operations, tolerances.relative)
    # Low-symmetry Wigner-Seitz vertices are path-defining points even when
    # their little group is trivial. Add one deterministic label per orbit.
    if lattice.bravais in (BravaisLattice.OBLIQUE, BravaisLattice.CENTERED_RECTANGULAR):
        vertex_fractional = cell.vertices[:, :2] @ np.linalg.inv(cell.basis[:, :2])
        vertex_fractional = vertex_fractional[
            np.lexsort((vertex_fractional[:, 1], vertex_fractional[:, 0]))
        ]
        vertex_points: list[SpecialKPoint] = []
        for fractional in vertex_fractional:
            candidate = _special_point(
                "candidate", fractional, cell, operations, tolerances.relative
            )
            if any(
                equivalent_mod_lattice(
                    candidate.cartesian,
                    orbit_member,
                    cell.basis,
                    tolerance=tolerances.relative,
                )
                for existing in vertex_points
                for orbit_member in existing.orbit
            ):
                continue
            label = f"H{len(vertex_points) + 1}"
            point = _special_point(label, fractional, cell, operations, tolerances.relative)
            vertex_points.append(point)
            points[label] = point
    irreducible = _irreducible_domain(cell, operations)
    return BrillouinZone(cell, irreducible, points, operations, lattice.bravais)


class BrillouinZoneSampler:
    """Sample a full Brillouin zone or one irreducible point-group chamber."""

    def __init__(self, cell_sampler: CellSampler | None = None) -> None:
        self.cell_sampler = CellSampler() if cell_sampler is None else cell_sampler

    def sample_full(
        self,
        zone: BrillouinZone,
        constraint: SamplingConstraint | None = None,
        center: NDArray[np.float64] | None = None,
    ) -> SamplingResult:
        sample = self.cell_sampler.sample(zone.cell, constraint, center)
        return SamplingResult(
            sample.points,
            sample.weights,
            sample.integration_element,
            sample.domain,
            metadata={"symmetry_operations": zone.point_group},
        )

    def sample_irreducible(
        self,
        zone: BrillouinZone,
        constraint: SamplingConstraint | None = None,
        center: NDArray[np.float64] | None = None,
    ) -> SamplingResult:
        full = self.cell_sampler.sample(zone.cell, constraint, center)
        mask = contains_points(zone.irreducible_domain, full.points)
        points = full.points[mask]
        if len(points) == 0:
            raise ValueError("sampling constraint produced no points in the irreducible domain")
        multiplicities = np.asarray(
            [len(point_orbit(point, zone.cell.basis, zone.point_group)) for point in points],
            dtype=float,
        )
        raw_weights = full.weights[mask] * multiplicities
        weights = raw_weights / np.sum(raw_weights)
        return SamplingResult(
            points,
            weights,
            zone.area,
            zone.cell.domain,
            metadata={
                "coordinate_domain": zone.irreducible_domain,
                "symmetry_operations": zone.point_group,
                "source_region": "ibz",
            },
        )


__all__ = [
    "BrillouinZone",
    "BrillouinZoneSampler",
    "SpecialKPoint",
    "make_brillouin_zone",
]
