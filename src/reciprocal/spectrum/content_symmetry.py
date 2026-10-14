"""Finite symmetries of non-periodic contents, independent of a lattice."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree

from reciprocal.cells.model import PolygonDomain
from reciprocal.integration_cells import OrbitIntegrationCell
from reciprocal.sampling import SamplingResult, _readonly

from .domains import (
    EvanescentDisk,
    IntersectionDomain,
    PolygonKDomain,
    PropagationDisk,
    PupilDomain,
    intersection_area,
)
from .model import CartesianGrid, CircularGrid, FieldSamples, PolarGrid


@dataclass(frozen=True, slots=True, eq=False)
class CartesianPointOperation:
    """An orthogonal planar operation with no fractional lattice coordinates."""

    cartesian: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.cartesian, dtype=float)
        if matrix.shape == (2, 2):
            extended = np.eye(3)
            extended[:2, :2] = matrix
            matrix = extended
        if (
            matrix.shape != (3, 3)
            or not np.all(np.isfinite(matrix))
            or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
        ):
            raise ValueError("operation must be a finite orthogonal Cartesian matrix")
        if not np.allclose(matrix[2], [0, 0, 1], atol=1e-12):
            raise ValueError("content symmetry must preserve the transverse plane and z direction")
        matrix = np.where(np.abs(matrix) < 1e-14, 0.0, matrix)
        object.__setattr__(
            self, "cartesian", np.frombuffer(matrix.tobytes(), dtype=float).reshape(3, 3)
        )

    @property
    def determinant(self) -> int:
        return int(round(np.linalg.det(self.cartesian)))

    def apply(self, points: ArrayLike) -> np.ndarray:
        array = np.asarray(points, dtype=float)
        if (
            array.ndim not in (1, 2)
            or array.shape[-1] not in (2, 3)
            or not np.all(np.isfinite(array))
        ):
            raise ValueError("points must be finite planar coordinates")
        if array.shape[-1] == 2:
            array = np.concatenate((array, np.zeros(array.shape[:-1] + (1,))), axis=-1)
        return array @ self.cartesian.T


def _rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


@dataclass(frozen=True, slots=True)
class SymmetryCone:
    """Counterclockwise angular interval in degrees; endpoints may cross 360."""

    start_degrees: float
    end_degrees: float

    def __post_init__(self) -> None:
        start, end = float(self.start_degrees), float(self.end_degrees)
        if not np.all(np.isfinite([start, end])) or not 0 < end - start <= 360:
            raise ValueError("cone must have a finite counterclockwise width in (0, 360]")
        object.__setattr__(self, "start_degrees", start)
        object.__setattr__(self, "end_degrees", end)

    def contains(self, points: ArrayLike) -> np.ndarray:
        points = np.atleast_2d(np.asarray(points, dtype=float))[:, :2]
        angles = (np.degrees(np.arctan2(points[:, 1], points[:, 0])) - self.start_degrees) % 360
        scale = max(float(np.max(np.linalg.norm(points, axis=1), initial=0)), np.finfo(float).tiny)
        return (
            (angles <= self.end_degrees - self.start_degrees + 1e-8)
            | (angles >= 360 - 1e-8)
            | (np.linalg.norm(points, axis=1) <= scale * 1e-12)
        )


@dataclass(frozen=True, slots=True)
class ContentSymmetry:
    """Declared symmetry of complete contents/materials and covariant responses.

    Cn and Dn name finite rotation and reflection groups. Orientation sets
    the first reflection axis and the start of the canonical incidence cone.
    """

    name: str
    orientation_degrees: float = 0.0
    operations: tuple[CartesianPointOperation, ...] = field(init=False)
    rotations: int = field(init=False)

    def __post_init__(self) -> None:
        match = re.fullmatch(r"([CD])([1-9][0-9]*)", self.name.upper())
        if match is None:
            raise ValueError("content symmetry must be a finite Cn or Dn group")
        angle = float(self.orientation_degrees)
        if not np.isfinite(angle):
            raise ValueError("orientation must be finite")
        angle %= 360.0
        name, count = match[1], int(match[2])
        orientation = _rotation(np.radians(angle))
        rotations = [_rotation(i * 2 * np.pi / count) for i in range(count)]
        matrices = rotations.copy()
        if name == "D":
            reflection = orientation @ np.diag([1.0, -1.0]) @ orientation.T
            matrices.extend(matrix @ reflection for matrix in rotations)
        object.__setattr__(self, "name", f"{name}{count}")
        object.__setattr__(self, "orientation_degrees", angle)
        object.__setattr__(self, "rotations", count)
        object.__setattr__(
            self, "operations", tuple(CartesianPointOperation(matrix) for matrix in matrices)
        )

    @property
    def cone(self) -> SymmetryCone:
        return SymmetryCone(
            self.orientation_degrees, self.orientation_degrees + 360 / len(self.operations)
        )


def _validate_domain(domain, symmetry: ContentSymmetry) -> None:
    bounds = domain.bounds()
    scale = max(bounds[1] - bounds[0], bounds[3] - bounds[2], np.finfo(float).tiny)
    if isinstance(domain, (PropagationDisk, EvanescentDisk, PupilDomain)):
        if any(
            not np.allclose(op.apply(domain.center)[:2], domain.center, rtol=0, atol=scale * 1e-9)
            for op in symmetry.operations
        ):
            raise ValueError("domain does not preserve the declared content symmetry")
    elif isinstance(domain, (PolygonKDomain, PolygonDomain)):
        vertices = (
            domain.polygon.vertices if isinstance(domain, PolygonKDomain) else domain.vertices
        )
        tree = cKDTree(vertices[:, :2])
        for operation in symmetry.operations:
            distances, indices = tree.query(operation.apply(vertices)[:, :2])
            if np.any(distances > scale * 1e-9) or len(set(indices)) != len(vertices):
                raise ValueError("domain does not preserve the declared content symmetry")
    elif isinstance(domain, IntersectionDomain):
        for member in domain.domains:
            _validate_domain(member, symmetry)
    else:
        raise ValueError("content-symmetry domain validation is not implemented for this domain")


def prepare_symmetric_grid(domain, grid, symmetry: ContentSymmetry, *, adapt: bool):
    _validate_domain(domain, symmetry)
    if isinstance(grid, CartesianGrid) and adapt:
        if any(abs(op.cartesian[0, 1]) > 1e-10 for op in symmetry.operations):
            grid = CartesianGrid((max(grid.shape), max(grid.shape)))
    elif isinstance(grid, PolarGrid) and grid.azimuthal % symmetry.rotations:
        if not adapt:
            raise ValueError(
                "angular count does not preserve content symmetry; use adapt_symmetry=True"
            )
        grid = PolarGrid(
            grid.radial, int(np.ceil(grid.azimuthal / symmetry.rotations)) * symmetry.rotations
        )
    elif (
        isinstance(grid, CircularGrid)
        and grid.azimuthal is not None
        and grid.azimuthal % symmetry.rotations
    ):
        if not adapt:
            raise ValueError(
                "outer ring count does not preserve content symmetry; use adapt_symmetry=True"
            )
        grid = replace(
            grid, azimuthal=int(np.ceil(grid.azimuthal / symmetry.rotations)) * symmetry.rotations
        )
    return grid


@dataclass(frozen=True, slots=True)
class SymmetryConeDomain:
    domain: Any
    cone: SymmetryCone

    @property
    def area(self) -> float:
        from .sampling import _clip_half_plane

        x0, x1, y0, y1 = self.domain.bounds()
        scale = max(x1 - x0, y1 - y0)
        area = 0.0
        width = self.cone.end_degrees - self.cone.start_degrees
        parts = max(1, int(np.ceil(width / 180)))
        for start, end in zip(
            np.linspace(self.cone.start_degrees, self.cone.end_degrees, parts + 1)[:-1],
            np.linspace(self.cone.start_degrees, self.cone.end_degrees, parts + 1)[1:],
        ):
            vertices = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            a, b = np.radians([start, end])
            for normal in ([np.sin(a), -np.cos(a)], [-np.sin(b), np.cos(b)]):
                vertices = _clip_half_plane(
                    vertices, np.asarray(normal), 0.0, np.finfo(float).eps * scale * 128
                )
            if len(vertices) >= 3:
                area += intersection_area(self.domain, PolygonDomain(vertices))
        return area

    def contains(self, points: ArrayLike) -> np.ndarray:
        return self.domain.contains(points) & self.cone.contains(points)

    def bounds(self) -> tuple[float, float, float, float]:
        return self.domain.bounds()


@dataclass(frozen=True, slots=True, eq=False)
class NonPeriodicSymmetryReduction:
    """Physical point orbits and value transformations; no lattice translations."""

    target: SamplingResult
    source_points: np.ndarray
    source_ids: np.ndarray
    full_to_reduced: np.ndarray
    operations: tuple[tuple[CartesianPointOperation, ...], ...]
    content_symmetry: ContentSymmetry

    def __post_init__(self) -> None:
        count = len(self.source_points)
        if not isinstance(self.target, SamplingResult) or not isinstance(
            self.content_symmetry, ContentSymmetry
        ):
            raise TypeError("reduction requires a target sampling and content symmetry")
        points, ids = np.asarray(self.source_points), np.asarray(self.source_ids)
        indices = np.asarray(self.full_to_reduced)
        if (
            count == 0
            or points.shape != (count, 2)
            or not np.all(np.isfinite(points))
            or ids.shape != (count,)
            or not np.issubdtype(ids.dtype, np.integer)
            or len(np.unique(ids)) != count
        ):
            raise ValueError("representative points and unique integer identities must align")
        if (
            indices.shape != (len(self.target.points),)
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            or np.any(indices >= count)
        ):
            raise ValueError("full_to_reduced must identify every target point")
        if len(self.operations) != len(indices) or any(
            not group or not all(isinstance(op, CartesianPointOperation) for op in group)
            for group in self.operations
        ):
            raise ValueError("every target must have concrete symmetry generators")
        bounds = self.target.domain.bounds()
        tolerance = max(bounds[1] - bounds[0], bounds[3] - bounds[2]) * 1e-8
        for point, index, group in zip(self.target.points, indices, self.operations):
            if any(
                not np.allclose(op.apply(points[index])[:2], point, rtol=0, atol=tolerance)
                for op in group
            ):
                raise ValueError("symmetry generators do not map representatives to target points")
        object.__setattr__(self, "source_points", _readonly(self.source_points, float))
        object.__setattr__(self, "source_ids", _readonly(self.source_ids, np.int64))
        object.__setattr__(self, "full_to_reduced", _readonly(self.full_to_reduced, np.int64))

    @property
    def speedup(self) -> float:
        return len(self.target.points) / len(self.source_points)

    def expand(self, data: FieldSamples, *, verify: bool = True) -> FieldSamples:
        if (
            not isinstance(data, FieldSamples)
            or not np.array_equal(data.sampling.point_ids, self.source_ids)
            or not np.array_equal(data.sampling.points, self.source_points)
        ):
            raise ValueError("field sampling does not match symmetry representatives")
        values, valid = [], []
        for point, index, operations in zip(
            self.target.points, self.full_to_reduced, self.operations
        ):
            transformed = [
                data.representation.transform(
                    data.values[index], operation, self.source_points[index], point, (0, 0)
                )
                for operation in operations
            ]
            if (
                verify
                and data.valid[index]
                and any(
                    not np.allclose(value, transformed[0], rtol=1e-8, atol=1e-11)
                    for value in transformed[1:]
                )
            ):
                raise ValueError(
                    "response is inconsistent with equivalent content-symmetry operations"
                )
            values.append(transformed[0])
            valid.append(data.valid[index])
        target = self.target
        if data.sampling.kz is not None:
            target = replace(target, kz=data.sampling.kz[self.full_to_reduced])
        return FieldSamples(
            target, np.asarray(values), data.representation, np.asarray(valid), data.units
        )


def _symmetry_permutations(full: SamplingResult, symmetry: ContentSymmetry):
    scale = max(
        full.domain.bounds()[1] - full.domain.bounds()[0],
        full.domain.bounds()[3] - full.domain.bounds()[2],
    )
    tree = cKDTree(full.points)
    permutations = []
    for operation in symmetry.operations:
        distances, indices = tree.query(operation.apply(full.points)[:, :2])
        if np.any(distances > scale * 1e-8) or len(np.unique(indices)) != len(full.points):
            raise ValueError(
                "grid does not preserve content symmetry; adapt the grid or use CircularGrid"
            )
        if not np.allclose(
            full.normalized_weights[indices], full.normalized_weights, rtol=1e-9, atol=1e-14
        ):
            raise ValueError("quadrature weights do not preserve content symmetry")
        permutations.append(indices)
    return permutations


def reduce_sampling(full: SamplingResult, symmetry: ContentSymmetry, cone) -> SamplingResult:
    if isinstance(cone, str):
        if cone != "irreducible":
            raise ValueError("named symmetry cone must be 'irreducible'")
        cone = symmetry.cone
    elif isinstance(cone, tuple):
        cone = SymmetryCone(*cone)
    if not isinstance(cone, SymmetryCone):
        raise TypeError("symmetry_cone must be 'irreducible', SymmetryCone, or an angular pair")
    if full.symmetry_reduction is not None:
        raise ValueError("sampling is already symmetry-reduced")
    permutations = _symmetry_permutations(full, symmetry)
    parents = np.arange(len(full.points))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return int(index)

    for indices in permutations:
        for first, second in enumerate(indices):
            a, b = root(first), root(second)
            parents[max(a, b)] = min(a, b)
    groups = {}
    for index in range(len(full.points)):
        groups.setdefault(root(index), []).append(index)
    inside = cone.contains(full.points)
    representatives = []
    for members in groups.values():
        candidates = [index for index in members if inside[index]]
        if not candidates:
            raise ValueError(
                "symmetry cone does not contain a representative of every physical orbit"
            )
        representatives.append(min(candidates))
    representatives.sort()
    lookup = {root(index): reduced for reduced, index in enumerate(representatives)}
    full_to_reduced = np.array([lookup[root(index)] for index in range(len(full.points))])
    weights = np.bincount(
        full_to_reduced, weights=full.normalized_weights, minlength=len(representatives)
    )
    operations = tuple(
        tuple(
            symmetry.operations[j]
            for j, permutation in enumerate(permutations)
            if permutation[representatives[index]] == target
        )
        for target, index in enumerate(full_to_reduced)
    )
    reduction = NonPeriodicSymmetryReduction(
        full,
        full.points[representatives],
        full.point_ids[representatives],
        full_to_reduced,
        operations,
        symmetry,
    )
    cells = (
        None
        if full.integration_cells is None
        else tuple(
            OrbitIntegrationCell(
                full.integration_cells[index], weights[j] * full.integration_element
            )
            for j, index in enumerate(representatives)
        )
    )
    return SamplingResult(
        full.points[representatives],
        weights,
        full.integration_element,
        full.domain,
        full.point_ids[representatives],
        None if full.kz is None else full.kz[representatives],
        {
            **full.metadata,
            "content_symmetry": symmetry,
            "coordinate_domain": SymmetryConeDomain(full.domain, cone),
            "symmetry_cone": cone,
        },
        cells,
        reduction,
    )


__all__ = [
    "CartesianPointOperation",
    "ContentSymmetry",
    "SymmetryCone",
    "SymmetryConeDomain",
    "NonPeriodicSymmetryReduction",
]
