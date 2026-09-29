"""Uniform reciprocal meshes and point-group reduction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from reciprocal.brillouin_zone import BrillouinZone
from reciprocal.numerics import DEFAULT_TOLERANCES, Tolerances
from reciprocal.sampling import SamplingResult
from reciprocal.symmetry import PointOperation

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _readonly(value: object, dtype: object) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _canonical_fractional(points: np.ndarray) -> np.ndarray:
    """Reduce fractional rows to the half-open interval ``[-1/2, 1/2)``."""

    return points - np.floor(points + 0.5)


class GridCentering(str, Enum):
    GAMMA = "gamma"
    MONKHORST_PACK = "monkhorst_pack"


@dataclass(frozen=True, slots=True)
class MonkhorstPackGrid:
    """Specification of a uniform two-dimensional reciprocal mesh.

    ``shift`` is measured in mesh steps. Thus ``0.5`` selects a half-step
    shift, independently of the number of points on the corresponding axis.
    """

    shape: tuple[int, int]
    centering: GridCentering | str = GridCentering.MONKHORST_PACK
    shift: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or any(isinstance(item, bool) for item in self.shape):
            raise ValueError("shape must contain two positive integers")
        shape = tuple(int(item) for item in self.shape)
        if shape != tuple(self.shape) or any(item <= 0 for item in shape):
            raise ValueError("shape must contain two positive integers")
        try:
            centering = GridCentering(self.centering)
        except ValueError as error:
            raise ValueError("centering must be 'gamma' or 'monkhorst_pack'") from error
        shift = np.asarray(self.shift, dtype=float)
        if shift.shape != (2,) or not np.all(np.isfinite(shift)):
            raise ValueError("shift must contain two finite mesh-step values")
        shift = np.mod(shift, 1.0)
        shift[np.isclose(shift, 1.0)] = 0.0
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "centering", centering)
        object.__setattr__(self, "shift", (float(shift[0]), float(shift[1])))


@dataclass(frozen=True, slots=True, eq=False)
class MonkhorstPackMetadata:
    grid: MonkhorstPackGrid
    mesh_indices: IntArray
    fractional_points: FloatArray
    canonical_keys: FloatArray
    degeneracies: IntArray | None = None

    def __post_init__(self) -> None:
        indices = np.asarray(self.mesh_indices)
        fractional = np.asarray(self.fractional_points, dtype=float)
        keys = np.asarray(self.canonical_keys, dtype=float)
        if indices.ndim != 2 or indices.shape[1] != 2 or not np.issubdtype(
            indices.dtype, np.integer
        ):
            raise ValueError("mesh_indices must have shape (N, 2) and integer dtype")
        if fractional.shape != indices.shape or keys.shape != indices.shape:
            raise ValueError("fractional mesh arrays must have shape (N, 2)")
        if not np.all(np.isfinite(fractional)) or not np.all(np.isfinite(keys)):
            raise ValueError("fractional mesh arrays must be finite")
        degeneracies = None
        if self.degeneracies is not None:
            degeneracies = np.asarray(self.degeneracies)
            if degeneracies.shape != (len(indices),) or not np.issubdtype(
                degeneracies.dtype, np.integer
            ):
                raise ValueError("degeneracies must contain one integer per point")
            if np.any(degeneracies <= 0):
                raise ValueError("degeneracies must be positive")
        object.__setattr__(self, "mesh_indices", _readonly(indices, np.int64))
        object.__setattr__(self, "fractional_points", _readonly(fractional, float))
        object.__setattr__(self, "canonical_keys", _readonly(keys, float))
        object.__setattr__(
            self,
            "degeneracies",
            None if degeneracies is None else _readonly(degeneracies, np.int64),
        )


@dataclass(frozen=True, slots=True, eq=False)
class MeshReduction:
    full: SamplingResult
    irreducible: SamplingResult
    full_to_irreducible: IntArray
    representative_indices: IntArray
    degeneracies: IntArray
    operations: tuple[PointOperation, ...]

    def __post_init__(self) -> None:
        mapping = np.asarray(self.full_to_irreducible)
        representatives = np.asarray(self.representative_indices)
        degeneracies = np.asarray(self.degeneracies)
        if mapping.shape != (len(self.full.points),):
            raise ValueError("full_to_irreducible must contain one entry per full point")
        if representatives.shape != (len(self.irreducible.points),):
            raise ValueError("representative_indices has the wrong length")
        if degeneracies.shape != representatives.shape:
            raise ValueError("degeneracies has the wrong length")
        if not all(np.issubdtype(item.dtype, np.integer) for item in (mapping, representatives, degeneracies)):
            raise TypeError("mesh-reduction index arrays must contain integers")
        if np.any(mapping < 0) or np.any(mapping >= len(representatives)):
            raise ValueError("full_to_irreducible contains an invalid index")
        if np.any(representatives < 0) or np.any(representatives >= len(self.full.points)):
            raise ValueError("representative_indices contains an invalid index")
        if np.any(degeneracies <= 0) or int(np.sum(degeneracies)) != len(self.full.points):
            raise ValueError("degeneracies must be positive and sum to the full mesh size")
        if not self.operations:
            raise ValueError("mesh reduction requires a nonempty preserving subgroup")
        object.__setattr__(self, "full_to_irreducible", _readonly(mapping, np.int64))
        object.__setattr__(self, "representative_indices", _readonly(representatives, np.int64))
        object.__setattr__(self, "degeneracies", _readonly(degeneracies, np.int64))
        object.__setattr__(self, "operations", tuple(self.operations))


def _fractional_grid(grid: MonkhorstPackGrid) -> tuple[np.ndarray, np.ndarray]:
    first, second = np.meshgrid(
        np.arange(grid.shape[0], dtype=np.int64),
        np.arange(grid.shape[1], dtype=np.int64),
        indexing="ij",
    )
    indices = np.column_stack((first.ravel(), second.ravel()))
    shape = np.asarray(grid.shape, dtype=float)
    if grid.centering is GridCentering.MONKHORST_PACK:
        fractional = (indices - 0.5 * (shape - 1.0)) / shape
    else:
        fractional = (indices - np.floor(shape / 2.0)) / shape
    fractional += np.asarray(grid.shift, dtype=float) / shape
    return indices, fractional


def _representatives_in_zone(keys: np.ndarray, zone: BrillouinZone) -> tuple[np.ndarray, np.ndarray]:
    basis = zone.cell.basis[:, :2]
    fractional = []
    cartesian = []
    for key in keys:
        candidates = []
        for first in range(-2, 3):
            for second in range(-2, 3):
                candidate = key + np.array([first, second])
                point = candidate @ basis
                if zone.cell.domain.contains(point)[0]:
                    candidates.append((candidate, point))
        if not candidates:
            raise RuntimeError("could not place mesh point in the Brillouin zone")
        candidates.sort(
            key=lambda item: (
                float(np.linalg.norm(item[1])),
                float(item[0][0]),
                float(item[0][1]),
            )
        )
        selected_fractional, selected_cartesian = candidates[0]
        fractional.append(selected_fractional)
        cartesian.append(selected_cartesian)
    return np.vstack(fractional), np.vstack(cartesian)


def _full_mesh(zone: BrillouinZone, grid: MonkhorstPackGrid) -> SamplingResult:
    if not isinstance(zone, BrillouinZone):
        raise TypeError("zone must be a BrillouinZone")
    if not isinstance(grid, MonkhorstPackGrid):
        raise TypeError("grid must be a MonkhorstPackGrid")
    indices, raw_fractional = _fractional_grid(grid)
    keys = _canonical_fractional(raw_fractional)
    fractional, points = _representatives_in_zone(keys, zone)
    metadata = MonkhorstPackMetadata(grid, indices, fractional, keys)
    count = len(points)
    return SamplingResult(
        points,
        np.full(count, 1.0 / count),
        zone.area,
        zone.cell.domain,
        metadata={"monkhorst_pack": metadata, "source_region": "bz"},
    )


def _key(value: np.ndarray, decimals: int) -> tuple[float, float]:
    rounded = np.round(_canonical_fractional(np.asarray(value)), decimals=decimals)
    return float(rounded[0]), float(rounded[1])


def reduce_monkhorst_pack(
    zone: BrillouinZone,
    grid: MonkhorstPackGrid,
    *,
    tolerances: Tolerances = DEFAULT_TOLERANCES,
) -> MeshReduction:
    """Reduce a uniform mesh by the point operations that preserve it."""

    if not isinstance(tolerances, Tolerances):
        raise TypeError("tolerances must be a Tolerances instance")
    full = _full_mesh(zone, grid)
    full_metadata = full.metadata["monkhorst_pack"]
    if not isinstance(full_metadata, MonkhorstPackMetadata):
        raise RuntimeError("full mesh is missing Monkhorst-Pack metadata")
    keys = full_metadata.canonical_keys
    decimals = max(8, int(np.ceil(-np.log10(max(tolerances.relative, 1e-15)))) + 2)
    lookup = {_key(point, decimals): index for index, point in enumerate(keys)}
    if len(lookup) != len(keys):
        raise RuntimeError("mesh contains duplicate reciprocal equivalence keys")

    permutations: list[np.ndarray] = []
    operations: list[PointOperation] = []
    for operation in zone.point_group:
        transformed = keys @ operation.fractional.T
        try:
            permutation = np.asarray(
                [lookup[_key(point, decimals)] for point in transformed], dtype=np.int64
            )
        except KeyError:
            continue
        if len(np.unique(permutation)) == len(keys):
            operations.append(operation)
            permutations.append(permutation)
    if not operations:
        raise RuntimeError("mesh-preserving subgroup is empty")

    unassigned = set(range(len(keys)))
    orbits: list[list[int]] = []
    while unassigned:
        seed = min(unassigned)
        orbit = {int(permutation[seed]) for permutation in permutations}
        # Closure is cheap for the small crystallographic point groups and also
        # protects against future operation lists that are not pre-closed.
        previous_size = -1
        while len(orbit) != previous_size:
            previous_size = len(orbit)
            orbit.update(
                int(permutation[index])
                for permutation in permutations
                for index in tuple(orbit)
            )
        ordered = sorted(orbit)
        orbits.append(ordered)
        unassigned.difference_update(ordered)
    orbits.sort(key=lambda orbit: orbit[0])

    representative_indices = np.asarray([orbit[0] for orbit in orbits], dtype=np.int64)
    degeneracies = np.asarray([len(orbit) for orbit in orbits], dtype=np.int64)
    mapping = np.empty(len(keys), dtype=np.int64)
    for reduced_index, orbit in enumerate(orbits):
        mapping[orbit] = reduced_index
    reduced_metadata = MonkhorstPackMetadata(
        grid,
        full_metadata.mesh_indices[representative_indices],
        full_metadata.fractional_points[representative_indices],
        full_metadata.canonical_keys[representative_indices],
        degeneracies,
    )
    irreducible = SamplingResult(
        full.points[representative_indices],
        degeneracies / len(full.points),
        full.integration_element,
        full.domain,
        point_ids=full.point_ids[representative_indices],
        metadata={
            "monkhorst_pack": reduced_metadata,
            "source_region": "irreducible_mesh",
            "symmetry_operations": tuple(operations),
        },
    )
    return MeshReduction(
        full,
        irreducible,
        mapping,
        representative_indices,
        degeneracies,
        tuple(operations),
    )


def sample_monkhorst_pack(
    zone: BrillouinZone,
    grid: MonkhorstPackGrid,
    *,
    irreducible: bool = False,
    tolerances: Tolerances = DEFAULT_TOLERANCES,
) -> SamplingResult:
    """Sample a full or point-group-reduced uniform reciprocal mesh."""

    if not isinstance(irreducible, bool):
        raise TypeError("irreducible must be a bool")
    if irreducible:
        return reduce_monkhorst_pack(zone, grid, tolerances=tolerances).irreducible
    return _full_mesh(zone, grid)


__all__ = [
    "GridCentering",
    "MeshReduction",
    "MonkhorstPackGrid",
    "MonkhorstPackMetadata",
    "reduce_monkhorst_pack",
    "sample_monkhorst_pack",
]
