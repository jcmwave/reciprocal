"""Common Brillouin-zone sampling and symmetry-reduction interface."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reciprocal.brillouin_zone import BrillouinZone
from reciprocal.cells.sampling import CellSampler
from reciprocal.numerics import DEFAULT_TOLERANCES, Tolerances
from reciprocal.reciprocal_mesh import (
    MeshReduction,
    MonkhorstPackGrid,
    _mesh_preserving_actions,
    _reduce_monkhorst_pack,
)
from reciprocal.sampling import MaxSpacing, PointCounts, SamplingConstraint, SamplingResult
from reciprocal.symmetry import PointOperation

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
PlacementCandidate: TypeAlias = tuple[
    tuple[float, int, float, float, int, int],
    np.ndarray,
    PointOperation,
    np.ndarray,
]


def _readonly(value: ArrayLike, dtype: object) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


class ZoneSamplingError(ValueError):
    """Base class for valid but incompatible zone-sampling requests."""


class IncompatibleGridSymmetryError(ZoneSamplingError):
    """Raised when a grid does not preserve a required point group."""


class CanonicalPlacementError(ZoneSamplingError):
    """Raised when reduced representatives cannot be placed in the canonical IBZ."""


class ZoneRegion(str, Enum):
    """Coordinate-count region requested from a Brillouin-zone sampler."""

    BZ = "bz"
    IRREDUCIBLE = "irreducible"


class RepresentativePlacement(str, Enum):
    """Where symmetry-reduced representatives are expressed."""

    COMPUTATIONAL = "computational"
    CANONICAL_IBZ = "canonical_ibz"


@dataclass(frozen=True, slots=True)
class BoundaryGrid:
    """Endpoint-inclusive polygon grid with periodic boundary ownership."""

    constraint: PointCounts | MaxSpacing = PointCounts(5)
    center: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if not isinstance(self.constraint, (PointCounts, MaxSpacing)):
            raise TypeError("constraint must be PointCounts or MaxSpacing")
        center = np.asarray(self.center, dtype=float)
        if center.shape != (2,) or not np.all(np.isfinite(center)):
            raise ValueError("center must contain two finite coordinates")
        object.__setattr__(self, "center", (float(center[0]), float(center[1])))


ZoneGrid: TypeAlias = BoundaryGrid | MonkhorstPackGrid


@dataclass(frozen=True, slots=True, eq=False)
class RepresentativeMap:
    """Provenance for computational representatives placed in a canonical IBZ."""

    source_indices: IntArray
    operations: tuple[PointOperation, ...]
    reciprocal_shifts: IntArray
    source_points: FloatArray
    placed_points: FloatArray

    def __post_init__(self) -> None:
        source_indices = np.asarray(self.source_indices)
        shifts = np.asarray(self.reciprocal_shifts)
        source = np.asarray(self.source_points, dtype=float)
        placed = np.asarray(self.placed_points, dtype=float)
        count = len(source_indices)
        if source_indices.shape != (count,) or not np.issubdtype(
            source_indices.dtype, np.integer
        ):
            raise ValueError("source_indices must be a one-dimensional integer array")
        if len(np.unique(source_indices)) != count:
            raise ValueError("source_indices must be unique")
        if shifts.shape != (count, 2) or not np.issubdtype(shifts.dtype, np.integer):
            raise ValueError("reciprocal_shifts must have shape (N, 2) and integer dtype")
        if source.shape != (count, 2) or placed.shape != (count, 2):
            raise ValueError("source_points and placed_points must have shape (N, 2)")
        if not np.all(np.isfinite(source)) or not np.all(np.isfinite(placed)):
            raise ValueError("representative coordinates must be finite")
        if len(self.operations) != count or not all(
            isinstance(operation, PointOperation) for operation in self.operations
        ):
            raise ValueError("operations must contain one PointOperation per representative")
        object.__setattr__(self, "source_indices", _readonly(source_indices, np.int64))
        object.__setattr__(self, "reciprocal_shifts", _readonly(shifts, np.int64))
        object.__setattr__(self, "source_points", _readonly(source, float))
        object.__setattr__(self, "placed_points", _readonly(placed, float))
        object.__setattr__(self, "operations", tuple(self.operations))


@dataclass(frozen=True, slots=True)
class GridSymmetry:
    """Lattice operations and the subgroup preserving a configured grid."""

    lattice_operations: tuple[PointOperation, ...]
    preserving_operations: tuple[PointOperation, ...]

    def __post_init__(self) -> None:
        if not self.lattice_operations or not self.preserving_operations:
            raise ValueError("grid symmetry requires nonempty operation groups")
        if not all(
            isinstance(operation, PointOperation)
            for operation in self.lattice_operations + self.preserving_operations
        ):
            raise TypeError("grid symmetry entries must be PointOperation values")
        lattice = tuple(self.lattice_operations)
        preserving = tuple(self.preserving_operations)
        if any(operation not in lattice for operation in preserving):
            raise ValueError("preserving operations must belong to the lattice point group")
        object.__setattr__(self, "lattice_operations", lattice)
        object.__setattr__(self, "preserving_operations", preserving)

    @property
    def preserves_full_group(self) -> bool:
        return len(self.preserving_operations) == len(self.lattice_operations) and all(
            operation in self.preserving_operations for operation in self.lattice_operations
        )


ZoneReduction = MeshReduction


def _typed_constraint(constraint: SamplingConstraint | None) -> PointCounts | MaxSpacing:
    if constraint is None:
        return PointCounts(5)
    if isinstance(constraint, (PointCounts, MaxSpacing)):
        return constraint
    if not isinstance(constraint, Mapping):
        raise TypeError("constraint must be PointCounts, MaxSpacing, a legacy mapping, or None")
    kind = constraint.get("type")
    value = constraint.get("value")
    if kind == "n_points":
        array = np.asarray(value)
        if array.ndim == 0:
            return PointCounts(array.item())
        if array.shape == (2,):
            return PointCounts(array[0].item(), array[1].item())
    if kind == "max_length":
        return MaxSpacing(float(value))
    raise ValueError("legacy constraint must specify 'n_points' or 'max_length'")


def _boundary_grid(
    constraint: SamplingConstraint | None,
    center: ArrayLike | None,
) -> BoundaryGrid:
    center_array = np.zeros(2) if center is None else np.asarray(center, dtype=float)
    if center_array.shape == (3,):
        if center_array[2] != 0.0:
            raise ValueError("center must lie in the x-y plane")
        center_array = center_array[:2]
    if center_array.shape != (2,) or not np.all(np.isfinite(center_array)):
        raise ValueError("center must be a finite planar point")
    return BoundaryGrid(
        _typed_constraint(constraint),
        (float(center_array[0]), float(center_array[1])),
    )


def _validate_request(
    zone: BrillouinZone,
    grid: ZoneGrid,
    tolerances: Tolerances,
    require_full_symmetry: bool,
) -> None:
    if not isinstance(zone, BrillouinZone):
        raise TypeError("zone must be a BrillouinZone")
    if not isinstance(grid, (BoundaryGrid, MonkhorstPackGrid)):
        raise TypeError("grid must be BoundaryGrid or MonkhorstPackGrid")
    if not isinstance(tolerances, Tolerances):
        raise TypeError("tolerances must be a Tolerances instance")
    if not isinstance(require_full_symmetry, bool):
        raise TypeError("require_full_symmetry must be a bool")


def _annotated_result(
    result: SamplingResult,
    *,
    grid: ZoneGrid,
    region: ZoneRegion,
    placement: RepresentativePlacement,
    operations: tuple[PointOperation, ...],
    coordinate_domain: object | None = None,
) -> SamplingResult:
    metadata = dict(result.metadata)
    metadata.update(
        {
            "grid": grid,
            "region": region,
            "placement": placement,
            "symmetry_operations": operations,
            "source_region": region.value,
            "integration_region": ZoneRegion.BZ,
        }
    )
    if coordinate_domain is None:
        metadata.pop("coordinate_domain", None)
    else:
        metadata["coordinate_domain"] = coordinate_domain
    return SamplingResult(
        result.points,
        result.normalized_weights,
        result.integration_element,
        result.domain,
        result.point_ids,
        result.kz,
        metadata,
    )


def _sample_boundary_full(
    zone: BrillouinZone,
    grid: BoundaryGrid,
    cell_sampler: CellSampler | None = None,
) -> SamplingResult:
    sampler = CellSampler() if cell_sampler is None else cell_sampler
    sample = sampler.sample(zone.cell, grid.constraint, np.asarray(grid.center))
    return _annotated_result(
        sample,
        grid=grid,
        region=ZoneRegion.BZ,
        placement=RepresentativePlacement.COMPUTATIONAL,
        operations=zone.point_group,
    )


def _canonical_fractional(value: np.ndarray, tolerance: float) -> np.ndarray:
    canonical = value - np.floor(value + 0.5)
    half = np.isclose(np.abs(canonical), 0.5, rtol=0.0, atol=tolerance)
    canonical[half] = -0.5
    canonical[np.isclose(canonical, 0.0, rtol=0.0, atol=tolerance)] = 0.0
    return canonical


def _orbit_signature(
    fractional: np.ndarray,
    operations: tuple[PointOperation, ...],
    decimals: int,
    tolerance: float,
) -> tuple[float, float]:
    members = []
    for operation in operations:
        transformed = operation.fractional @ fractional
        canonical = _canonical_fractional(transformed, tolerance)
        rounded = np.round(canonical, decimals=decimals)
        members.append((float(rounded[0]), float(rounded[1])))
    return min(members)


def _preserving_operations_for_points(
    zone: BrillouinZone,
    points: np.ndarray,
    tolerances: Tolerances,
) -> tuple[PointOperation, ...]:
    basis = zone.cell.basis[:, :2]
    fractional = points @ np.linalg.inv(basis)
    decimals = max(8, int(np.ceil(-np.log10(max(tolerances.relative, 1e-15)))) + 2)
    keys = Counter(
        tuple(np.round(_canonical_fractional(point, tolerances.relative), decimals=decimals))
        for point in fractional
    )
    preserving = []
    for operation in zone.point_group:
        transformed = fractional @ operation.fractional.T
        transformed_keys = Counter(
            tuple(np.round(_canonical_fractional(point, tolerances.relative), decimals=decimals))
            for point in transformed
        )
        if transformed_keys == keys:
            preserving.append(operation)
    if not preserving:
        raise RuntimeError("boundary-grid preserving subgroup is empty")
    return tuple(preserving)


def _reduce_boundary(
    zone: BrillouinZone,
    grid: BoundaryGrid,
    tolerances: Tolerances,
    cell_sampler: CellSampler | None = None,
) -> MeshReduction:
    full = _sample_boundary_full(zone, grid, cell_sampler)
    operations = _preserving_operations_for_points(zone, full.points, tolerances)
    basis = zone.cell.basis[:, :2]
    inverse_basis = np.linalg.inv(basis)
    fractional = full.points @ inverse_basis
    decimals = max(8, int(np.ceil(-np.log10(max(tolerances.relative, 1e-15)))) + 2)
    grouped: dict[tuple[float, float], list[int]] = {}
    for index, coordinate in enumerate(fractional):
        signature = _orbit_signature(
            coordinate,
            operations,
            decimals,
            tolerances.relative,
        )
        grouped.setdefault(signature, []).append(index)
    groups = sorted(grouped.values(), key=lambda indices: indices[0])
    representatives = np.asarray([indices[0] for indices in groups], dtype=np.int64)
    mapping = np.empty(len(full.points), dtype=np.int64)
    for reduced_index, indices in enumerate(groups):
        mapping[indices] = reduced_index
    degeneracies = np.asarray([len(indices) for indices in groups], dtype=np.int64)
    weights = np.bincount(
        mapping,
        weights=full.normalized_weights,
        minlength=len(groups),
    )
    computational = SamplingResult(
        full.points[representatives],
        weights,
        zone.area,
        zone.cell.domain,
        point_ids=full.point_ids[representatives],
        metadata={
            "grid": grid,
            "region": ZoneRegion.IRREDUCIBLE,
            "placement": RepresentativePlacement.COMPUTATIONAL,
            "symmetry_operations": operations,
            "source_region": ZoneRegion.IRREDUCIBLE.value,
            "integration_region": ZoneRegion.BZ,
        },
    )
    return MeshReduction(
        full,
        computational,
        mapping,
        representatives,
        degeneracies,
        operations,
    )


def _place_in_canonical_ibz(
    reduction: MeshReduction,
    zone: BrillouinZone,
    grid: ZoneGrid,
) -> MeshReduction:
    basis = zone.cell.basis[:, :2]
    inverse_basis = np.linalg.inv(basis)
    placed_points = []
    selected_operations = []
    selected_shifts = []
    for source in reduction.irreducible.points:
        candidates: list[PlacementCandidate] = []
        for operation_index, operation in enumerate(reduction.operations):
            transformed = operation.apply(source)[:2]
            fractional = transformed @ inverse_basis
            for first_shift in range(-2, 3):
                for second_shift in range(-2, 3):
                    shift = np.array([first_shift, second_shift], dtype=np.int64)
                    candidate = (fractional + shift) @ basis
                    if zone.irreducible_domain.contains(candidate)[0]:
                        key = (
                            float(np.dot(shift, shift)),
                            operation_index,
                            float(candidate[0]),
                            float(candidate[1]),
                            first_shift,
                            second_shift,
                        )
                        candidates.append((key, candidate, operation, shift))
        if not candidates:
            raise CanonicalPlacementError(
                "could not place a reduced representative in the canonical IBZ"
            )
        _, point, operation, shift = min(candidates, key=lambda item: item[0])
        expected = operation.apply(source)[:2] + shift @ basis
        if not np.allclose(point, expected, rtol=1e-10, atol=1e-12):
            raise RuntimeError("canonical representative provenance is inconsistent")
        placed_points.append(point)
        selected_operations.append(operation)
        selected_shifts.append(shift)
    points = np.asarray(placed_points)
    if not np.all(zone.irreducible_domain.contains(points)):
        raise CanonicalPlacementError("placed representatives do not lie in the canonical IBZ")
    placement_map = RepresentativeMap(
        reduction.representative_indices,
        tuple(selected_operations),
        np.asarray(selected_shifts),
        reduction.irreducible.points,
        points,
    )
    placed = _annotated_result(
        SamplingResult(
            points,
            reduction.irreducible.normalized_weights,
            reduction.irreducible.integration_element,
            reduction.irreducible.domain,
            reduction.irreducible.point_ids,
            reduction.irreducible.kz,
            reduction.irreducible.metadata,
        ),
        grid=grid,
        region=ZoneRegion.IRREDUCIBLE,
        placement=RepresentativePlacement.CANONICAL_IBZ,
        operations=reduction.operations,
        coordinate_domain=zone.irreducible_domain,
    )
    return MeshReduction(
        reduction.full,
        placed,
        reduction.full_to_irreducible,
        reduction.representative_indices,
        reduction.degeneracies,
        reduction.operations,
        placement_map,
    )


def grid_symmetry(
    zone: BrillouinZone,
    grid: ZoneGrid,
    *,
    tolerances: Tolerances = DEFAULT_TOLERANCES,
) -> GridSymmetry:
    """Return the lattice group and subgroup preserving ``grid``."""

    _validate_request(zone, grid, tolerances, False)
    if isinstance(grid, BoundaryGrid):
        sample = _sample_boundary_full(zone, grid)
        preserving = _preserving_operations_for_points(zone, sample.points, tolerances)
    else:
        _, _, preserving, _ = _mesh_preserving_actions(
            zone,
            grid,
            tolerances=tolerances,
        )
    return GridSymmetry(zone.point_group, preserving)


def _symmetry_error(zone: BrillouinZone, grid: ZoneGrid, symmetry: GridSymmetry) -> str:
    description = (
        f"grid preserves {len(symmetry.preserving_operations)} of "
        f"{len(symmetry.lattice_operations)} lattice point-group operations"
    )
    if isinstance(grid, MonkhorstPackGrid):
        description += (
            f" (shape={grid.shape}, centering={grid.centering.value!r}, shift={grid.shift}); "
            "use symmetry-compatible dimensions and shifts, such as equal odd conventional "
            "counts or equal Gamma-centered counts for a hexagonal lattice"
        )
    return description


def reduce_brillouin_zone(
    zone: BrillouinZone,
    grid: ZoneGrid,
    *,
    placement: RepresentativePlacement | str = RepresentativePlacement.COMPUTATIONAL,
    require_full_symmetry: bool = False,
    tolerances: Tolerances = DEFAULT_TOLERANCES,
) -> ZoneReduction:
    """Reduce a configured BZ grid under its mesh-preserving point group."""

    _validate_request(zone, grid, tolerances, require_full_symmetry)
    try:
        resolved_placement = RepresentativePlacement(placement)
    except ValueError as error:
        raise ValueError("placement must be 'computational' or 'canonical_ibz'") from error
    if isinstance(grid, BoundaryGrid):
        reduction = _reduce_boundary(zone, grid, tolerances)
    else:
        legacy = _reduce_monkhorst_pack(zone, grid, tolerances=tolerances)
        full = _annotated_result(
            legacy.full,
            grid=grid,
            region=ZoneRegion.BZ,
            placement=RepresentativePlacement.COMPUTATIONAL,
            operations=legacy.operations,
        )
        reduced = _annotated_result(
            legacy.irreducible,
            grid=grid,
            region=ZoneRegion.IRREDUCIBLE,
            placement=RepresentativePlacement.COMPUTATIONAL,
            operations=legacy.operations,
        )
        reduction = MeshReduction(
            full,
            reduced,
            legacy.full_to_irreducible,
            legacy.representative_indices,
            legacy.degeneracies,
            legacy.operations,
        )
    symmetry = GridSymmetry(zone.point_group, reduction.operations)
    if require_full_symmetry and not symmetry.preserves_full_group:
        raise IncompatibleGridSymmetryError(_symmetry_error(zone, grid, symmetry))
    if resolved_placement is RepresentativePlacement.CANONICAL_IBZ:
        if not symmetry.preserves_full_group:
            raise CanonicalPlacementError(_symmetry_error(zone, grid, symmetry))
        reduction = _place_in_canonical_ibz(reduction, zone, grid)
    return reduction


def sample_brillouin_zone(
    zone: BrillouinZone,
    grid: ZoneGrid,
    *,
    region: ZoneRegion | str = ZoneRegion.BZ,
    placement: RepresentativePlacement | str | None = None,
    require_full_symmetry: bool = False,
    tolerances: Tolerances = DEFAULT_TOLERANCES,
) -> SamplingResult:
    """Sample a BZ using a boundary-aware or Monkhorst--Pack grid."""

    _validate_request(zone, grid, tolerances, require_full_symmetry)
    try:
        resolved_region = ZoneRegion(region)
    except ValueError as error:
        raise ValueError("region must be 'bz' or 'irreducible'") from error
    if placement is None:
        resolved_placement = (
            RepresentativePlacement.CANONICAL_IBZ
            if resolved_region is ZoneRegion.IRREDUCIBLE and isinstance(grid, BoundaryGrid)
            else RepresentativePlacement.COMPUTATIONAL
        )
    else:
        try:
            resolved_placement = RepresentativePlacement(placement)
        except ValueError as error:
            raise ValueError("placement must be 'computational' or 'canonical_ibz'") from error
    if resolved_region is ZoneRegion.BZ:
        if resolved_placement is not RepresentativePlacement.COMPUTATIONAL:
            raise CanonicalPlacementError("full-BZ sampling requires computational placement")
        if require_full_symmetry:
            symmetry = grid_symmetry(zone, grid, tolerances=tolerances)
            if not symmetry.preserves_full_group:
                raise IncompatibleGridSymmetryError(_symmetry_error(zone, grid, symmetry))
        if isinstance(grid, BoundaryGrid):
            return _sample_boundary_full(zone, grid)
        sample, _, operations, _ = _mesh_preserving_actions(
            zone,
            grid,
            tolerances=tolerances,
        )
        return _annotated_result(
            sample,
            grid=grid,
            region=ZoneRegion.BZ,
            placement=RepresentativePlacement.COMPUTATIONAL,
            operations=operations,
        )
    return reduce_brillouin_zone(
        zone,
        grid,
        placement=resolved_placement,
        require_full_symmetry=require_full_symmetry,
        tolerances=tolerances,
    ).reduced


__all__ = [
    "BoundaryGrid",
    "CanonicalPlacementError",
    "GridSymmetry",
    "IncompatibleGridSymmetryError",
    "RepresentativeMap",
    "RepresentativePlacement",
    "ZoneGrid",
    "ZoneReduction",
    "ZoneRegion",
    "ZoneSamplingError",
    "grid_symmetry",
    "reduce_brillouin_zone",
    "sample_brillouin_zone",
]
