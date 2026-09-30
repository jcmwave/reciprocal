import numpy as np
import pytest

from reciprocal import (
    BoundaryGrid,
    CanonicalPlacementError,
    GridCentering,
    IncompatibleGridSymmetryError,
    Lattice,
    MaxSpacing,
    MeshReduction,
    MonkhorstPackGrid,
    PointCounts,
    RepresentativePlacement,
    ZoneRegion,
    grid_symmetry,
    reduce_brillouin_zone,
    sample_brillouin_zone,
)
from reciprocal.brillouin_zone import BrillouinZoneSampler
from reciprocal.reciprocal_mesh import reduce_monkhorst_pack, sample_monkhorst_pack


CASES = [
    (1.3, 1.0, 73.0),
    (1.5, 1.0, 90.0),
    (1.0, 1.0, 70.0),
    (1.0, 1.0, 90.0),
    (1.0, 1.0, 60.0),
]


def _zone(parameters=(1.0, 1.0, 90.0)):
    return Lattice.from_lengths_angle(*parameters).make_reciprocal().brillouin_zone


@pytest.mark.parametrize("parameters", CASES)
@pytest.mark.parametrize(
    "grid",
    [BoundaryGrid(PointCounts(5)), MonkhorstPackGrid((5, 5))],
)
def test_common_interface_full_reduced_and_canonical_contract(parameters, grid):
    zone = _zone(parameters)
    full = sample_brillouin_zone(zone, grid)
    reduction = reduce_brillouin_zone(zone, grid)
    canonical = reduce_brillouin_zone(
        zone,
        grid,
        placement=RepresentativePlacement.CANONICAL_IBZ,
        require_full_symmetry=True,
    )

    assert isinstance(reduction, MeshReduction)
    assert reduction.reduced is reduction.irreducible
    np.testing.assert_array_equal(reduction.full_to_reduced, reduction.full_to_irreducible)
    np.testing.assert_array_equal(
        reduction.degeneracies,
        np.bincount(reduction.full_to_reduced),
    )
    np.testing.assert_allclose(
        reduction.reduced.normalized_weights,
        np.bincount(
            reduction.full_to_reduced,
            weights=reduction.full.normalized_weights,
        ),
    )
    np.testing.assert_allclose(
        reduction.reduced.points,
        reduction.full.points[reduction.representative_indices],
    )
    assert full.integrate(np.ones(len(full.points))) == pytest.approx(zone.area)
    assert reduction.reduced.integrate(
        np.ones(len(reduction.reduced.points))
    ) == pytest.approx(zone.area)
    assert np.all(zone.irreducible_domain.contains(canonical.reduced.points))
    assert canonical.placement is not None
    assert canonical.reduced.metadata["coordinate_domain"] is zone.irreducible_domain


@pytest.mark.parametrize("parameters", CASES)
def test_canonical_placement_retains_operation_and_translation_provenance(parameters):
    zone = _zone(parameters)
    reduction = reduce_brillouin_zone(
        zone,
        MonkhorstPackGrid((7, 7)),
        placement="canonical_ibz",
    )
    placement = reduction.placement
    assert placement is not None
    np.testing.assert_array_equal(placement.source_indices, reduction.representative_indices)
    np.testing.assert_allclose(placement.placed_points, reduction.reduced.points)
    basis = zone.cell.basis[:, :2]
    for source, placed, operation, shift in zip(
        placement.source_points,
        placement.placed_points,
        placement.operations,
        placement.reciprocal_shifts,
    ):
        np.testing.assert_allclose(
            placed,
            operation.apply(source)[:2] + shift @ basis,
            rtol=1e-10,
            atol=1e-12,
        )


def test_hexagonal_grid_symmetry_distinguishes_even_mp_and_gamma_centering():
    zone = _zone((1.0, 1.0, 60.0))
    even_mp = MonkhorstPackGrid((12, 12))
    odd_mp = MonkhorstPackGrid((15, 15))
    even_gamma = MonkhorstPackGrid((12, 12), GridCentering.GAMMA)

    assert not grid_symmetry(zone, even_mp).preserves_full_group
    assert grid_symmetry(zone, odd_mp).preserves_full_group
    assert grid_symmetry(zone, even_gamma).preserves_full_group
    with pytest.raises(IncompatibleGridSymmetryError, match="preserves 4 of 12"):
        sample_brillouin_zone(zone, even_mp, require_full_symmetry=True)
    with pytest.raises(CanonicalPlacementError, match="preserves 4 of 12"):
        reduce_brillouin_zone(zone, even_mp, placement="canonical_ibz")


def test_asymmetric_boundary_grid_reports_its_actual_preserving_subgroup():
    zone = _zone((1.0, 1.0, 90.0))
    grid = BoundaryGrid(PointCounts(5, 7))
    symmetry = grid_symmetry(zone, grid)
    assert len(symmetry.lattice_operations) == 8
    assert len(symmetry.preserving_operations) == 4
    with pytest.raises(CanonicalPlacementError, match="preserves 4 of 8"):
        sample_brillouin_zone(zone, grid, region="irreducible")


def test_boundary_grid_supports_max_spacing_and_typed_metadata():
    zone = _zone()
    grid = BoundaryGrid(MaxSpacing(0.3 * min(np.linalg.norm(v) for v in zone.cell.basis)))
    result = sample_brillouin_zone(zone, grid, region=ZoneRegion.BZ)
    assert result.metadata["grid"] is grid
    assert result.metadata["region"] is ZoneRegion.BZ
    assert result.metadata["placement"] is RepresentativePlacement.COMPUTATIONAL
    assert result.metadata["integration_region"] is ZoneRegion.BZ


def test_compatibility_entry_points_delegate_to_common_results():
    zone = _zone()
    boundary = BoundaryGrid(PointCounts(5))
    sampler = BrillouinZoneSampler()
    common_full = sample_brillouin_zone(zone, boundary)
    common_ibz = sample_brillouin_zone(zone, boundary, region="irreducible")
    np.testing.assert_allclose(sampler.sample_full(zone).points, common_full.points)
    np.testing.assert_allclose(sampler.sample_full(zone).weights, common_full.weights)
    np.testing.assert_allclose(sampler.sample_irreducible(zone).points, common_ibz.points)
    np.testing.assert_allclose(sampler.sample_irreducible(zone).weights, common_ibz.weights)

    grid = MonkhorstPackGrid((5, 5))
    common_mp = sample_brillouin_zone(zone, grid)
    common_reduction = reduce_brillouin_zone(zone, grid)
    np.testing.assert_allclose(sample_monkhorst_pack(zone, grid).points, common_mp.points)
    np.testing.assert_array_equal(
        reduce_monkhorst_pack(zone, grid).full_to_reduced,
        common_reduction.full_to_reduced,
    )


def test_request_validation_is_explicit():
    zone = _zone()
    with pytest.raises(TypeError, match="PointCounts or MaxSpacing"):
        BoundaryGrid({"type": "n_points", "value": 5})  # type: ignore[arg-type]
    with pytest.raises(CanonicalPlacementError, match="full-BZ"):
        sample_brillouin_zone(
            zone,
            BoundaryGrid(),
            placement=RepresentativePlacement.CANONICAL_IBZ,
        )
    with pytest.raises(ValueError, match="region"):
        sample_brillouin_zone(zone, BoundaryGrid(), region="invalid")
