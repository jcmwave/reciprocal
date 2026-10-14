import numpy as np
import pytest

from reciprocal import (
    BoundaryGrid,
    KSpace,
    Lattice,
    MaxSpacing,
    MonkhorstPackGrid,
    PointCounts,
    resolve_point_counts,
    sample_brillouin_zone,
)
from reciprocal.sampling import normalize_constraint
from reciprocal.spectrum import CartesianGrid, CircularGrid, PolarGrid, resolve_sampling_grid


def test_one_and_two_dimensional_constraints_and_grid_placement():
    assert PointCounts(7).values == (7,)
    assert PointCounts(7, 5).values == (7, 5)
    assert PointCounts(7).second == 7  # 1.x attribute compatibility.
    assert resolve_point_counts(2.0, PointCounts(7)) == (7,)
    assert resolve_point_counts([1.0, 2.0], PointCounts(7)) == (7, 7)
    assert resolve_point_counts([1.0, 2.0], PointCounts(7, 5)) == (7, 5)
    assert resolve_point_counts([1.0, 2.0], MaxSpacing((0.3, 0.4))) == (4, 5)
    assert resolve_point_counts([1.0, 2.0], MaxSpacing((0.3, 0.4)), placement="endpoints") == (5, 6)
    assert resolve_point_counts(2.0, MaxSpacing(0.4), placement="closed") == (5,)
    with pytest.raises(ValueError, match="dimensions"):
        resolve_point_counts(1.0, PointCounts(5, 5))
    with pytest.raises(ValueError, match="dimensions"):
        resolve_point_counts(1.0, MaxSpacing((0.1, 0.2)))


@pytest.mark.parametrize("constraint", [PointCounts(9, 7), MaxSpacing((0.2, 0.3))])
def test_legacy_and_typed_constraints_resolve_to_identical_grids(constraint):
    assert normalize_constraint(constraint.as_legacy()).values == (
        (constraint.first, constraint.second)
        if isinstance(constraint, PointCounts)
        else constraint.values
    )
    assert resolve_point_counts([1.0, 2.0], constraint) == resolve_point_counts(
        [1.0, 2.0], constraint.as_legacy()
    )


@pytest.mark.parametrize("scheme", ["cartesian", "polar", "circular"])
def test_maximum_spacing_is_obeyed_by_all_spectrum_schemes(scheme):
    space = KSpace.propagating(2 * np.pi)
    hr, ha = 0.17, 0.23
    result = space.sample_domain(scheme, constraint=MaxSpacing((hr, ha)), include_cells=True)
    assert result.integrate(np.ones(len(result.points))) == pytest.approx(np.pi)
    cells = result.integration_cells
    if scheme == "cartesian":
        for cell in cells:
            x0, x1, y0, y1 = cell.bounds
            assert x1 - x0 <= hr * (1 + 1e-12)
            assert y1 - y0 <= ha * (1 + 1e-12)
    else:
        wedges = cells[1:] if scheme == "circular" else cells
        for cell in wedges:
            assert cell.outer_radius - cell.inner_radius <= hr * (1 + 1e-12)
            # Polar cells obey the bound even at their outer edge; circular
            # rings specify azimuthal separation at the actual sample radius.
            radius = (
                (cell.inner_radius + cell.outer_radius) / 2
                if scheme == "circular"
                else cell.outer_radius
            )
            assert radius * (cell.theta_end - cell.theta_start) <= ha * (1 + 1e-12)


def test_counts_define_scheme_specific_dimensions_and_explicit_grid_conflicts():
    space = KSpace.propagating(2 * np.pi)
    assert resolve_sampling_grid(
        space.spectrum_domain, "cartesian", PointCounts(5, 7)
    ) == CartesianGrid((5, 7))
    assert resolve_sampling_grid(space.spectrum_domain, "polar", PointCounts(5, 7)) == PolarGrid(
        5, 7
    )
    circular = resolve_sampling_grid(space.spectrum_domain, "circular", PointCounts(5, 24))
    assert circular == CircularGrid(5, 24)
    sampled = space.sample_domain("circular", constraint=PointCounts(5, 24))
    radii = np.linalg.norm(sampled.points, axis=1)
    assert np.count_nonzero(np.isclose(radii, radii.max())) == 24
    with pytest.raises(ValueError, match="explicitly resolved"):
        space.sample_domain(CartesianGrid((5, 5)), constraint=PointCounts(7))
    with pytest.raises(ValueError, match="azimuthal"):
        space.sample_domain("circular", constraint=PointCounts(1, 8))


def test_periodic_boundary_and_monkhorst_pack_use_physical_axis_spacing():
    direct = Lattice.from_lengths_angle(2 * np.pi, np.pi, 90)
    zone = direct.make_reciprocal().brillouin_zone
    constraint = MaxSpacing((0.3, 0.4))
    mesh = MonkhorstPackGrid.from_constraint(zone, constraint)
    assert mesh.shape == (4, 5)
    assert np.all(
        np.linalg.norm(zone.cell.basis[:, :2], axis=1) / mesh.shape <= np.array(constraint.values)
    )
    result = sample_brillouin_zone(zone, BoundaryGrid(constraint))
    fractional = result.points @ np.linalg.inv(zone.cell.basis[:, :2])
    for axis, maximum in enumerate(constraint.values):
        coordinates = np.unique(np.round(fractional[:, axis], 12))
        assert np.max(np.diff(coordinates)) * np.linalg.norm(zone.cell.basis[axis]) <= maximum * (
            1 + 1e-10
        )
    space = KSpace.propagating(2 * np.pi)
    plan = space.with_periodic_structure(direct).plan(constraint=constraint, target="bz")
    assert plan.representatives.integrate(
        np.ones(len(plan.representatives.points))
    ) == pytest.approx(zone.area)


@pytest.mark.parametrize(
    "value", [0, -1, np.nan, np.inf, (0.1, -0.2), (0.1,), (0.1, 0.2, 0.3), True]
)
def test_invalid_spacing_is_rejected(value):
    with pytest.raises((ValueError, TypeError)):
        MaxSpacing(value)


def test_invalid_schemes_dimensions_and_extents_are_rejected():
    space = KSpace.propagating(2 * np.pi)
    with pytest.raises(ValueError, match="scheme"):
        resolve_sampling_grid(space.spectrum_domain, "random", PointCounts(5))
    for lengths in ([0, 1], [np.nan], [1, 2, 3]):
        with pytest.raises(ValueError, match="extents"):
            resolve_point_counts(lengths, PointCounts(5))
