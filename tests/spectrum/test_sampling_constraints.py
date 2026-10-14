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


