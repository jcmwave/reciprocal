import numpy as np
import pytest

from reciprocal import Lattice
from reciprocal.reciprocal_mesh import (
    GridCentering,
    MonkhorstPackGrid,
    reduce_monkhorst_pack,
    sample_monkhorst_pack,
)


def _zone(length1=1.0, length2=1.0, angle=90.0):
    return Lattice.from_lengths_angle(length1, length2, angle).make_reciprocal().brillouin_zone


def test_reference_fractional_meshes_and_constant_integration():
    zone = _zone()
    conventional = sample_monkhorst_pack(zone, MonkhorstPackGrid((2, 2)))
    metadata = conventional.metadata["monkhorst_pack"]
    np.testing.assert_allclose(
        metadata.canonical_keys,
        [[-0.25, -0.25], [-0.25, 0.25], [0.25, -0.25], [0.25, 0.25]],
    )
    assert conventional.integrate(np.ones(4)) == pytest.approx(zone.area)

    gamma = sample_monkhorst_pack(
        zone, MonkhorstPackGrid((3, 3), GridCentering.GAMMA)
    )
    assert np.any(np.all(np.isclose(gamma.metadata["monkhorst_pack"].canonical_keys, 0.0), axis=1))


@pytest.mark.parametrize(
    ("length1", "length2", "angle"),
    [(1.0, 1.0, 90.0), (1.5, 1.0, 90.0), (1.0, 1.0, 70.0), (1.0, 1.0, 60.0), (1.3, 1.0, 73.0)],
)
def test_irreducible_mesh_degeneracies_reconstruct_full_mesh(length1, length2, angle):
    zone = _zone(length1, length2, angle)
    reduction = reduce_monkhorst_pack(zone, MonkhorstPackGrid((5, 4)))
    assert np.sum(reduction.degeneracies) == 20
    assert len(reduction.full_to_irreducible) == 20
    np.testing.assert_array_equal(
        np.bincount(reduction.full_to_irreducible), reduction.degeneracies
    )
    assert reduction.irreducible.integrate(
        np.ones(len(reduction.irreducible.points))
    ) == pytest.approx(zone.area)


def test_asymmetric_shift_uses_only_mesh_preserving_operations():
    zone = _zone()
    reduction = reduce_monkhorst_pack(
        zone,
        MonkhorstPackGrid((4, 3), shift=(0.25, 0.0)),
    )
    assert 1 <= len(reduction.operations) < len(zone.point_group)


def test_grid_validation_and_shift_canonicalization():
    assert MonkhorstPackGrid((2, 3), shift=(-0.5, 1.5)).shift == (0.5, 0.5)
    with pytest.raises(ValueError, match="positive integers"):
        MonkhorstPackGrid((0, 2))
    with pytest.raises(ValueError, match="centering"):
        MonkhorstPackGrid((2, 2), "invalid")
