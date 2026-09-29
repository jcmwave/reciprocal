import itertools

import numpy as np
import pytest

from reciprocal.bravais import BravaisLattice
from reciprocal.lattice import LatticeVectors
from reciprocal.symmetry import (
    little_group,
    point_group,
    point_orbit,
    point_orbit_with_operations,
)


@pytest.mark.parametrize(
    "angle,bravais,order",
    [
        (73.0, BravaisLattice.CENTERED_RECTANGULAR, 4),
        (90.0, BravaisLattice.SQUARE, 8),
        (120.0, BravaisLattice.HEXAGONAL, 12),
    ],
)
def test_point_groups_are_closed_and_obey_orbit_stabilizer(angle, bravais, order):
    vectors = LatticeVectors.from_lengths_angle(1.0, 1.0, angle)
    basis = np.vstack((vectors.vec1, vectors.vec2))
    group = point_group(basis, bravais)

    assert len(group) == order
    for left, right in itertools.product(group, repeat=2):
        product = left.fractional @ right.fractional
        assert any(np.array_equal(product, operation.fractional) for operation in group)

    point = 0.5 * vectors.vec1
    stabilizer = little_group(point, basis, group)
    orbit = point_orbit(point, basis, group)
    assert len(stabilizer) * len(orbit) == len(group)


def test_point_orbit_retains_all_generating_operations():
    vectors = LatticeVectors.from_lengths_angle(1.0, 1.0, 90.0)
    basis = np.vstack((vectors.vec1, vectors.vec2))
    group = point_group(basis, BravaisLattice.SQUARE)
    orbit = point_orbit_with_operations(np.zeros(3), basis, group)

    assert len(orbit.members) == 1
    assert len(orbit.members[0].generating_operations) == len(group)
    np.testing.assert_allclose(orbit.points, [[0.0, 0.0, 0.0]])
