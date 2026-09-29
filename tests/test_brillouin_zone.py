import numpy as np
import pytest

from reciprocal.brillouin_zone import BrillouinZoneSampler
from reciprocal.cells.geometry import contains_points
from reciprocal.lattice import Lattice, LatticeVectors


@pytest.mark.parametrize(
    "parameters,labels,order",
    [
        ((1.0, 1.3, 70.0), {"Γ", "X", "Y", "C", "H1"}, 2),
        ((1.0, 2.0, 90.0), {"Γ", "X", "Y", "S"}, 4),
        ((1.0, 1.0, 70.0), {"Γ", "X", "Y", "S", "H1"}, 4),
        ((1.0, 1.0, 90.0), {"Γ", "X", "M"}, 8),
        ((1.0, 1.0, 60.0), {"Γ", "M", "K"}, 12),
    ],
)
def test_brillouin_zone_invariants(parameters, labels, order):
    lattice = Lattice(LatticeVectors.from_lengths_angle(*parameters), "reciprocal")
    zone = lattice.brillouin_zone

    assert lattice.brillouin_zone is zone
    assert set(zone.special_points) == labels
    assert len(zone.point_group) == order
    assert zone.irreducible_domain.area * order == pytest.approx(zone.area)
    for point in zone.special_points.values():
        assert contains_points(zone.cell.domain, [point.cartesian])[0]
        assert len(point.orbit) * len(point.little_group) == order


def test_brillouin_zone_sampler_integrates_a_constant_over_the_full_zone():
    zone = Lattice(LatticeVectors.from_lengths_angle(1.0, 1.0, 60.0), "reciprocal").brillouin_zone
    sampler = BrillouinZoneSampler()

    for result in (sampler.sample_full(zone), sampler.sample_irreducible(zone)):
        assert np.sum(result.weights) == pytest.approx(1.0)
        integral = np.dot(result.weights, np.ones(len(result.points))) * result.integration_element
        assert integral == pytest.approx(zone.area)
