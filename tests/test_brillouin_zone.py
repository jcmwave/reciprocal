import numpy as np
import pytest

from reciprocal.brillouin_zone import BrillouinZoneSampler
from reciprocal.cells.geometry import contains_points
from reciprocal.lattice import Lattice, LatticeVectors


def _rotate(vector, angle):
    radians = np.radians(angle)
    rotation = np.array(
        [[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]]
    )
    return np.asarray(vector)[:2] @ rotation.T


def _assert_same_vertices(actual, expected):
    assert len(actual) == len(expected)
    for vertex in expected:
        assert np.any(np.all(np.isclose(actual[:, :2], vertex, rtol=1e-10, atol=1e-12), axis=1))


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


def test_centered_rectangular_h1_uses_the_canonical_ibz_representative():
    zone = Lattice.from_lengths_angle(500.0, 500.0, 70.0).make_reciprocal().brillouin_zone
    h1 = zone.special_points["H1"].cartesian

    assert contains_points(zone.irreducible_domain, [h1])[0]
    assert h1[0] > 0.0
    assert np.any(
        np.all(np.isclose(zone.vertices[:, :2], h1[:2], rtol=1e-10, atol=1e-12), axis=1)
    )


def test_oblique_h1_uses_the_canonical_ibz_representative():
    zone = Lattice.from_lengths_angle(1000.0, 500.0, 75.0).make_reciprocal().brillouin_zone
    h1 = zone.special_points["H1"].cartesian

    assert contains_points(zone.irreducible_domain, [h1])[0]
    assert h1[0] > 0.0
    assert np.any(
        np.all(np.isclose(zone.vertices[:, :2], h1[:2], rtol=1e-10, atol=1e-12), axis=1)
    )


@pytest.mark.parametrize("rotation", [0.0, 31.0])
def test_square_ibz_is_oriented_to_the_first_direct_lattice_vector(rotation):
    first = _rotate([1.0, 0.0], rotation)
    second = _rotate([0.0, 1.0], rotation)
    reciprocal = Lattice.from_vectors(first, second).make_reciprocal()
    zone = reciprocal.brillouin_zone
    reciprocal_basis = reciprocal.vectors.basis[:, :2]
    expected_x = 0.5 * reciprocal_basis[0]
    expected_m = 0.5 * (reciprocal_basis[0] + reciprocal_basis[1])

    np.testing.assert_allclose(zone.special_points["X"].cartesian[:2], expected_x)
    np.testing.assert_allclose(zone.special_points["M"].cartesian[:2], expected_m)
    assert np.dot(zone.special_points["X"].cartesian[:2], first) > 0.0
    _assert_same_vertices(zone.irreducible_domain.vertices, [[0.0, 0.0], expected_x, expected_m])


@pytest.mark.parametrize("angle", [60.0, 120.0])
@pytest.mark.parametrize("rotation", [0.0, 27.0])
def test_hexagonal_ibz_places_k_along_the_first_direct_lattice_vector(angle, rotation):
    base = LatticeVectors.from_lengths_angle(1.0, 1.0, angle)
    first = _rotate(base.vec1, rotation)
    second = _rotate(base.vec2, rotation)
    direct = Lattice.from_vectors(first, second)
    zone = direct.make_reciprocal().brillouin_zone
    k_point = zone.special_points["K"].cartesian[:2]
    first_direction = first / np.linalg.norm(first)

    cross = first_direction[0] * k_point[1] - first_direction[1] * k_point[0]
    assert cross == pytest.approx(0.0, abs=1e-12)
    assert np.dot(k_point, first_direction) > 0.0
    if rotation == 0.0:
        assert k_point[1] == pytest.approx(0.0, abs=1e-12)
        assert k_point[0] > 0.0
    _assert_same_vertices(
        zone.irreducible_domain.vertices,
        [[0.0, 0.0], k_point, zone.special_points["M"].cartesian[:2]],
    )
