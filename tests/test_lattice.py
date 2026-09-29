import pytest
from reciprocal import BravaisLattice, lattice
from reciprocal.lattice import classify_bravais
import numpy as np


def test_bravais_lattice_uses_standard_two_dimensional_names():
    assert {member.name for member in BravaisLattice} == {
        "OBLIQUE",
        "RECTANGULAR",
        "CENTERED_RECTANGULAR",
        "SQUARE",
        "HEXAGONAL",
    }


def test_lattice_vectors_lengths():
    lat = lattice.LatticeVectors.from_lengths_angle(1000, 1000, 90.0)
    tol = 1e-6
    assert abs(lat.vec1[0] - 1000.0) < tol
    assert abs(lat.vec1[1] - 0.0) < tol
    assert abs(lat.vec2[0] - 0.0) < tol
    assert abs(lat.vec2[1] - 1000.0) < tol


def test_lattice_vectors_vectors():
    vec1 = np.array([500.0, 250.0])
    vec2 = np.array([500.0, -250.0])
    lat = lattice.LatticeVectors(vec1, vec2)
    tol = 1e-6
    assert abs(lat.vec1[0] - 500.0) < tol
    assert abs(lat.vec1[1] - 250.0) < tol
    assert abs(lat.vec2[0] - 500.0) < tol
    assert abs(lat.vec2[1] + 250.0) < tol


def test_lattice_vectors_are_genuinely_immutable():
    vectors = lattice.LatticeVectors([1.0, 0.0], [0.0, 1.0])

    with pytest.raises(ValueError):
        vectors.basis.setflags(write=True)
    with pytest.raises(ValueError):
        vectors.vec1.setflags(write=True)
    with pytest.raises(AttributeError):
        vectors.vec1 = vectors.vec2


def test_lattice_vectors_angle():
    a = 1000.0
    b = 1000.0
    angle = 45.0
    lat = lattice.LatticeVectors.from_lengths_angle(a, b, angle)
    tol = 1e-6
    v1 = a * np.array([1.0, 0.0])
    v2 = b * np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
    assert abs(lat.vec1[0] - v1[0]) < tol
    assert abs(lat.vec1[1] - v1[1]) < tol
    assert abs(lat.vec2[0] - v2[0]) < tol
    assert abs(lat.vec2[1] - v2[1]) < tol


def test_reciprocal_vectors():
    a = 1000.0
    b = 1000.0
    angle = 60.0
    lat = lattice.LatticeVectors.from_lengths_angle(a, b, angle)
    rlat = lat.reciprocal_vectors()
    tol = 1e-6
    assert abs(rlat.vec1[0] - 0.00628319) < tol
    assert abs(rlat.vec1[1] - (-0.0036276)) < tol
    assert abs(rlat.vec2[0] - 0.0) < tol
    assert abs(rlat.vec2[1] - 0.0072552) < tol


@pytest.mark.parametrize("length1,length2,angle", [(2.0, 3.0, 40.0), (1.0, 1.0, 90.0)])
def test_reciprocal_vector_identity(length1, length2, angle):
    """Reciprocal vectors use radians per length: a_i dot b_j = 2*pi delta_ij."""
    direct = lattice.LatticeVectors.from_lengths_angle(length1, length2, angle)
    reciprocal = direct.reciprocal_vectors()

    products = np.array(
        [
            [np.dot(direct.vec1, reciprocal.vec1), np.dot(direct.vec1, reciprocal.vec2)],
            [np.dot(direct.vec2, reciprocal.vec1), np.dot(direct.vec2, reciprocal.vec2)],
        ]
    )
    np.testing.assert_allclose(products, 2 * np.pi * np.eye(2), atol=1e-10)


@pytest.mark.parametrize(
    "length1,length2,angle,message",
    [
        (0.0, 1.0, 90.0, "positive"),
        (1.0, 1.0, 0.0, "strictly between"),
        (1.0, 1.0, 180.0, "strictly between"),
    ],
)
def test_invalid_lattice_parameters(length1, length2, angle, message):
    with pytest.raises(ValueError, match=message):
        lattice.LatticeVectors.from_lengths_angle(length1, length2, angle)


def test_parallel_lattice_vectors_are_rejected():
    with pytest.raises(ValueError, match="linearly independent"):
        lattice.LatticeVectors(np.array([1.0, 0.0]), np.array([2.0, 0.0]))


def test_lattice():
    a = 1000.0
    b = 1000.0
    angle = 60.0
    lat_vectors = lattice.LatticeVectors.from_lengths_angle(a, b, angle)
    lat = lattice.Lattice(lat_vectors)


def test_lattice_identity_is_immutable_and_derived_cells_remain_valid():
    lat = lattice.Lattice(lattice.LatticeVectors.from_lengths_angle(1.0, 1.0, 90.0))
    primitive = lat.primitive_cell

    with pytest.raises(AttributeError, match="immutable"):
        lat.vectors = lattice.LatticeVectors.from_lengths_angle(2.0, 2.0, 90.0)
    with pytest.raises(AttributeError, match="immutable"):
        lat.bravais = BravaisLattice.OBLIQUE
    assert lat.primitive_cell is primitive
    assert primitive.area == pytest.approx(1.0)


def test_legacy_unit_cell_is_lazy_and_deprecated():
    lat = lattice.Lattice.from_lengths_angle(1.0, 1.0, 90.0)
    assert "unit_cell" not in lat.__dict__

    with pytest.warns(DeprecationWarning, match="primitive_cell"):
        legacy = lat.unit_cell
    assert lat.unit_cell is legacy


def test_lattice_from_keywords():
    lat_vec_args = {}
    lat_vec_args["length1"] = 1000.0
    lat_vec_args["length2"] = 1000.0
    lat_vec_args["angle"] = 60.0
    lat = lattice.Lattice.from_lat_vec_args(**lat_vec_args)


def test_explicit_lattice_constructors_and_strict_compatibility_arguments():
    from_vectors = lattice.Lattice.from_vectors([1.0, 0.0], [0.0, 2.0])
    from_parameters = lattice.Lattice.from_lengths_angle(1.0, 2.0, 90.0)

    assert from_vectors == from_parameters
    with pytest.raises(ValueError, match="expected exactly"):
        lattice.Lattice.from_lat_vec_args(length1=1.0, length2=2.0, angle=90.0, unexpected=True)


def test_reciprocal_of_reciprocal_is_rejected():
    direct = lattice.Lattice.from_lengths_angle(1.0, 2.0, 90.0)
    reciprocal = direct.make_reciprocal()

    assert reciprocal.lattice_type == "reciprocal"
    with pytest.raises(ValueError, match="real-space"):
        reciprocal.make_reciprocal()


def test_classification_is_available_as_a_pure_function():
    vectors = lattice.LatticeVectors.from_lengths_angle(1.0, 1.0, 60.0)
    assert classify_bravais(vectors) is BravaisLattice.HEXAGONAL


def test_shortest_vectors_use_a_reduced_basis():
    vectors = lattice.LatticeVectors([1.0, 0.0], [100.0, 1.0])
    first, second = vectors.get_shortest_vectors()

    np.testing.assert_allclose(np.linalg.norm(first), 1.0)
    np.testing.assert_allclose(np.linalg.norm(second), 1.0)
    np.testing.assert_allclose(abs(np.cross(first, second)[2]), 1.0)


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_distance_groups_are_scale_aware(scale):
    lat = lattice.Lattice.from_lengths_angle(scale, 2.0 * scale, 90.0)
    groups, distances = lat.orders_by_distance(2)

    assert len(groups) == 7
    np.testing.assert_allclose(
        distances[:4] / scale,
        [1.0, 2.0, np.sqrt(5.0), np.sqrt(8.0)],
        rtol=1e-12,
    )


def test_translation_shells_are_complete_for_a_poor_input_basis():
    lat = lattice.Lattice.from_vectors([1.0, 0.0], [100.0, 1.0])
    groups, distances = lat.translation_shells(2)

    assert len(groups[0]) == 4
    assert len(groups[1]) == 4
    np.testing.assert_allclose(distances, [1.0, np.sqrt(2.0)])
    translated = groups[0] @ lat.vectors.basis[:, :2]
    actual = {tuple(np.round(point, 12)) for point in translated}
    assert actual == {(-1.0, 0.0), (0.0, -1.0), (0.0, 1.0), (1.0, 0.0)}


def test_centered_rectangular_classification_survives_reciprocal_reduction():
    direct = lattice.Lattice.from_lengths_angle(1.0, 1.0, 20.0)

    assert direct.bravais is BravaisLattice.CENTERED_RECTANGULAR
    assert direct.make_reciprocal().bravais is BravaisLattice.CENTERED_RECTANGULAR


@pytest.mark.parametrize(
    "method_name", ["enumerate_orders", "orders_by_distance", "translation_shells"]
)
def test_order_enumeration_validates_integer_bounds(method_name):
    lat = lattice.Lattice.from_lengths_angle(1.0, 1.0, 90.0)
    method = getattr(lat, method_name)

    with pytest.raises(TypeError, match="integer"):
        method(1.5)
    with pytest.raises(ValueError, match="non-negative"):
        method(-1)


@pytest.mark.parametrize(
    "vector1,vector2,expected_vertices",
    [
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 4),
        ([1.0, 0.0, 0.0], [0.5, np.sqrt(3.0) / 2.0, 0.0], 6),
        ([1.0, 0.0, 0.0], [0.95, 0.2, 0.0], 6),
    ],
)
def test_wigner_seitz_cell_uses_lattice_half_planes(vector1, vector2, expected_vertices):
    vectors = lattice.LatticeVectors(vector1, vector2)
    reciprocal_lattice = lattice.Lattice(vectors, lattice_type="reciprocal")
    vertices = reciprocal_lattice.brillouin_zone.vertices

    assert vertices.shape == (expected_vertices, 3)
    np.testing.assert_allclose(vertices[:, 2], 0.0)

    x = vertices[:, 0]
    y = vertices[:, 1]
    polygon_area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    lattice_area = abs(np.cross(np.asarray(vector1), np.asarray(vector2))[2])
    assert polygon_area == pytest.approx(lattice_area, rel=1e-11)

    for first_order in range(-8, 9):
        for second_order in range(-8, 9):
            if first_order == 0 and second_order == 0:
                continue
            translation = first_order * np.asarray(vector1[:2]) + second_order * np.asarray(
                vector2[:2]
            )
            offset = 0.5 * np.dot(translation, translation)
            assert np.all(vertices[:, :2] @ translation <= offset + 1e-10)
