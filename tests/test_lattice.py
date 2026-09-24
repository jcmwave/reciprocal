import pytest
from reciprocal import lattice
import numpy as np


def test_lattice_vectors_lengths():
    lat = lattice.LatticeVectors.from_lengths_angle(1000, 1000, 90.)
    tol = 1e-6
    assert abs(lat.vec1[0]-1000.) < tol
    assert abs(lat.vec1[1]-0.) < tol
    assert abs(lat.vec2[0]-0.) < tol
    assert abs(lat.vec2[1]-1000.) < tol

def test_lattice_vectors_vectors():
    vec1 = np.array([500., 250.])
    vec2 = np.array([500., -250.])
    lat = lattice.LatticeVectors(vec1, vec2)
    tol = 1e-6
    assert abs(lat.vec1[0]-500.) < tol
    assert abs(lat.vec1[1]-250.) < tol
    assert abs(lat.vec2[0]-500.) < tol
    assert abs(lat.vec2[1]+250.) < tol

def test_lattice_vectors_angle():
    a = 1000.
    b = 1000.
    angle = 45.
    lat = lattice.LatticeVectors.from_lengths_angle(a, b, angle)
    tol = 1e-6
    v1 = a*np.array([1., 0.])
    v2 = b*np.array([np.cos(np.radians(angle)),
                     np.sin(np.radians(angle))])
    assert abs(lat.vec1[0]-v1[0]) < tol
    assert abs(lat.vec1[1]-v1[1]) < tol
    assert abs(lat.vec2[0]-v2[0]) < tol
    assert abs(lat.vec2[1]-v2[1]) < tol


def test_reciprocal_vectors():
    a = 1000.
    b = 1000.
    angle = 60.
    lat = lattice.LatticeVectors.from_lengths_angle(a, b, angle)
    rlat = lat.reciprocal_vectors()
    tol = 1e-6
    assert abs(rlat.vec1[0]-0.00628319) < tol
    assert abs(rlat.vec1[1]-(-0.0036276)) < tol
    assert abs(rlat.vec2[0]-0.) < tol
    assert abs(rlat.vec2[1]-0.0072552) < tol


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
    a = 1000.
    b = 1000.
    angle = 60.
    lat_vectors = lattice.LatticeVectors.from_lengths_angle(a, b, angle)
    lat = lattice.Lattice(lat_vectors)

def test_lattice_from_keywords():
    lat_vec_args = {}
    lat_vec_args['length1'] = 1000.
    lat_vec_args['length2'] = 1000.
    lat_vec_args['angle'] = 60.    
    lat = lattice.Lattice.from_lat_vec_args(**lat_vec_args)
