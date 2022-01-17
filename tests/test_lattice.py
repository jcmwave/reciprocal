import pytest
from reciprocal import lattice
import numpy as np

@pytest.fixture
def fail_state_msg():
    fail_message = ("inputs must either be two vector lengths and "+
                    "lattice angle or the lattice vectors")
    yield fail_message

def test_lattice_vectors_lengths():
    lat = lattice.LatticeVectors(1000, 1000, 90.)
    tol = 1e-6
    assert abs(lat.vec1[0]-1000.) < tol
    assert abs(lat.vec1[1]-0.) < tol
    assert abs(lat.vec2[0]-0.) < tol
    assert abs(lat.vec2[1]-1000.) < tol

def test_lattice_vectors_vectors():
    vec1 = np.array([500., 250.])
    vec2 = np.array([500., -250.])
    lat = lattice.LatticeVectors(vector1=vec1,
                                 vector2=vec2)
    tol = 1e-6
    assert abs(lat.vec1[0]-500.) < tol
    assert abs(lat.vec1[1]-250.) < tol
    assert abs(lat.vec2[0]-500.) < tol
    assert abs(lat.vec2[1]+250.) < tol

def test_lattice_vectors_angle():
    a = 1000.
    b = 1000.
    angle = 45.
    lat = lattice.LatticeVectors(a, b, angle)
    tol = 1e-6
    v1 = a*np.array([1., 0.])
    v2 = b*np.array([np.cos(np.radians(angle)),
                     np.sin(np.radians(angle))])
    assert abs(lat.vec1[0]-v1[0]) < tol
    assert abs(lat.vec1[1]-v1[1]) < tol
    assert abs(lat.vec2[0]-v2[0]) < tol
    assert abs(lat.vec2[1]-v2[1]) < tol

def test_lattice_vectors_failstate1(fail_state_msg):
    try:
        lat = lattice.LatticeVectors(1000, 1000)
    except ValueError as e:
        if str(e) != fail_state_msg:
            raise e

def test_lattice_vectors_failstate2(fail_state_msg):
    try:
        lat = lattice.LatticeVectors(length1=1000, angle=45)
    except ValueError as e:
        if str(e) != fail_state_msg:
            raise e

def test_lattice_vectors_failstate3(fail_state_msg):
    try:
        lat = lattice.LatticeVectors(vector1=[100., 0.],
                                     vector2=[0., 100.],
                                     angle=45)
    except ValueError as e:
        if str(e) != fail_state_msg:
            raise e

def test_reciprocal_vectors():
    a = 1000.
    b = 1000.
    angle = 60.
    lat = lattice.LatticeVectors(a, b, angle)
    rlat = lat.reciprocal_vectors()
    tol = 1e-6
    assert abs(rlat.vec1[0]-0.00628319) < tol
    assert abs(rlat.vec1[1]-(-0.0036276)) < tol
    assert abs(rlat.vec2[0]-0.) < tol
    assert abs(rlat.vec2[1]-0.0072552) < tol

def test_lattice():
    lat = lattice.Lattice(1000., 1000., 60.)
