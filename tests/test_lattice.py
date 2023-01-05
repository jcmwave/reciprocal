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
