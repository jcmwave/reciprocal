"""Deterministic property tests for the public numerical invariants."""

import numpy as np
import pytest

from reciprocal import KSpace, Lattice, LatticeVectors, Symmetry

hypothesis = pytest.importorskip("hypothesis")
given = hypothesis.given
settings = hypothesis.settings
st = hypothesis.strategies


FINITE_LENGTHS = st.floats(
    min_value=1e-9,
    max_value=1e9,
    allow_nan=False,
    allow_infinity=False,
)
VALID_ANGLES = st.floats(
    min_value=1.0,
    max_value=179.0,
    allow_nan=False,
    allow_infinity=False,
)
PROPERTY_SETTINGS = settings(max_examples=30, deadline=None, derandomize=True)


@PROPERTY_SETTINGS
@given(
    scale=FINITE_LENGTHS,
    length_ratio=st.floats(0.1, 10.0, allow_nan=False, allow_infinity=False),
    angle=VALID_ANGLES,
)
def test_reciprocal_basis_identity_for_random_scales(scale, length_ratio, angle):
    direct = LatticeVectors.from_lengths_angle(scale, scale * length_ratio, angle)
    reciprocal = direct.reciprocal_vectors()
    products = np.array(
        [
            [np.dot(direct.vec1, reciprocal.vec1), np.dot(direct.vec1, reciprocal.vec2)],
            [np.dot(direct.vec2, reciprocal.vec1), np.dot(direct.vec2, reciprocal.vec2)],
        ]
    )
    np.testing.assert_allclose(products, 2 * np.pi * np.eye(2), rtol=2e-10, atol=2e-10)


@PROPERTY_SETTINGS
@given(
    x=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
    y=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
)
def test_dihedral_symmetry_is_closed_and_norm_preserving(x, y):
    symmetry = Symmetry.from_string("D4")
    point = np.array([x, y, 0.0])
    orbit = symmetry.apply_symmetry_operators(point)
    np.testing.assert_allclose(
        np.linalg.norm(orbit, axis=1), np.linalg.norm(point), rtol=1e-12, atol=1e-9
    )

    reapplied = symmetry.apply_symmetry_operators(orbit)
    for transformed in reapplied:
        close = np.isclose(orbit, transformed, rtol=1e-10, atol=1e-9)
        assert np.any(np.all(close, axis=1))


@PROPERTY_SETTINGS
@given(radius=st.floats(1e-6, 1e6, allow_nan=False, allow_infinity=False))
def test_regular_sampling_stays_inside_domain_and_integrates_constant(radius):
    space = KSpace(wavelength=2 * np.pi / radius, fermi_radius=radius)
    points, weights = space.regular_sampler.sample(constraint={"type": "n_points", "value": 8})
    norms = np.linalg.norm(points.k[:, :2], axis=1)
    assert np.all(norms <= radius * (1 + 1e-12))
    assert np.sum(weights) == pytest.approx(1.0, rel=0.2)


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_bravais_classification_is_scale_invariant(scale):
    square = Lattice(LatticeVectors.from_lengths_angle(scale, scale, 90.0))
    rectangle = Lattice(LatticeVectors.from_lengths_angle(scale, 2 * scale, 90.0))
    assert square.bravais.name == "SQUARE"
    assert rectangle.bravais.name == "RECTANGULAR"
