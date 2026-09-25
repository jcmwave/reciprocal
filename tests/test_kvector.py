import numpy as np
import pytest
import reciprocal

from reciprocal.kvector import KVector, KVectorGroup


def test_cartesian_polar_round_trip():
    vector = KVector(
        wavelength=2 * np.pi,
        theta=60.0,
        phi=30.0,
        n=2.0,
        normal=1,
    )

    rebuilt = KVector(
        wavelength=vector.wavelength,
        kx=vector.kx,
        ky=vector.ky,
        kz=vector.kz,
    )

    np.testing.assert_allclose(rebuilt.k, vector.k, rtol=1e-12, atol=1e-12)
    assert rebuilt.theta == pytest.approx(60.0, abs=1e-12)
    assert rebuilt.phi == pytest.approx(30.0, abs=1e-12)
    assert rebuilt.n == pytest.approx(2.0, rel=1e-12)


def test_group_round_trip_and_norms():
    group = KVectorGroup(
        wavelength=2 * np.pi,
        n_rows=3,
        theta=np.array([0.0, 30.0, 60.0]),
        phi=np.array([0.0, 90.0, -45.0]),
        n=np.ones(3),
        normal=np.ones(3),
    )

    np.testing.assert_allclose(np.linalg.norm(group.k, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(group.theta, [0.0, 30.0, 60.0], atol=1e-12)
    np.testing.assert_allclose(group.phi[1:], [90.0, -45.0], atol=1e-12)


def test_invalid_kvector_inputs():
    with pytest.raises(ValueError, match="wavelength"):
        KVector(0.0, kx=0.0, ky=0.0, kz=1.0)
    with pytest.raises(ValueError, match="enough information"):
        KVector(1.0, kx=0.0)
    with pytest.raises(ValueError, match="exceed"):
        KVector(2 * np.pi, kx=2.0, ky=0.0, n=1.0, normal=1)


def test_groups_with_different_wavelengths_cannot_be_combined():
    first = KVectorGroup(1.0, 1, kx=[0.0], ky=[0.0], kz=[1.0])
    second = KVectorGroup(2.0, 1, kx=[0.0], ky=[0.0], kz=[1.0])
    with pytest.raises(ValueError, match="different wavelengths"):
        first + second


@pytest.mark.parametrize(
    "order,expected",
    [
        ("ascending", [-0.5, 0.0, 0.75]),
        ("descending", [0.75, 0.0, -0.5]),
    ],
)
def test_group_sort_respects_direction(order, expected):
    group = KVectorGroup(
        2 * np.pi,
        3,
        kx=[-0.5, 0.75, 0.0],
        ky=[0.0, 0.0, 0.0],
        n=[1.0, 1.0, 1.0],
        normal=[1.0, 1.0, 1.0],
    )

    group.sort("kx", order=order)

    np.testing.assert_allclose(group.kx, expected)


def test_camel_case_method_has_deprecation_path():
    vector = KVector(2 * np.pi, kx=0.0, ky=0.0, kz=1.0)
    assert vector.get_n_from_k() == pytest.approx(1.0)
    with pytest.warns(DeprecationWarning, match="get_n_from_k"):
        assert vector.getNFromK() == pytest.approx(1.0)
