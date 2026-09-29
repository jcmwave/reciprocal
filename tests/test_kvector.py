import numpy as np
import pytest
import reciprocal

from reciprocal.kvector import BlochFamily, BlochVector, KVector, KVectorGroup


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


@pytest.mark.parametrize("direction", [-1, 1])
def test_theta_is_unsigned_and_direction_round_trips(direction):
    vector = KVector.from_angles(
        wavelength=2 * np.pi,
        theta=60.0,
        phi=30.0,
        n=2.0,
        normal=direction,
    )
    rebuilt = KVector.from_cartesian(vector.wavelength, *vector.k)

    assert rebuilt.theta == pytest.approx(60.0)
    assert rebuilt.normal_ == direction
    np.testing.assert_allclose(rebuilt.k, vector.k)


def test_angles_reject_obtuse_theta():
    with pytest.raises(ValueError, match=r"\[0, 90\]"):
        KVector.from_angles(1.0, n=1.0, theta=120.0, phi=0.0, normal=-1)


@pytest.mark.parametrize("direction", [-1, 1])
def test_evanescent_transverse_vector(direction):
    vector = KVector.from_transverse(
        wavelength=2 * np.pi, n=1.0, kx=2.0, ky=0.0, normal=direction
    )

    assert vector.is_evanescent
    assert np.isnan(vector.theta)
    assert vector.kz == pytest.approx(direction * 1j * np.sqrt(3.0))
    rebuilt = KVector.from_cartesian(vector.wavelength, *vector.k)
    assert rebuilt.n == pytest.approx(1.0)
    assert rebuilt.normal_ == direction


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


def test_group_keeps_weights_and_supports_evanescent_rows():
    group = KVectorGroup.from_transverse(
        2 * np.pi,
        n=[1.0, 1.0],
        kx=[0.0, 2.0],
        ky=[0.0, 0.0],
        normal=[1, -1],
        weighting=[0.25, 0.75],
    )

    np.testing.assert_allclose(group.weighting, [0.25, 0.75])
    np.testing.assert_array_equal(group.is_evanescent, [False, True])
    assert np.isnan(group.theta[1])


def test_invalid_kvector_inputs():
    with pytest.raises(ValueError, match="wavelength"):
        KVector(0.0, kx=0.0, ky=0.0, kz=1.0)
    with pytest.raises(ValueError, match="enough information"):
        KVector(1.0, kx=0.0)
    with pytest.raises(ValueError, match="complex refractive"):
        KVector(2 * np.pi, theta=0.0, phi=0.0, n=1.0 + 0.1j, normal=1)


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


def test_bloch_family_retains_and_validates_generation_metadata():
    group = KVectorGroup.from_transverse(
        2 * np.pi,
        n=[3.0, 3.0],
        kx=[0.25, 2.25],
        ky=[0.5, 0.5],
        normal=[1, 1],
    )
    family = BlochFamily.from_kvector_group(
        group,
        representative=[0.25, 0.5],
        reciprocal_basis=[[2.0, 0.0], [0.0, 2.0]],
        order1=[0, 1],
        order2=[0, 0],
    )

    member = family.slice(1)
    assert isinstance(member, BlochVector)
    assert member.order == (1, 0)
    combined = family + family
    assert isinstance(combined, BlochFamily)
    np.testing.assert_array_equal(combined.orders, [[0, 0], [1, 0], [0, 0], [1, 0]])

    with pytest.raises(ValueError, match="representative"):
        BlochFamily.from_kvector_group(
            group,
            representative=[0.0, 0.0],
            reciprocal_basis=[[2.0, 0.0], [0.0, 2.0]],
            order1=[0, 1],
            order2=[0, 0],
        )
