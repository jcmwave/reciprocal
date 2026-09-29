import numpy as np
import pytest

from reciprocal import SamplingResult
from reciprocal.cells import PolygonDomain, SamplingResult as CellSamplingResult
from reciprocal.spectrum import KSampling


def _domain():
    return PolygonDomain(
        np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    )


def test_every_public_sampling_name_is_the_same_contract():
    assert CellSamplingResult is SamplingResult
    assert KSampling is SamplingResult


def test_contract_weights_ids_integration_and_immutability():
    domain = _domain()
    source = np.array([[-0.5, 0.0], [0.5, 0.0]])
    result = SamplingResult(source, [0.25, 0.75], domain.area, domain)

    assert result.integrate(np.ones(2)) == pytest.approx(domain.area)
    assert result.average([2.0, 4.0]) == pytest.approx(3.5)
    np.testing.assert_allclose(result.physical_weights, [1.0, 3.0])
    np.testing.assert_array_equal(result.point_ids, [0, 1])
    source[:] = 0.0
    np.testing.assert_allclose(result.points, [[-0.5, 0.0], [0.5, 0.0]])
    assert not result.points.flags.writeable
    assert not result.normalized_weights.flags.writeable


def test_coordinate_domain_can_differ_from_integration_domain():
    full = _domain()
    half = PolygonDomain(
        np.array([[0.0, -1.0], [1.0, -1.0], [1.0, 1.0], [0.0, 1.0]])
    )
    result = SamplingResult(
        [[0.25, 0.0]],
        [1.0],
        full.area,
        full,
        metadata={"coordinate_domain": half},
    )
    assert result.integrate([1.0]) == pytest.approx(full.area)
