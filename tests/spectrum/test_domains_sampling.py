import numpy as np
import pytest

from reciprocal.spectrum import (
    CartesianGrid,
    EvanescentDisk,
    NonPeriodicSampler,
    PolarGrid,
    PropagationDisk,
    PupilDomain,
)


def test_polar_pupil_sampling_has_physical_weights_and_kz_branch():
    domain = PupilDomain(2.0, 0.5)
    sampling = NonPeriodicSampler().sample(domain, PolarGrid(4, 12)).with_kz(1.0)

    assert len(sampling.points) == 48
    assert np.sum(sampling.physical_weights) == pytest.approx(domain.area)
    assert sampling.integrate(np.ones(len(sampling.points))) == pytest.approx(domain.area)
    assert np.all(np.imag(sampling.kz[np.linalg.norm(sampling.points, axis=1) > 1.0]) > 0.0)


@pytest.mark.parametrize("shape", [(8, 7), (9, 9), (1, 5)])
@pytest.mark.parametrize("center", [(0.0, 0.0), (0.3, -0.4)])
def test_cartesian_sampling_masks_fixed_grid_and_assigns_equal_weights(shape, center):
    domain = PropagationDisk(1.0, center)
    sampling = NonPeriodicSampler().sample(domain, CartesianGrid(shape))
    x0, x1, y0, y1 = domain.bounds()
    x_edges = np.linspace(x0, x1, shape[0] + 1)
    y_edges = np.linspace(y0, y1, shape[1] + 1)
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    y = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    candidates = np.column_stack((xx.ravel(), yy.ravel()))
    expected = candidates[domain.contains(candidates)]

    np.testing.assert_array_equal(sampling.points, expected)
    assert np.all(domain.contains(sampling.points))
    np.testing.assert_allclose(np.diff(x), 2.0 / shape[0])
    np.testing.assert_allclose(np.diff(y), 2.0 / shape[1])
    np.testing.assert_array_equal(
        sampling.normalized_weights, np.full(len(expected), 1.0 / len(expected))
    )
    np.testing.assert_allclose(sampling.physical_weights, domain.area / len(expected))
    assert np.sum(sampling.physical_weights) == pytest.approx(np.pi)
    assert sampling.integrate(np.ones(len(expected))) == pytest.approx(domain.area)


def test_cartesian_grid_does_not_move_a_point_into_an_annular_domain():
    with pytest.raises(ValueError, match="no points"):
        NonPeriodicSampler().sample(PupilDomain(1.0, 0.5), CartesianGrid((1, 1)))


def test_evanescent_domain_requires_finite_extension():
    with pytest.raises(ValueError, match="exceed"):
        EvanescentDisk(1.0, 1.0)
