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


def test_cartesian_sampling_clips_boundary_cells_to_disk():
    domain = PropagationDisk(1.0)
    sampling = NonPeriodicSampler().sample(domain, CartesianGrid((8, 7)))

    assert np.all(domain.contains(sampling.points))
    assert np.sum(sampling.physical_weights) == pytest.approx(np.pi)
    assert np.ptp(sampling.normalized_weights) > 0.0


def test_evanescent_domain_requires_finite_extension():
    with pytest.raises(ValueError, match="exceed"):
        EvanescentDisk(1.0, 1.0)

