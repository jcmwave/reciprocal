import pytest
from reciprocal import lattice
from reciprocal import kspace
import numpy as np

@pytest.fixture
def example_lattice():
    lat_vec_args = {}
    lat_vec_args['length1'] = 1.
    lat_vec_args['length2'] = 1.
    lat_vec_args['angle'] = 90.
    lat = lattice.Lattice.from_lat_vec_args(**lat_vec_args)
    yield lat

def test_kspace():
    wvl = np.pi
    kspace_obj = kspace.KSpace(wvl)

def test_kspace_symmetry():
    wvl = np.pi
    symmetry = 'C4'
    kspace_obj = kspace.KSpace(wvl, symmetry=symmetry, fermi_radius=1.0)

def test_kspace_regular_sampling():
    wvl = np.pi
    k = np.pi*2/wvl
    kspace_obj = kspace.KSpace(wvl, fermi_radius=k)
    vectors, weights = kspace_obj.regular_sampler.sample()
    assert vectors.n_rows == len(weights)
    assert np.all(np.linalg.norm(vectors.k[:, :2], axis=1) <= k + 1e-12)
    assert np.all(weights >= 0.0)
    assert np.sum(weights) == pytest.approx(1.0, rel=0.15)

def test_kspace_with_lattice(example_lattice):
    wvl = np.pi
    symmetry = 'D4'    
    kspace_obj = kspace.KSpace(wvl, symmetry=symmetry, fermi_radius=1.0)
    kspace_obj.apply_lattice(example_lattice)

def test_kspace_periodic_sampling(example_lattice):
    wvl = np.pi
    k = np.pi*2/wvl
    symmetry = 'D4'    
    kspace_obj = kspace.KSpace(wvl, symmetry=symmetry, fermi_radius=k)
    kspace_obj.apply_lattice(example_lattice)
    kvectors = kspace_obj.periodic_sampler.sample()
    assert kvectors.n_rows > 0
    assert np.all(np.linalg.norm(kvectors.k[:, :2], axis=1) <= k + 1e-9)


def test_set_symmetry_after_applying_lattice(example_lattice):
    kspace_obj = kspace.KSpace(np.pi, fermi_radius=2.0)
    kspace_obj.apply_lattice(example_lattice)

    kspace_obj.set_symmetry("D4")

    assert str(kspace_obj.symmetry) == "(SIGMA_H, C4)"
    assert kspace_obj.symmetry_cone is not None


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"wavelength": 0.0}, "wavelength"),
        ({"wavelength": 1.0, "fermi_radius": 0.0}, "fermi_radius"),
    ],
)
def test_invalid_kspace_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        kspace.KSpace(**kwargs)


@pytest.mark.parametrize(
    "constraint,message",
    [
        ({"type": "density", "value": 0.0}, "positive"),
        ({"type": "max_length", "value": -1.0}, "positive"),
        ({"type": "n_points", "value": 2.5}, "integers"),
        ({"type": "mystery", "value": 2}, "invalid"),
    ],
)
def test_invalid_sampling_constraints(constraint, message):
    kspace_obj = kspace.KSpace(np.pi, fermi_radius=2.0)
    with pytest.raises(ValueError, match=message):
        kspace_obj.regular_sampler.sample(constraint=constraint)
