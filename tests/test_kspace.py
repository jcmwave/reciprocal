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
    kspace_obj.regular_sampler.sample()

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

