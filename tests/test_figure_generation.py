import pytest
import numpy as np
import os
import reciprocal
from reciprocal.kspace import KSpace
from reciprocal.canvas import Canvas, choose_color
from reciprocal.lattice import LatticeVectors, Lattice
from reciprocal.kvector import KVectorGroup
from reciprocal.symmetry import Symmetry
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib.patches import Circle
import matplotlib.cm as cm
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['image.cmap']='plasma'

def get_lattice_vectors(shape):
    period = 1.
    if shape == 'square':
        return LatticeVectors.from_lengths_angle(period, period, 90.)
    elif shape == 'rectangle':
        return LatticeVectors.from_lengths_angle(period*1.5, period, 90.)
    elif shape == 'hexagon':
        return LatticeVectors.from_lengths_angle(period, period, 60.)
    elif shape == 'oblique':
        return LatticeVectors.from_lengths_angle(period, period, 75.)
    
def get_symmetry(shape):
    if shape == 'square':
        return 'D4'
    elif shape == 'rectangle':
        return 'D2'
    elif shape == 'hexagon':
        return 'D6'
    elif shape == 'oblique':
        return 'C2'

@pytest.fixture(autouse=True, scope="session")
def figures_dir():
    figures_dir = os.path.abspath("figures")
    if not os.path.isdir(figures_dir):
        os.makedirs(figures_dir)
    
    yield figures_dir
    shutil.rmtree(figures_dir, ignore_errors=True)

def test_real_space_lattice_only(figures_dir):
    fig = plt.figure(figsize=(12,12))
    gs = gridspec.GridSpec(2, 2, fig, wspace=0.4, hspace=0.4)

    for ilat, lat_shape in enumerate(['square', 'rectangle', 'hexagon', 'oblique']):
        index =  np.unravel_index(ilat, (2,2))
        ax = fig.add_subplot(gs[index])    
        canvas = Canvas(ax=ax)
        lat_vec = get_lattice_vectors(lat_shape)
        lat = Lattice(lat_vec)
        canvas.plot_tesselation(lat)
        canvas.plot_lattice(lat)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.title(lat_shape, fontsize=20)
    fig_path = os.path.join(figures_dir, "01_test_real_space_lattice_only.png")
    plt.savefig(fig_path)                             

def test_real_space_lattice_sampled(figures_dir):
    fig = plt.figure(figsize=(12,12))
    gs = gridspec.GridSpec(2, 2, fig, wspace=0.4, hspace=0.4)

    for ilat, lat_shape in enumerate(['square', 'rectangle', 'hexagon', 'oblique']):
        index =  np.unravel_index(ilat, (2,2))
        ax = fig.add_subplot(gs[index])    
        canvas = Canvas(ax=ax)
        lat_vec = get_lattice_vectors(lat_shape)
        lat = Lattice(lat_vec)
        canvas.plot_tesselation(lat)
        #canvas.plot_lattice(lat)    
        sampling, weighting, int_element = lat.unit_cell.sample(use_symmetry=False)
        canvas.plot_point_sampling(sampling)
        print(np.unique(weighting))
        print(np.sum(weighting*int_element)/lat.unit_cell.area())
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.title(lat_shape, fontsize=20)
    fig_path = os.path.join(figures_dir, "02_test_real_space_lattice_sampled.png")
    plt.savefig(fig_path)

def test_reciprocal_lattice_only(figures_dir):
    fig = plt.figure(figsize=(12,12))
    gs = gridspec.GridSpec(2, 2, fig, wspace=0.4, hspace=0.4)

    for ilat, lat_shape in enumerate(['square', 'rectangle', 'hexagon', 'oblique']):
        index =  np.unravel_index(ilat, (2,2))
        ax = fig.add_subplot(gs[index])    
        canvas = Canvas(ax=ax)
        lat_vec = get_lattice_vectors(lat_shape)
        lat = Lattice(lat_vec)
        rlat = lat.make_reciprocal()
        canvas.plot_tesselation(rlat)    
        canvas.plot_vectors(rlat)    
        canvas.plot_irreducible_uc(rlat.unit_cell)
        canvas.plot_special_points(rlat.unit_cell)


        plt.xlabel('kx')
        plt.ylabel('ky')
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        plt.title(lat_shape, fontsize=20)
    fig_path = os.path.join(figures_dir, "02_test_reciprocal_lattice_only.png")
    plt.savefig(fig_path)

def test_weighted_sampling_of_ibz_no_symmetry(figures_dir):
    fig = plt.figure(figsize=(12,12))
    gs = gridspec.GridSpec(2, 2, fig, wspace=0.4, hspace=0.4)

    for ilat, lat_shape in enumerate(['square', 'rectangle', 'hexagon', 'oblique']):
        #print(lat_shape)
        index =  np.unravel_index(ilat, (2,2))
        ax = fig.add_subplot(gs[index])    
        canvas = Canvas(ax=ax)
        lat_vec = get_lattice_vectors(lat_shape)
        lat = Lattice(lat_vec)
        rlat = lat.make_reciprocal()
        #print("lattice lengths: [{} {}]".format(rlat.vectors.length1, rlat.vectors.length2))
        canvas.plot_tesselation(rlat)    
        constraint = {'type':'n_points', 'value':5}
        sampling, weighting, int_element = rlat.unit_cell.sample(use_symmetry=False,
                                                                constraint=constraint)
        #print(sampling)
        #print("integration element: {}".format(int_element))
        print("unique weighting: {}".format(np.unique(weighting)))
        #print(np.sum(weighting))
        #print(weighting)
        canvas.plot_point_sampling_weighted(sampling, weighting)
        full_area = rlat.unit_cell.area()
        weight_sum = np.sum(int_element*weighting)
        print("Area of reciprocal lattice: {:.6f}".format(rlat.unit_cell.area()))
        print("integration value check: {:.6f}".format(weight_sum/rlat.unit_cell.area()))
        #print("simple check: {}".format(full_area/int_element))
        plt.xlabel('kx')
        plt.ylabel('ky')
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        plt.title(lat_shape, fontsize=20)
    fig_path = os.path.join(figures_dir, "04_test_weighted_sampling_of_ibz_no_symmetry.png")
    plt.savefig(fig_path)

def test_weighted_sampling_of_ibz_with_symmetry(figures_dir):
    fig = plt.figure(figsize=(12,12))
    gs = gridspec.GridSpec(2, 2, fig, wspace=0.4, hspace=0.4)

    for ilat, lat_shape in enumerate(['square', 'rectangle', 'hexagon', 'oblique']):
        print(lat_shape)
        index =  np.unravel_index(ilat, (2,2))
        ax = fig.add_subplot(gs[index])    
        canvas = Canvas(ax=ax)
        lat_vec = get_lattice_vectors(lat_shape)
        lat = Lattice(lat_vec)
        rlat = lat.make_reciprocal()
        canvas.plot_tesselation(rlat)    
        sampling, weighting, int_element, sym_ops = rlat.unit_cell.sample_irreducible()

        print("integration element: {:.6f}".format(int_element))
        print("unique weighting: {}".format(np.unique(weighting)))
        canvas.plot_point_sampling_weighted(sampling, weighting)
        full_area = rlat.unit_cell.area()
        weight_sum = np.sum(int_element*weighting)
        symmetry_multiplier = rlat.unit_cell.symmetry().get_n_symmetry_ops()
        print("integration value check: {:.6f}".format(symmetry_multiplier*weight_sum/full_area))
        plt.xlabel('kx')
        plt.ylabel('ky')
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        plt.title(lat_shape, fontsize=20)
    fig_path = os.path.join(figures_dir, "05_test_weighted_sampling_of_ibz_with_symmetry.png")
    plt.savefig(fig_path)

def test_bz_smapling_with_symmetry(figures_dir):
    fig = plt.figure(figsize=(12,12))
    gs = gridspec.GridSpec(2, 2, fig, wspace=0.4, hspace=0.4)

    for ilat, lat_shape in enumerate(['square', 'rectangle', 'hexagon', 'oblique']):
        print(lat_shape)
        index =  np.unravel_index(ilat, (2,2))
        ax = fig.add_subplot(gs[index])    
        canvas = Canvas(ax=ax)
        lat_vec = get_lattice_vectors(lat_shape)
        lat = Lattice(lat_vec)
        rlat = lat.make_reciprocal()
        #print("lattice lengths: [{} {}]".format(rlat.vectors.length1, rlat.vectors.length2))
        canvas.plot_tesselation(rlat)    
        canvas.plot_vectors(rlat)  
        constraint = {'type':'n_points', 'value':4}
        sampling, weighting, int_element = rlat.unit_cell.sample(use_symmetry=True,
                                                                constraint=constraint)
        #print(sampling)
        #print("integration element: {}".format(int_element))
        #print("unique weighting: {}".format(np.unique(weighting)))
        #print(np.sum(weighting))
        canvas.plot_point_sampling_weighted(sampling, weighting)
        #full_area = rlat.unit_cell.area()
        weight_sum = np.sum(weighting)
        print("integration value check: {}".format(weight_sum))
        #print("simple check: {}".format(full_area/int_element))
        plt.xlabel('kx')
        plt.ylabel('ky')
        plt.xlim([-7, 7])
        plt.ylim([-7, 7])
        plt.title(lat_shape, fontsize=20)
    fig_path = os.path.join(figures_dir, "06_test_bz_with_symmetry.png")
    plt.savefig(fig_path)

def lorentz(gamma, x0, x):
    return (1./np.pi) * 0.5*gamma / ( (x-x0)**2 + (0.5*gamma)**2)

def int_lorentz(gamma, x0, x):
    return (1./np.pi)*np.arctan(2*(x-x0)/gamma)

def int_lorentz_radial(gamma, x0, x):
    return (1./np.pi)*(gamma*np.log( 4*(x-x0)**2 + gamma**2)/4.0 + x0 * np.arctan( 2*(x-x0)/gamma))

def cusp_function(rho, width, pos):
    L = lorentz(width, pos, rho)
    return L
    
def int_cusp_function(rho):
    L_int = int_lorentz(0.05, 0.5, rho)
    return L_int + (10./4.)*rho**4 + 10*rho

def int_cusp_function_radial(rho):
    L_int_radial = int_lorentz_radial(0.05, 0.5, rho)
    return L_int_radial + (10./5.)*rho**5 + (10/2.)*rho**2

def radial_cusp_function(kx, ky, kmax):
    kr = np.sqrt( kx**2 + ky**2)/kmax    
    pos = kmax*0.4
    width = kmax/20
    return cusp_function(kr, width, pos)

def test_rlattice_in_kspace(figures_dir):
    fig = plt.figure(figsize=(12,12))
    gs = gridspec.GridSpec(2, 2, fig, wspace=0.4, hspace=0.4)

    for ilat, lat_shape in enumerate(['square', 'rectangle', 'hexagon', 'oblique']):
        index =  np.unravel_index(ilat, (2,2))
        ax = fig.add_subplot(gs[index])    
        canvas = Canvas(ax=ax)
        lat_vec = get_lattice_vectors(lat_shape)
        lat = Lattice(lat_vec)
        rlat = lat.make_reciprocal()
        canvas.plot_tesselation(rlat)
        canvas.plot_lattice(rlat)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        plt.title(lat_shape, fontsize=20)
    fig_path = os.path.join(figures_dir, "07_test_rlattice_in_kspace.png")
    plt.savefig(fig_path)

def test_bloch_families_in_kspace(figures_dir):
    fig = plt.figure(figsize=(12,12))
    gs = gridspec.GridSpec(2, 2, fig, wspace=0.4, hspace=0.4)

    for ilat, lat_shape in enumerate(['square', 'rectangle', 'hexagon', 'oblique']):
        index =  np.unravel_index(ilat, (2,2))
        ax = fig.add_subplot(gs[index])    
        canvas = Canvas(ax=ax)
        wvl = 0.5
        k0 = 2*np.pi/wvl
        kspace = KSpace(wvl, symmetry=get_symmetry(lat_shape), fermi_radius=k0)
        lat_vec = get_lattice_vectors(lat_shape)
        lat = Lattice(lat_vec)
        rlat = lat.make_reciprocal()
        kspace.apply_lattice(rlat)
        canvas.plot_tesselation(rlat)            
        canvas.plot_fermi_circle(kspace)
        families = kspace.periodic_sampler.sample_bloch_families()
        canvas.plot_bloch_families(families)



        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([-15, 15])
        plt.ylim([-15, 15])
        plt.title(lat_shape, fontsize=20)
    fig_path = os.path.join(figures_dir, "08_test_bloch_families_in_kspace.png")
    plt.savefig(fig_path)


def test_bloch_families_in_sym_cone(figures_dir):
    fig = plt.figure(figsize=(12,12))
    gs = gridspec.GridSpec(2, 2, fig, wspace=0.4, hspace=0.4)

    for ilat, lat_shape in enumerate(['square', 'rectangle', 'hexagon', 'oblique']):
        index =  np.unravel_index(ilat, (2,2))
        ax = fig.add_subplot(gs[index])    
        canvas = Canvas(ax=ax)
        wvl = 0.5
        k0 = 2*np.pi/wvl
        kspace = KSpace(wvl, symmetry=get_symmetry(lat_shape), fermi_radius=k0)
        lat_vec = get_lattice_vectors(lat_shape)
        lat = Lattice(lat_vec)
        rlat = lat.make_reciprocal()
        kspace.apply_lattice(rlat)
        canvas.plot_tesselation(rlat)            
        canvas.plot_fermi_circle(kspace)
        families = kspace.periodic_sampler.sample_bloch_families(restrict_to_sym_cone=True)
        canvas.plot_bloch_families(families)
        canvas.plot_symmetry_cone(kspace)


        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([-15, 15])
        plt.ylim([-15, 15])
        plt.title(lat_shape, fontsize=20)
    fig_path = os.path.join(figures_dir, "09_test_bloch_families_in_sym_cone.png")
    plt.savefig(fig_path)



def test_symmetrised_bloch_families_in_kspace(figures_dir):
    fig = plt.figure(figsize=(12,12))
    gs = gridspec.GridSpec(2, 2, fig, wspace=0.4, hspace=0.4)

    for ilat, lat_shape in enumerate(['square', 'rectangle', 'hexagon', 'oblique']):
        index =  np.unravel_index(ilat, (2,2))
        ax = fig.add_subplot(gs[index])    
        canvas = Canvas(ax=ax)
        wvl = 0.5
        k0 = 2*np.pi/wvl
        kspace = KSpace(wvl, symmetry=get_symmetry(lat_shape), fermi_radius=k0)
        lat_vec = get_lattice_vectors(lat_shape)
        lat = Lattice(lat_vec)
        rlat = lat.make_reciprocal()
        kspace.apply_lattice(rlat)
        canvas.plot_tesselation(rlat)
        canvas.plot_fermi_circle(kspace)
        sampling = kspace.periodic_sampler.sample(restrict_to_sym_cone=True, constraint={'type':'n_points', 'value':4})
        #canvas.plot_sampling(sampling.k,  color='k')
        groups = kspace.symmetrise_sample(sampling)
        canvas.plot_symmetry_cone(kspace)    

        cmap = mpl.colormaps['tab20']
        NColors = 20
        for i in range(len(groups)):
            color = np.zeros((1,4))
            color[0,:] = cmap((float(i))/NColors)
            canvas.plot_point_sampling(groups[i], plot_n_points='all', color = color)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([-15, 15])
        plt.ylim([-15, 15])
        plt.title(lat_shape, fontsize=20)
    fig_path = os.path.join(figures_dir, "10_test_symmetrised_bloch_families_in_kspace.png")
    plt.savefig(fig_path)


def test_woods_anomalies_no_symmetry(figures_dir):
    fig = plt.figure(figsize=(12,12))
    gs = gridspec.GridSpec(2, 2, fig, wspace=0.4, hspace=0.4)

    for ilat, lat_shape in enumerate(['square', 'rectangle', 'hexagon', 'oblique']):
        index =  np.unravel_index(ilat, (2,2))
        ax = fig.add_subplot(gs[index])    
        canvas = Canvas(ax=ax)
        wvl = 2.0
        k0 = 2*np.pi*wvl
        kspace = KSpace(wvl, symmetry=get_symmetry(lat_shape), fermi_radius=k0)
        lat_vec = get_lattice_vectors(lat_shape)
        lat = Lattice(lat_vec)
        rlat = lat.make_reciprocal()
        kspace.apply_lattice(rlat)
        canvas.plot_tesselation(rlat)            
        canvas.plot_fermi_circle(kspace)
        woods1 = kspace.periodic_sampler.calc_woods_anomalies(1, n_refinements = 4, restrict_to_sym_cone=True)
        cmap = mpl.colormaps['tab20']
        NColors = 20
        for i in range(len(woods1)):
            color = np.zeros((1,4))
            color[0,:] = cmap((float(i))/NColors)    
            canvas.plot_point_sampling(woods1[i], plot_n_points='all', color =color)
        canvas.plot_symmetry_cone(kspace)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([-15, 15])
        plt.ylim([-15, 15])
        plt.title(lat_shape, fontsize=20)
    fig_path = os.path.join(figures_dir, "11_woods_anomalies_no_symmetry.png")
    plt.savefig(fig_path)

def test_woods_anomalies_with_symmetry(figures_dir):
    fig = plt.figure(figsize=(12,12))
    gs = gridspec.GridSpec(2, 2, fig, wspace=0.4, hspace=0.4)

    for ilat, lat_shape in enumerate(['square', 'rectangle', 'hexagon', 'oblique']):
        index =  np.unravel_index(ilat, (2,2))
        ax = fig.add_subplot(gs[index])    
        canvas = Canvas(ax=ax)
        wvl = 2.0
        k0 = 2*np.pi*wvl
        kspace = KSpace(wvl, symmetry=get_symmetry(lat_shape), fermi_radius=k0)
        lat_vec = get_lattice_vectors(lat_shape)
        lat = Lattice(lat_vec)
        rlat = lat.make_reciprocal()
        kspace.apply_lattice(rlat)
        canvas.plot_tesselation(rlat)            
        canvas.plot_fermi_circle(kspace)
        woods1 = kspace.periodic_sampler.calc_woods_anomalies(1, n_refinements = 4, restrict_to_sym_cone=True)
        cmap = mpl.colormaps['tab20']
        NColors = 20
        for i in range(len(woods1)):
            color = np.zeros((1,4))
            color[0,:] = cmap((float(i))/NColors)    
            #canvas.plot_point_sampling(woods1[i], plot_n_points='all', color =color)
            sym_woods = kspace.symmetrise_sample(woods1[i])
            for i_wood in range(len(sym_woods)):
                canvas.plot_point_sampling(sym_woods[i_wood], color=color)

        canvas.plot_symmetry_cone(kspace)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([-15, 15])
        plt.ylim([-15, 15])
        plt.title(lat_shape, fontsize=20)
    fig_path = os.path.join(figures_dir, "12_woods_anomalies_with_symmetry.png")
    plt.savefig(fig_path)

       
def test_regular_circular_sampling_of_kspace(figures_dir):
    fig = plt.figure(figsize=(12,12))
    gs = gridspec.GridSpec(1, 1, fig, wspace=0.4, hspace=0.4)


    ax = fig.add_subplot(gs[0])    
    canvas = Canvas(ax=ax)    
    wvl = 2.0
    k0 = 2*np.pi*wvl
    kspace = KSpace(wvl, fermi_radius=k0)
    canvas.plot_fermi_circle(kspace)
    plt.xlabel('k$_x$')
    plt.ylabel('k$_y$')
    plt.xlim([-15, 15])
    plt.ylim([-15, 15])
    max_length = k0/3.
    #constraint={'type':'max_length', 'value':max_length}
    constraint={'type':'n_points', 'value':5}
    
    sampling, weighting = kspace.regular_sampler.sample(
        grid_type="circular",
        constraint=constraint
    )
    print(sampling)
    canvas.plot_point_sampling_weighted(sampling, weighting)

    fig_path = os.path.join(figures_dir, "13_test_regular_circular_sampling_of_kspace.png")
    plt.savefig(fig_path)

def test_regular_cartesian_sampling_of_kspace(figures_dir):
    fig = plt.figure(figsize=(12,12))
    gs = gridspec.GridSpec(1, 1, fig, wspace=0.4, hspace=0.4)


    ax = fig.add_subplot(gs[0])    
    canvas = Canvas(ax=ax)    
    wvl = 2.0
    k0 = 2*np.pi*wvl
    kspace = KSpace(wvl, fermi_radius=k0)
    canvas.plot_fermi_circle(kspace)
    plt.xlabel('k$_x$')
    plt.ylabel('k$_y$')
    plt.xlim([-15, 15])
    plt.ylim([-15, 15])
    max_length = k0/3.
    #constraint={'type':'max_length', 'value':max_length}
    constraint={'type':'n_points', 'value':5}
    
    sampling, weighting = kspace.regular_sampler.sample(
        grid_type="cartesian",
        constraint=constraint
    )
    print(sampling)
    canvas.plot_point_sampling_weighted(sampling, weighting)

    fig_path = os.path.join(figures_dir, "14_test_regular_cartesian_sampling_of_kspace.png")
    plt.savefig(fig_path)    
