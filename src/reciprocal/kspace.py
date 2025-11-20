import numpy as np
#import matplotlib as mpl
from matplotlib import cm
from matplotlib.patches import Wedge, Circle, Rectangle
#import matplotlib.pyplot as plt
#from matplotlib.patches import Polygon
#from numpy.lib.scimath import sqrt as csqrt
from copy import copy
import scipy.spatial
#import shapely.geometry
import scipy.optimize
#from shapely.geometry.point import Point
#from shapely.geometry.linestring import LineString
from shapely.geometry.polygon import Polygon
#from descartes import PolygonPatch
from abc import ABC
from reciprocal.symmetry import Symmetry, SpecialPoint, PointSymmetry, symmetry_from_type
from reciprocal.utils import (order_lexicographically,
                              lies_on_poly, lies_on_vertex) #apply_symmetry_operators,
from reciprocal.kvector import KVectorGroup, BlochFamily
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
#from kspacesampling import KVector
#from kspacesampling import BrillouinZone
#from kspacesampling import Symmetry
#import kspacesampling

def test_for_duplicates(old_points, new_point):
    already_included = False
    for kk, sampled in enumerate(old_points):
        if np.isclose(new_point, sampled,rtol=1e-2,atol=1e-4).all():
            already_included = True
            if already_included:
                return True
    return False

def get_bloch_statistics(bloch_sampling):
    info = {}
    info['NFamilies'] = len(bloch_sampling)
    family_sizes = {}
    n_points_total = 0
    for key in bloch_sampling:
        family = bloch_sampling[key]
        if isinstance(family, np.ndarray):
            n_siblings = family.shape[0]
        else:
            n_siblings = family.k.shape[0]
        n_points_total += n_siblings
        family_sizes[key] = n_siblings
    info['NKPoints'] = n_points_total
    info['FamilySizes'] = family_sizes
    info['Speedup'] = info['NKPoints']/info['NFamilies']
    return info

def spiral_arc_length(b, phi0, phi1):
    lower = phi0*np.sqrt(1+phi0**2) + np.arcsinh(phi0)
    upper = phi1*np.sqrt(1+phi1**2) + np.arcsinh(phi1)
    return b*0.5*(upper-lower)

def jac_spiral_arc_length(b, phi1):
    g = 1 + phi1**2
    upper = g**(-0.5) + g**(0.5) + phi1**2 * g**(-0.5)
    return b*0.5*upper

def compare_arc_length(x, b, phi0, goal_length):
    arc_length = spiral_arc_length(b, phi0, x[0])
    #jac = jac_compare_arc_length(x, b)
    #return_val = np.array([goal_length-arc_length, jac])
    #print(return_val)
    return arc_length-goal_length

def jac_compare_arc_length(x, b):
    jac_arc_length = jac_spiral_arc_length(b, x[0])
    return jac_arc_length


class KSpace():
    """
    class to obtain 2D samplings of points in reciprocal space.

    Attributes
    ----------
    wavelength: float
        wavelength of light in vacuum in meters
    k0: float
        length of the wave vector in vacuum in 1/m
    fermi_radius :float
        radius of the Fermi circle in 1/m.
    symmetry: Symmetry
        Symmetry instance defining the symmetry of the real space structure
        (see Symmetry.py for more information).
    symmetry_cone: np.array<float>(3,2))
        x,y vertices of the symmetry cone in 1/m.
    KSampling: np.array<float>(N,2))
        all k points inside the Fermi radius.
    sample_sets: list of <KVectorGroup>
        list of k kvector groups defining k-space samplings
    periodic_sampler: PeriodicSampler
        subsystem to create k-space samplings based on periodic structures
    """

    def __init__(self, wavelength, symmetry=None, fermi_radius=None):
        #self.bzone = brillouinZone
        self.wavelength = wavelength
        self.k0 = np.pi*2/wavelength
        self.fermi_radius = fermi_radius
        self.symmetry_cone = None
        self.sample_sets = []
        self.periodic_sampler = None
        self.regular_sampler = RegularSampler(self)
        if symmetry is not None:
            self.symmetry = Symmetry.from_string(symmetry)
            self.calc_symmetry_cone()
        else:
            self.symmetry = None

    def set_symmetry(self, symmetry):
        """
        Parameters
        ----------
        symmetry: string
        """
        sym = Symmetry.from_string(symmetry)
        if periodic_sampler is not None:
            lattice_sym = periodic_sampler.lattice.unit_cell.symmetry()
            if not lattice_sym.compatible(sym):
                raise ValueError("symmetry {} ".format(sym)+
                                 "is not compatible with lattice symmetry"+
                                 " {}".format(lattice_sym))
            if lattice_sym.get_n_symmetry_ops() < sym.get_n_symmetry_ops():
                raise ValueError("kspace symmetry {} ".format(sym)+
                                 "may not have a higher degree of symmetry "+
                                 "than lattice symmetry"+
                                 " {}".format(lattice_sym))
        self.symmetry = sym

    def calc_symmetry_cone(self):
        """
        calculate the 3 points of the symmetry wedge.
        """
        if self.fermi_radius is None:
            return
        angle0 = 0.0
        angle1 = self.symmetry.get_symmetry_cone_angle()
        xy0 = np.array([0.0,0.0])
        xy1 = np.array([self.fermi_radius,0.0])
        xy2 = self.fermi_radius*np.array([np.cos(angle1),np.sin(angle1)])
        self.symmetry_cone = np.vstack([xy1, xy0, xy2])


    def symmetrise_sample(self, kv_group):
        """
        given a k-space sampling (KVectorGroup), return a list of KVectorGroups
        which contain all the symmetry points for each.
        """
        n_points = kv_group.n_rows

        opening_angle = self.symmetry.get_symmetry_cone_angle()

        counter = 0
        symmetry_groups = []

        for i in range(self.symmetry.get_n_symmetry_ops()):
            symmetry_groups.append([])

        for row in range(n_points):
            k = kv_group.k[row,:]
            kxy = np.array(k[:2])
            kxyz = np.array(k[:])
            gamma_vertex = {SpecialPoint.GAMMA:np.array([0., 0., 0.])}
            on_vertex, special_point = lies_on_vertex(kxy, gamma_vertex)
            if on_vertex:
                points = np.array([[0., 0., 0.]])
            elif lies_on_poly(kxy, self.symmetry_cone, closed=False):
                # try:
                #     reduced_sym = self.symmetry.reduce()
                # except ValueError:
                reduced_sym = self.symmetry
                points = reduced_sym.apply_symmetry_operators(kxyz)
            else:
                points = self.symmetry.apply_symmetry_operators(kxyz)
            for sym_row in range(points.shape[0]):
                point = points[sym_row, :]
                symmetry_groups[sym_row].append(point)
        for i_sym, point_list in enumerate(symmetry_groups):
            point_array = np.vstack(point_list)[:, :2]
            #point_array = order_lexicographically(point_array)
            kvs = self.convert_to_KVectors(point_array, 1.0, 1)
            symmetry_groups[i_sym] = kvs
        return symmetry_groups

    def symmetrise_sample_weighted(self, kv_group, weighting):
        """
        given a k-space sampling (KVectorGroup), return a list of KVectorGroups
        which contain all the symmetry points for each.
        """
        n_points = kv_group.n_rows

        opening_angle = self.symmetry.get_symmetry_cone_angle()

        counter = 0
        symmetry_groups = []
        weight_groups = []

        for i in range(self.symmetry.get_n_symmetry_ops()):
            symmetry_groups.append([])
            weight_groups.append([])
        for row in range(n_points):
            k = kv_group.k[row,:]
            weight = weighting[row]
            kxy = np.array(k[:2])
            kxyz = np.array(k[:])
            gamma_vertex = {SpecialPoint.GAMMA:np.array([0., 0., 0.])}
            on_vertex, special_point = lies_on_vertex(kxy, gamma_vertex)
            if on_vertex:
                points = np.array([[0., 0., 0.]])
            elif lies_on_poly(kxy, self.symmetry_cone, closed=False):
                # try:
                #     reduced_sym = self.symmetry.reduce()
                # except ValueError:
                reduced_sym = self.symmetry
                points, operators = reduced_sym.apply_symmetry_operators(kxy)
            else:
                points, operators = self.smmetry.apply_symmetry_operators(kxy)
            for sym_row in range(points.shape[0]):
                point = points[sym_row, :]
                symmetry_groups[sym_row].append(point)
                weight_groups[sym_row].append(weight)
        for i_sym, point_list in enumerate(symmetry_groups):
            point_array = np.vstack(point_list)
            weight_array = np.vstack(weight_groups[i_sym])
            #point_array = order_lexicographically(point_array)
            kvs = self.convert_to_KVectors(point_array, 1.0, 1)
            symmetry_groups[i_sym] = kvs
            weight_groups[i_sym] = weight_array
        return symmetry_groups, weight_groups

    def convert_to_KVectors(self, points, n, direction):
        """
        convert a (N,2) np.array of kx,ky values into a KVectorGroup object.
        """
        nRows = points.shape[0]
        n = self.fermi_radius/self.k0
        return KVectorGroup(self.wavelength, nRows,
                            kx=points[:,0],
                            ky=points[:,1],
                            n=n, normal=direction)

    def apply_lattice(self, lattice):
        """
        creates a periodic sampler using a Lattice (lattice.py) object.
        """
        if self.symmetry is not None:
            lattice_sym = lattice.unit_cell.symmetry()
            if not lattice_sym.compatible(self.symmetry):
                raise ValueError("lattice symmetry {} ".format(lattice_sym)+
                                 "is not compatible with kspace symmetry"+
                                 " {}".format(self.symmetry))
            if lattice_sym.get_n_symmetry_ops() < self.symmetry.get_n_symmetry_ops():
                raise ValueError("kspace symmetry {} ".format(self.symmetry)+
                                  "may not have a higher degree of symmetry "+
                                  "than lattice symmetry"+
                                  " {}".format(lattice_sym))
        self.periodic_sampler = PeriodicSampler(lattice, self)

    def restrict_to_fermi_radius(self, points, tol=1e-3, return_indices=False):
        lengths = np.linalg.norm(points[:,:2], axis=1)
        keep = lengths <= self.fermi_radius*(1-tol)
        new_points = points[keep, :]
        if return_indices:
            return new_points, keep
        else:
            return new_points

    def restrict_to_sym_cone(self, points, tol=1e-3, return_indices=False):
        #if trial_point[1]> trial_point[0]*np.tan(opening_angle)+tol:
        #    return True
        opening_angle = self.symmetry.get_symmetry_cone_angle()
        angles = np.angle(points[:, 0]+1j*points[:, 1])
        keep = np.logical_and(angles>-tol, angles<opening_angle+tol)
        new_points = points[keep, :]
        if return_indices:
            return new_points, keep
        else:
            return new_points

    def restrict_to_reflection_cone(self, points, tol=1e-6, return_indices=False):
        #if trial_point[1]> trial_point[0]*np.tan(opening_angle)+tol:
        #    return True
        #opening_angle = self.symmetry.get_symmetry_cone_angle()
        #angles = np.angle(points[:, 0]+1j*points[:, 1])
        keep = points[:, 0] >= -tol
        new_points = points[keep, :]
        if return_indices:
            return new_points, keep
        else:
            return new_points

class Sampler(ABC):
    """ abstract base class for Sampling objects

    """

    def __init__(self, kspace):
        self.kspace = kspace

    def test_outside_symmetry_cone(self, trial_point,opening_angle):
        tol = 1e-3
        #if trial_point[1]> trial_point[0]*np.tan(opening_angle)+tol:
        #    return True
        angle = np.angle(trial_point[0]+1j*trial_point[1])
        if angle<-tol or angle>opening_angle+tol:
            return True
        ktrans_sq = trial_point[0]**2 + trial_point[1]**2
        if self.kspace.fermi_radius**2 - ktrans_sq < 0.0:
            return True
        return False

class PseudoRandomSampler(Sampler):

    """
    class to obtain a psuedo-random k space sampling

    kspace(Kspace): reference to parent kspace
    """

    def __init__(self, kspace):
        """
        Parameters
        ----------
        kspace: dispersion.kspace.KSpace
            reference to parent kspace
        """
        super().__init__(kspace)


class RegularSampler(Sampler):

    """
    class to obtain a k space sampling on a regular grid

    kspace(Kspace): reference to parent kspace
    """

    def __init__(self, kspace):
        """
        Parameters
        ----------
        kspace: dispersion.kspace.KSpace
            reference to parent kspace
        """
        super().__init__(kspace)

    def sample(self, grid_type='cartesian', constraint=None, center=False,
               cutoff_tol=1e-5, restrict_to_sym_cone=False, return_artists=False):
        if grid_type == 'cartesian':
            sampling_output = self._sample_cartesian(constraint, center, cutoff_tol,
                                           restrict_to_sym_cone, return_artists)
        elif grid_type == 'circular':
            sampling_output = self._sample_circular(constraint, center, cutoff_tol,
                                           restrict_to_sym_cone, return_artists)
        elif grid_type == 'spiral':
            sampling_output = self._sample_spiral(constraint, center, cutoff_tol,
                                           restrict_to_sym_cone, return_artists)
        else:
            raise ValueError("unknown grid type: {}, allowed types |cartesian|circular|spiral".format(grid_type))
        return sampling_output

    def _npoints_from_constraint(self, vector_lengths, constraint):
        """
        Return number of k space sampling points based on constraint

        Parameters
        ----------
        constraint: dict
            constraints for number of points

        Returns
        -------
        int
            number of grid points
        """
        if constraint is None:
            constraint = {'type':'n_points', 'value':5}
        if constraint['type'] == "density":
            density = constraint['value']
            max_length = 1/(np.sqrt(density)*2)
            n_grid_points = []
            for length in vector_lengths:
                n_grid_points.append(int(length/max_length))
        elif constraint['type'] == 'max_length':
            max_length = constraint['value']
            n_grid_points = []
            for length in vector_lengths:
                n_grid_points.append(int(length/max_length))
        elif constraint['type'] == "n_points":
            try:
                n_grid_points = [constraint['value'][0], constraint['value'][1]]
            except:
                n_grid_points = [constraint['value'], constraint['value']]
        else:
            raise ValueError(f"constraint type {constraint['type']} invalid, valid choices are: density|max_length|n_points")

        for ii, gp in enumerate(n_grid_points):
            if gp == 0:
                n_grid_points[ii] = 1

        return n_grid_points

    def _max_lengths_from_constraint(self, vector_lengths, constraint):
        """
        Return number of k space sampling points based on constraint

        Parameters
        ----------
        constraint: dict
            constraints for number of points

        Returns
        -------
        int
            number of grid points
        """
        if constraint is None:
            constraint = {'type':'n_points', 'value':5}
        if constraint['type'] == "density":
            density = constraint['value']
            max_length = 1/(np.sqrt(density)*2)
            max_lengths = [max_length, max_length]
        elif constraint['type'] == 'max_length':
            max_length = constraint['value']
            try:
                max_lengths = [constraint['value'][0], constraint['value'][1]]
            except:
                max_lengths = [constraint['value'], constraint['value']]
        elif constraint['type'] == "n_points":
            max_lengths = []
            for il, length in enumerate(vector_lengths):
                try:
                    n_points = constraint['value'][il]
                except:
                    n_points = constraint['value']
                max_lengths.append(length/n_points)

        return max_lengths

    def _sample_spiral(self, constraint, center, cutoff_tol,
                          restrict_to_sym_cone, return_artists):
        vector1 = np.array([self.kspace.fermi_radius, 0.])
        vector_lengths = np.array([self.kspace.fermi_radius])
        if restrict_to_sym_cone:
            opening_angle = self.kspace.symmetry.get_symmetry_cone_angle()
        else:
            opening_angle = 2*np.pi

        n_grid_points = self._npoints_from_constraint(vector_lengths, constraint)
        #print("n grid points: {}".format(n_grid_points))
        radial_range = range(0, n_grid_points[0])

        #circumference = np.pi*2*self.kspace.fermi_radius
        #n_phis = self._npoints_from_constraint([circumference], constraint)

        all_points = []
        weighting = []
        artists = []
        rad_spacing = vector_lengths[0]/(n_grid_points[0]-0.5)*3.
        #print("rad spacing: {}".format(rad_spacing))
        a = 0.
        b = rad_spacing/(np.pi*2.0)
        if center:
            raise ValueError("center no supported for spiral grids")

        total_area = np.pi*self.kspace.fermi_radius**2*opening_angle/(2*np.pi)


        step = 0
        for spiral_n in range(3):
            if spiral_n == 0:
                rotation = 0.
            elif spiral_n == 1:
                rotation = np.pi*2/3.
            elif spiral_n == 2:
                rotation = np.pi*2*2./3.
            phi0 = 0.
            current_outer_radius = 0.
            while current_outer_radius < self.kspace.fermi_radius:
                #for n_r in radial_range:
                #print(step, spiral_n)
                if step == 0:
                    all_points.append(np.array([0., 0.]))
                    #center_circle_area = np.pi*(rad_spacing/2.)**2*opening_angle/(2*np.pi)
                    weighting.append(1.)
                    artists.append(Circle((0.,0.), radius=rad_spacing/6., fill=False, edgecolor='k'))
                    step += 1
                    continue

                opt_fun = lambda x: compare_arc_length(x, b, phi0, rad_spacing*0.333)
                opt_jac = lambda x: jac_compare_arc_length(x, b)
                x0 = np.array([phi0])
                res = scipy.optimize.root_scalar(opt_fun, x0=x0, fprime=opt_jac, method='newton')
                phi1 = res.root[0]
                current_outer_radius = b*phi1

                x = current_outer_radius*np.cos(phi1+rotation)
                y = current_outer_radius*np.sin(phi1+rotation)
                trial_point = np.array([x, y])
                length = np.linalg.norm(trial_point)
                if length > self.kspace.fermi_radius*(1-cutoff_tol):
                    break

                weighting.append(1.)
                all_points.append(trial_point)
                artists.append(Circle((x, y), radius=rad_spacing/6., fill=False, edgecolor='k'))
                phi0 = phi1
                step += 1

        all_point_array = np.vstack(all_points)
        all_point_array, sort_indices = order_lexicographically(all_point_array,
                                                       return_sort_indices=True)
        all_kvs = self.kspace.convert_to_KVectors(all_point_array, 1., 1.)
        weighting_array = np.array(weighting)
        weighting_array /= np.sum(weighting_array)
        #weighting_array /= total_area
        weighting_array = weighting_array[sort_indices]
        if return_artists:
            artists = np.array(artists)
            artists = artists[sort_indices]
            return all_kvs, weighting_array, artists
        else:
            return all_kvs, weighting_array

    def _sample_circular(self, constraint, center, cutoff_tol,
                          restrict_to_sym_cone, return_artists):
        vector1 = np.array([self.kspace.fermi_radius, 0.])
        vector_lengths = np.array([self.kspace.fermi_radius])
        if restrict_to_sym_cone:
            opening_angle = self.kspace.symmetry.get_symmetry_cone_angle()
        else:
            opening_angle = 2*np.pi
        n_grid_points = self._npoints_from_constraint(vector_lengths, constraint)
        max_lengths = self._max_lengths_from_constraint(vector_lengths, constraint)
        azimuthal_constraint = {'type':'max_length', 'value':np.min(max_lengths)}
        radial_range = range(0, n_grid_points[0])

        all_points = []
        weighting = []
        artists = []
        rad_spacing = vector_lengths[0]/(n_grid_points[0]-0.5)

        if center:
            raise ValueError("center not supported for circular grids")

        total_area = np.pi*self.kspace.fermi_radius**2*opening_angle/(2*np.pi)

        for n_r in radial_range:
            if n_r == 0:
                all_points.append(np.array([0., 0.]))
                center_circle_area = np.pi*(rad_spacing/2.)**2*opening_angle/(2*np.pi)
                weighting.append(center_circle_area/total_area)
                if np.isclose(opening_angle, np.pi*2.):
                    artists.append(Circle((0.,0.), radius=rad_spacing/2., fill=False, edgecolor='k'))
                else:
                    artists.append(Wedge((0.,0.), rad_spacing/2., 0.,
                                    np.degrees(opening_angle),
                                    fill=False, edgecolor='k'))
                continue
            radius = rad_spacing*n_r
            upper_radius = rad_spacing*(n_r+0.5)
            lower_radius = rad_spacing*(n_r-0.5)
            circumference = opening_angle*radius
            n_phis = self._npoints_from_constraint([circumference], azimuthal_constraint)
            phis = np.linspace(0, np.degrees(opening_angle), n_phis[0])
            if phis.size > 1:
                phis = phis[:-1]
            if phis.size == 1:
                phi_spacing = np.degrees(opening_angle)
            else:
                phi_spacing = phis[1]-phis[0]
            if restrict_to_sym_cone:
                phis = phis[phis<=np.degrees(opening_angle)]
            for phi in phis:
                x = radius*np.cos(np.radians(phi))
                y = radius*np.sin(np.radians(phi))
                trial_point = np.array([x, y])
                length = np.linalg.norm(trial_point)
                if length > self.kspace.fermi_radius*(1-cutoff_tol):
                    pass
                    #continue
                wedge_area = 0.5*(upper_radius**2-lower_radius**2)*np.radians(phi_spacing)
                #print(upper_radius, lower_radius, phi_spacing, wedge_area/total_area)
                weighting.append(wedge_area/total_area)
                all_points.append(trial_point)
                wedge = Wedge(np.array([0., 0.]), upper_radius,
                              phi-phi_spacing*0.5, phi+phi_spacing*0.5,
                              width=upper_radius-lower_radius, fill=False)
                artists.append(wedge)

        all_point_array = np.vstack(all_points)
        all_point_array, sort_indices = order_lexicographically(all_point_array,
                                                       return_sort_indices=True)
        all_kvs = self.kspace.convert_to_KVectors(all_point_array, 1., 1.)
        weighting_array = np.array(weighting)
        weighting_array = weighting_array[sort_indices]
        if return_artists:
            artists = np.array(artists)
            artists = artists[sort_indices]
            return all_kvs, weighting_array, artists
        else:
            return all_kvs, weighting_array

    def _sample_cartesian(self, constraint, center, cutoff_tol,
                         restrict_to_sym_cone, return_artists):
        vector1 = np.array([self.kspace.fermi_radius, 0.])
        vector2 = np.array([0., self.kspace.fermi_radius])
        vector_lengths = np.array([self.kspace.fermi_radius, self.kspace.fermi_radius])
        if restrict_to_sym_cone:
            opening_angle = self.kspace.symmetry.get_symmetry_cone_angle()
            wedge = Wedge((0., 0.), r=self.kspace.fermi_radius*2, theta1=0.,
                          theta2=np.degrees(opening_angle))
            wedge_vertices = wedge._path._vertices
            wedge_polygon = Polygon(wedge_vertices)
            #print("Wedge: {}".format(wedge_polygon))
        else:
            opening_angle = 2*np.pi
        n_grid_points = self._npoints_from_constraint(vector_lengths, constraint)
        max_lengths = self._max_lengths_from_constraint(vector_lengths, constraint)

        range1 = range(-n_grid_points[0], n_grid_points[0]+1)
        range2 = range(-n_grid_points[1], n_grid_points[1]+1)

        all_points = []
        weighting = []
        artists = []
        total_area = np.pi*self.kspace.fermi_radius**2*opening_angle/(2*np.pi)
        #print(max_lengths)
        vector1 = np.array([max_lengths[0], 0.])
        vector2 = np.array([0., max_lengths[0]])
        vector_lengths = max_lengths
        #vector1 /= n_grid_points[0]
        #vector2 /= n_grid_points[1]
        #vector_lengths /= n_grid_points

        if center:
            if not restrict_to_sym_cone:
                central_point = 0.5*(vector1 + vector2)
            else:
                shift_length = np.mean(vector_lengths)*0.5*np.sqrt(2)
                central_point = shift_length*np.array([np.cos(opening_angle*0.5), np.sin(opening_angle*0.5)])

        else:
            central_point = np.array([0., 0.])
        for nx in range1:
            for ny in range2:
                trial_point = nx*vector1 + ny*vector2 + central_point
                length = np.linalg.norm(trial_point)
                if length > self.kspace.fermi_radius*(1-cutoff_tol):
                    continue

                if restrict_to_sym_cone:
                    if self.test_outside_symmetry_cone(trial_point,
                                                      opening_angle):
                        continue
                rect_center = (nx-0.5)*vector1 + (ny-0.5)*vector2 + central_point
                square = Rectangle(rect_center, vector_lengths[0], vector_lengths[1])
                if restrict_to_sym_cone:
                    #square_vertices = square._path._vertices
                    #square_vertices = [rect_center-np.array([vector_lengths[0], 0.])*0.5]
                    #square_vertices.append(rect_center-np.array([vector_lengths[0], 0.])*0.5)
                    p0 = (trial_point -vector1*0.5 -vector2*0.5).tolist()
                    p1 = (trial_point +vector1*0.5 -vector2*0.5).tolist()
                    p2 = (trial_point +vector1*0.5 +vector2*0.5).tolist()
                    p3 = (trial_point -vector1*0.5 +vector2*0.5).tolist()
                    square_vertices = [p0, p1, p2, p3]
                    square_polygon = Polygon(square_vertices)
                    intersect = wedge_polygon.intersection(square_polygon)
                    weight = intersect.area
                    weighting.append(weight)
                    #print(type(intersect))
                    xy = np.array(intersect.boundary.xy)
                    #new_patch = Polygon(xy.T)
                    new_patch = square
                    #print("Trial Point: {}".format(trial_point))
                    #print("Square Polygon: {}".format(square_polygon))
                    #print("Intersection: {}".format(intersect))
                    #new_patch = PolygonPatch(intersect)
                    artists.append(new_patch)
                else:
                    artists.append(square)
                    weighting.append(vector_lengths[0]*vector_lengths[1]/total_area)
                all_points.append(trial_point)

        all_point_array = np.vstack(all_points)
        all_point_array, sort_indices = order_lexicographically(all_point_array,
                                                       return_sort_indices=True)
        all_kvs = self.kspace.convert_to_KVectors(all_point_array, 1., 1.)
        weighting_array = np.array(weighting)
        if restrict_to_sym_cone:
            weighting_array /= weighting_array.sum()
        weighting_array = weighting_array[sort_indices]
        if return_artists:
            return all_kvs, weighting_array, artists
        else:
            return all_kvs, weighting_array

class PeriodicSampler(Sampler):

    """
    class to obtain k space sampling based on periodic structures

    lattice(Lattice): a reciprocal lattice used to generate the k-space sampling
    kspace(Kspace): reference to parent kspace
    """

    def __init__(self, lattice, kspace):
        """
        Parameters
        ----------
        lattice: dispersion.lattice.Lattice
            a reciprocal lattice used to generate the k-space sampling
        kspace: dispersion.kspace.KSpace
            reference to parent kspace
        """
        super().__init__(kspace)
        self.lattice = lattice

    def calc_woods_anomalies(self, order, n_refinements=0,
                             radius=None,
                             restrict_to_sym_cone=False):
        """
        return sets of samplings along Wood anomalies

        Parameters
        ----------
        order: int
            the order of Wood anomaly
        n_refinements: int
            number of refinements of the wood anomaly discretisation

        Returns
        -------
        list (N,3) <np.double> np.array
            list of the wood anomalies
        """
        if radius is None:
            radius = self.kspace.fermi_radius
        #r = self.kspace.fermi_radius
        n_points = 12*(1+n_refinements)

        vec1 = np.tile(self.lattice.vectors.vec1, n_points).reshape(n_points,3)
        vec2 = np.tile(self.lattice.vectors.vec2, n_points).reshape(n_points,3)
        opening_angle = self.kspace.symmetry.get_symmetry_cone_angle()
        woods_kvs = []

        #print(vec1, vec2)
        vlength = self.lattice.vectors.length1
        phis = np.linspace(0, np.pi*2., n_points+1)[:-1]

        circ_x = radius*np.cos(phis)
        circ_y = radius*np.sin(phis)
        circ_z = np.zeros(phis.shape)
        circ_points = np.vstack([circ_x, circ_y, circ_z]).T
        max_order = order**2
        order_groups, distances = self.lattice.orders_by_distance(max_order)
        group = order_groups[order-1]

        for row in range(group.shape[0]):
            order1 = group[row, 0]
            order2 = group[row, 1]
            shift = order1*vec1 + order2*vec2
            shifted_circ =  circ_points + shift
            if restrict_to_sym_cone:
                angle = np.angle(shifted_circ[:,0]+1j*shifted_circ[:,1])
                shifted_circ = shifted_circ[np.logical_and(angle >= 0. ,angle <= opening_angle), :]
            origin_distance = np.linalg.norm(shifted_circ, axis=1)
            woods_points = shifted_circ[ origin_distance<=self.kspace.fermi_radius, :]
            if woods_points.shape[0] == 0:
                continue
            n = self.kspace.fermi_radius / self.kspace.k0
            woods_kvs.append(self.kspace.convert_to_KVectors(woods_points, n, 1.))
        return woods_kvs


    """
    cos_factor = np.cos(np.radians(phi))
    sin_factor = np.sin(np.radians(phi))
    p = 2*(cos_factor*vec_total[0]+sin_factor*vec_total[1])
    q = (vec_total[0]**2+vec_total[1]**2-k_norm**2)
    for solution in [-1., 1.]:
        k_r1 = (-0.5*p) +solution*csqrt( (0.5*p)**2 -q)
        #k_r2 = (-0.5*p) + csqrt( (0.5*p)**2 -q)
        if (np.abs(np.imag(k_r1)) == 0.0 and
            0 <= np.real(k_r1) <= k_norm):
            kx = k_r1*np.cos(np.radians(phi))
            ky = k_r1*np.sin(np.radians(phi))
            kxy = np.array([kx, ky])
            already_included = False
            for point in woods_points:
                if (np.isclose(kxy[0], point[0]) and
                    np.isclose(kxy[1], point[1])):
                    already_included = True
            if not already_included:
                woods_points.append(kxy)
    """

    def sample_bloch_families(self, constraint=None, center=np.array([0., 0.]),
                              use_symmetry=True, cutoff_tol=1e-5,
                              restrict_to_sym_cone=False):
        """
        Return a point sampling of k-space in bloch families

        Parameters
        ----------
        constraint: dict
            constraints for determining the number of points in the sampling
        center: np.array
            center of sampling
        use_symmetry: bool
            make sample based on symmetry of unit cell
        cutoff_tol: float
            how close to the fermi_radius to exclude points from sampling
        restrict_to_sym_cone: bool


        Returns
        -------
        dict
            the sample points sorted into bloch families
        """
        families, all_points, sym_ops = self._bloch_fam_sampler(constraint, center,
                                                       use_symmetry, cutoff_tol,
                                                       restrict_to_sym_cone)
        return families

    def sample(self, constraint=None, center=np.array([0., 0.]),
                              use_symmetry=True, cutoff_tol=1e-5,
                              restrict_to_sym_cone=False):
        families, all_points, sym_ops = self._bloch_fam_sampler(constraint, center,
                                                       use_symmetry, cutoff_tol,
                                                       restrict_to_sym_cone)
        return all_points

    def _bloch_fam_sampler(self, constraint, center, use_symmetry, cutoff_tol,
                           restrict_to_sym_cone, sample_irreducible=False):
        """
        Return a point sampling of k-space in bloch families

        Parameters
        ----------
        constraint: dict
            constraints for determining the number of points in the sampling
        center: np.array
            center of the sampling
        use_symmetry: bool
            make sample based on symmetry of unit cell
        cutoff_tol: float
            how close to the fermi_radius to exclude points from sampling

        Returns
        -------
        dict
            the sample points sorted into bloch families
        (N,3) <np.double> np.array
            the sample points
        list
            sym_ops

        """
        if sample_irreducible:
            sampling, weighting, int_element, sym_ops = self.lattice.unit_cell.sample_irreducible(constraint=constraint,
                                                                                                  center=center)
        else:
            sampling, weighting, int_element = self.lattice.unit_cell.sample(constraint=constraint,
                                                                             center=center,
                                                                             use_symmetry=use_symmetry)
        sampling, weighting, symmetries = self.lattice.unit_cell.weight_and_sym_sample(sampling)
        return self._bloch_fam_from_sample(sampling, cutoff_tol, restrict_to_sym_cone,
                                           symmetries)

    def _bloch_fam_from_sample(self, sampling, cutoff_tol, restrict_to_sym_cone,
                               symmetries):

        angle0 = 0.0
        #opening_angle = self.kspace.symmetry.get_symmetry_cone_angle()
        n_sample_points = sampling.shape[0]
        if sampling.shape[1] == 2:
            sampling = np.hstack([sampling, np.zeros((sampling.shape[0], 1))])
        """
        longest_vector = 0.0
        for key, val in self.lattice.unit_cell.special_points.items():
            if np.linalg.norm(val) > longest_vector:
                longest_vector = np.linalg.norm(val)
        """
        n_unit_cells1 = int(np.ceil(self.kspace.fermi_radius/(0.5*self.lattice.vectors.length1)))
        #range1 = range(-n_unit_cells1, n_unit_cells1+1)
        n_unit_cells2 = int(np.ceil(self.kspace.fermi_radius/(0.5*self.lattice.vectors.length2)))
        n_max = np.max([n_unit_cells1, n_unit_cells2])
        #print("n_max: {}".format(n_max))
        #range2 = range(-n_unit_cells2, n_unit_cells2+1)
        all_points = []
        bloch_families = {}
        # family_syms = []
        sym_ops = []
        #symmetryFamilies = {}

        trans_sym = symmetry_from_type(PointSymmetry.T)
        trans_sym.vector1 = self.lattice.vectors.vec1
        trans_sym.vector2 = self.lattice.vectors.vec2
        counter = 0
        no_sym = symmetry_from_type(PointSymmetry.C1)
        try:
            reduced_sym = self.lattice.unit_cell.symmetry().stack[0]
        except:
            reduced_sym = no_sym
        reduced_sym = self.lattice.unit_cell.symmetry()
        use_symmetry = True
        for i_family in range(n_sample_points):
            central_point = sampling[i_family]



            bloch_family = []
            lattice_orders = []
            # is_sp = False
            # for sp_point in self.lattice.unit_cell.special_points.values():
            #     if np.linalg.norm(central_point[:2]-sp_point[:2]) < 1e-6:
            #             refl_rot_sym = symmetries[i_family][0]
            #             is_sp = True
            # if self.lattice.unit_cell.lies_on_ibz(central_point) and not is_sp:
            #     refl_rot_sym = symmetry_from_type(PointSymmetry.SIGMA_H) + symmetry_from_type(PointSymmetry.C1)
            # else:
            refl_rot_sym = symmetries[i_family][0]
            #trans_sym = symmetries[i_family][1]
            # sym_ops = []
            new_points, n1, n2 = trans_sym.apply_symmetry_operators(central_point, n=n_max, return_orders=True)

            # dummy_, keep = self.lattice.unit_cell.crop_to_bz(new_points, return_indices=True)
            # keep = np.logical_not(keep)
            # zeroth_order = np.where(np.logical_and(n1==0, n2==0))
            # keep[zeroth_order] = True
            # new_points = new_points[keep, :]
            # n1 = n1[keep]
            # n2 = n2[keep]

            #if self.lattice.unit_cell.lies_on_bz(central_point):
            #refl_rot_sym = reduced_sym #self.lattice.unit_cell.symmetry()


            new_points, keep = self.kspace.restrict_to_fermi_radius(new_points, tol=cutoff_tol,
                                                                    return_indices=True)
            n1 = n1[keep]
            n2 = n2[keep]

            #new_points, keep = self.kspace.restrict_to_reflection_cone(new_points, return_indices=True)
            #n1 = n1[keep]
            #n2 = n2[keep]

            #inside_bz = inside_bz[keep]
            if restrict_to_sym_cone:
                new_points, keep = self.kspace.restrict_to_sym_cone(new_points, return_indices=True)
                n1 = n1[keep]
                n2 = n2[keep]
            else:
                pass
                # dummy, inside_sym_cone = self.kspace.restrict_to_sym_cone(new_points, return_indices=True)
                # is_sp = False
                # for sp_point in self.lattice.unit_cell.special_points.values():
                #      if np.linalg.norm(central_point[:2]-sp_point[:2]) < 1e-6:
                #          is_sp =True
                         #central point on exterior

                #             refl_rot_sym = symmetries[i_family][0]
                #             is_sp = True
                #inside_bz = inside_bz[keep]

            all_points.append(new_points)
            sym_ops.append(refl_rot_sym)
            if new_points.shape[0]> 0:
                bloch_array = new_points
                kv_group = self.kspace.convert_to_KVectors(bloch_array, 1., 1)
                bloch_fam = BlochFamily.from_kvector_group(kv_group)
                bloch_fam.set_orders(n1, n2)
                bloch_families[i_family] = bloch_fam
            # for row in range(new_points.shape[0]):
            #     if inside_bz[row]:
            #         sym_ops.append(no_sym)
            #     else:
            #         sym_ops.append(refl_rot_sym)
            # family_syms.append(sym_ops)
        all_point_array = np.vstack(all_points)
        all_point_array = order_lexicographically(all_point_array)
        all_kvs = self.kspace.convert_to_KVectors(all_point_array, 1., 1.)

        return bloch_families, all_kvs, sym_ops

    def symmetrise_bloch_family(self, bloch_family:KVectorGroup,
                                ibz_symmetry, values=None, tolerance=1e-6):
        """
        apply symmetry opterators to a bloch family while avoiding duplicates
        """

        difference = self.lattice.unit_cell.symmetry()-ibz_symmetry[0]
        if values is None:
            sym_points = difference.apply_symmetry_operators(bloch_family.k)
        else:
            #value = values[keep]
            sym_points, sym_values = difference.apply_symmetry_operators(bloch_family.k,
                                                                         values=values)
            all_values = copy(values).tolist()
        all_points = copy(bloch_family.k).tolist()
        power = int(-1*np.log10(tolerance))
        test_set = set()
        for point in all_points:
            str_point1 = np.format_float_scientific(point[0], power)
            str_point2 = np.format_float_scientific(point[1], power)
            str_point = str_point1+str_point2
            test_set.add(hash(str_point))
        for row in range(sym_points.shape[0]):
            current_k = sym_points[row, :]
            str_point1 = np.format_float_scientific(current_k[0], power)
            str_point2 = np.format_float_scientific(current_k[1], power)
            str_point = str_point1+str_point2
            current_hash = hash(str_point)
            add_point = True
            for row2 in range(len(all_points)):
                if current_hash in test_set:
                    add_point = False
                    break
            if add_point:
                all_points.append(current_k)
                test_set.add(current_hash)

                if values is not None:
                    all_values.append(sym_values[row])
        all_point_array = np.vstack(all_points)
        all_point_array, sorting = order_lexicographically(all_point_array, return_sort_indices=True)
        all_kvs = self.kspace.convert_to_KVectors(all_point_array, 1., 1.)

        if values is None:
            return all_kvs
        else:
            all_values = np.vstack(all_values)
            all_values = all_values[sorting]
            return all_kvs, all_values

    def plotSymmetryFamilies(self,ax,n='all',color=None):
        if self.symmetry_families is None:
            self.calcKSampling()

        plt.sca(ax)

        if n=='all':
            nPoints = self.sampling.shape[0]
        else:
            nPoints =n

        if color == None:
            if nPoints<= 10:
                cmap = cm.get_cmap('tab10')
                NColors =10.0
            elif nPoints > 10 and nPoints <=20:
                cmap = cm.get_cmap('tab20')
                NColors =20.0
            elif nPoints > 20 and nPoints <=40:
                cmap = cm.get_cmap('tab20c')
                NColors =40.0
            else:
                cmap = cm.get_cmap('jet')
                NColors =float(nPoints)
            colors = cmap((np.arange(nPoints)/NColors))

        if color is None:
            for i in range(nPoints):
                family = self.symmetry_families[i][0]
                color = colors[i].reshape(1,4)
                plt.scatter(family[:,0],family[:,1],c=color)
        if color == "Frombloch_families":
            #nBFs = np.amax(np.array(list(self.bloch_families.keys())))
            #nBFs = len(self.bloch_families.keys())
            nBFs = self.brillouinZone.BZSampling.shape[0]
            for i in range(nPoints):
                family = self.symmetry_families[i][0]
                eldest = family[0,:]
                color = 'k'
                for key,bf in sorted(self.bloch_families.items()):
                    xCheck = np.isclose( eldest[0],bf[:,0],rtol=1e-6)
                    yCheck = np.isclose( eldest[1],bf[:,1],rtol=1e-6)
                    if np.any(xCheck*yCheck):
                        color = self.chooseColor(key,nBFs)
                plt.scatter(family[:,0],family[:,1],c=color)
        else:
            for i in range(nPoints):
                family = self.symmetry_families[i][0]
                plt.scatter(family[:,0],family[:,1],c=color)


def sum_triangles(xy, z, triangles):
    """ integrate scattered data, given a triangulation
    zsum, areasum = sumtriangles( xy, z, triangles )
    In:
        xy: npt, dim data points in 2d, 3d ...
        z: npt data values at the points, scalars or vectors
        triangles: ntri, dim+1 indices of triangles or simplexes, as from
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html
    Out:
        zsum: sum over all triangles of (area * z at midpoint).
            Thus z at a point where 5 triangles meet
            enters the sum 5 times, each weighted by that triangle's area / 3.
        areasum: the area or volume of the convex hull of the data points.
            For points over the unit square, zsum outside the hull is 0,
            so zsum / areasum would compensate for that.
            Or, make sure that the corners of the square or cube are in xy.
    """
    # z concave or convex => under or overestimates
    npt, dim = xy.shape
    ntri, dim1 = triangles.shape
    assert npt == len(z), "shape mismatch: xy %s z %s" % (xy.shape, z.shape)
    assert dim1 == dim+1, "triangles ? %s" % triangles.shape
    zsum = np.zeros( z[0].shape )
    areasum = 0
    dimfac = np.prod( np.arange( 1, dim+1 ))
    for tri in triangles:
        corners = xy[tri]
        t = corners[1:] - corners[0]
        if dim == 2:
            area = abs( t[0,0] * t[1,1] - t[0,1] * t[1,0] ) / 2
        else:
            area = abs( np.linalg.det( t )) / dimfac  # v slow
        zsum += area * z[tri].mean(axis=0)
        areasum += area
    return (zsum, areasum)

class Interpolator():

    """
    class to obtain k space sampling based on periodic structures

    lattice(Lattice): a reciprocal lattice used to generate the k-space sampling
    kspace(Kspace): reference to parent kspace
    """

    def __init__(self, kspace, sample_points, values):
        """
        Parameters
        ----------
        kspace: reciprocal.kspace.KSpace
            reference to parent kspace
        sample_points: reciprocal.kvector.KVectorGroup
            the points in kpsace to interpolate over
        values: <N,1> np.ndarray of floats
            the function values to interpolate
        """
        self.kspace = kspace
        self.sample_points = sample_points
        self.values = values

    def _triangulate(self):
        xi = self.sample_points.k[:,[0,1]]
        self.triangulation = scipy.spatial.Delaunay(xi)

    def _integrate(self):
        zsum, areasum = sum_triangles(self.sample_points.k[:,[0, 1]],
                                      self.values,
                                      self.triangulation.vertices)
        return zsum, areasum
