import numpy as np
import matplotlib as mpl
from matplotlib import cm
from matplotlib.patches import Wedge, Circle, Rectangle
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from numpy.lib.scimath import sqrt as csqrt
import scipy.spatial
from abc import ABC
from reciprocal.symmetry import Symmetry, SpecialPoint
from reciprocal.utils import (apply_symmetry_operators, order_lexicographically,
                              lies_on_poly, lies_on_vertex)
from reciprocal.kvector import KVectorGroup, BlochFamily
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
            n_sSiblings = family.k.shape[0]
        n_points_total += nSiblings
        family_sizes[key] = nSiblings
    info['NKPoints'] = n_points_total
    info['FamilySizes'] = family_sizes
    info['Speedup'] = info['NKPoints']/info['NFamilies']
    return info


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
            gamma_vertex = {SpecialPoint.GAMMA:np.array([0., 0., 0.])}
            on_vertex, special_point = lies_on_vertex(kxy, gamma_vertex)
            if on_vertex:
                points = np.array([[0., 0.]])
            elif lies_on_poly(kxy, self.symmetry_cone, closed=False):
                points, operators = apply_symmetry_operators(kxy,
                                                             self.symmetry.reduce())
            else:
                points, operators = apply_symmetry_operators(kxy, self.symmetry)
            for row in range(points.shape[0]):
                point = points[row, :]
                symmetry_groups[row].append(point)
        for i_sym, point_list in enumerate(symmetry_groups):
            point_array = np.vstack(point_list)
            kvs = self.convert_to_KVectors(point_array, 1.0, 1)
            symmetry_groups[i_sym] = kvs
        return symmetry_groups

    def convert_to_KVectors(self, points, n, direction):
        """
        convert a (N,2) np.array of kx,ky values into a KVectorGroup object.
        """
        nRows = points.shape[0]
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

    def sample(self, grid_type='cartesian', constraint=None, shifted=False,
               cutoff_tol=1e-5, restrict_to_sym_cone=False):
        if grid_type == 'cartesian':
            all_points = self._sample_cartesian(constraint, shifted, cutoff_tol,
                                           restrict_to_sym_cone)
        if grid_type == 'circular':
            all_points = self._sample_circular(constraint, shifted, cutoff_tol,
                                           restrict_to_sym_cone)
        return all_points

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
        for ii, gp in enumerate(n_grid_points):
            if gp == 0:
                n_grid_points[ii] = 1
        return n_grid_points

    def _sample_circular(self, constraint, shifted, cutoff_tol,
                          restrict_to_sym_cone):
        vector1 = np.array([self.kspace.fermi_radius, 0.])
        vector_lengths = np.array([self.kspace.fermi_radius])

        if restrict_to_sym_cone:
            opening_angle = self.kspace.symmetry.get_symmetry_cone_angle()
        n_grid_points = self._npoints_from_constraint(vector_lengths, constraint)
        radial_range = range(0, n_grid_points[0])

        all_points = []
        weighting = []
        artists = []
        rad_spacing = vector_lengths[0]/(n_grid_points[0]-0.5)

        if shifted:
            raise ValueError("shifted no supported for circular grids")

        total_area = np.pi*self.kspace.fermi_radius**2

        for n_r in radial_range:
            if n_r == 0:
                all_points.append(np.array([0., 0.]))
                center_circle_area = np.pi*(rad_spacing/2.)**2
                weighting.append(center_circle_area/total_area)
                artists.append(Circle((0.,0.), radius=rad_spacing/2., fill=False, edgecolor='k'))
                continue
            radius = rad_spacing*n_r
            upper_radius = rad_spacing*(n_r+0.5)
            lower_radius = rad_spacing*(n_r-0.5)
            circumference = np.pi*2*radius
            n_phis = self._npoints_from_constraint([circumference], constraint)
            phis = np.linspace(0, 360, n_phis[0])[:-1]
            if phis.size == 1:
                phi_spacing = 360.
            else:
                phi_spacing = phis[1]-phis[0]
            if restrict_to_sym_cone:
                phis = phis[phis<=opening_angle]
            for phi in phis:
                x = radius*np.cos(np.radians(phi))
                y = radius*np.sin(np.radians(phi))
                trial_point = np.array([x, y])
                length = np.linalg.norm(trial_point)
                if length > self.kspace.fermi_radius*(1-cutoff_tol):
                    pass
                    #continue
                wedge_area = 0.5*(upper_radius**2-lower_radius**2)*np.radians(phi_spacing)
                weighting.append(wedge_area/total_area)
                all_points.append(trial_point)
                wedge = Wedge(np.array([0., 0.]), upper_radius,
                              phi-phi_spacing*0.5, phi+phi_spacing*0.5,
                              width=upper_radius-lower_radius, fill=False)
                artists.append(wedge)

        all_point_array = np.vstack(all_points)
        all_point_array = order_lexicographically(all_point_array)
        all_kvs = self.kspace.convert_to_KVectors(all_point_array, 1., 1.)
        weighting_array = np.array(weighting)
        return all_kvs, weighting_array

    def _sample_cartesian(self, constraint, shifted, cutoff_tol,
                         restrict_to_sym_cone):
        vector1 = np.array([self.kspace.fermi_radius, 0.])
        vector2 = np.array([0., self.kspace.fermi_radius])
        vector_lengths = np.array([self.kspace.fermi_radius, self.kspace.fermi_radius])
        if restrict_to_sym_cone:
            opening_angle = self.kspace.symmetry.get_symmetry_cone_angle()
        n_grid_points = self._npoints_from_constraint(vector_lengths, constraint)
        range1 = range(-n_grid_points[0], n_grid_points[0]+1)
        range2 = range(-n_grid_points[1], n_grid_points[1]+1)

        all_points = []
        weighting = []
        artists = []
        total_area = np.pi*self.kspace.fermi_radius**2
        vector1 /= n_grid_points[0]
        vector2 /= n_grid_points[1]
        vector_lengths /= n_grid_points

        if shifted:
            central_point = 0.5*np.array([vector1, vector2])
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
                artists.append(square)
                weighting.append(vector_lengths[0]*vector_lengths[1]/total_area)
                all_points.append(trial_point)

        all_point_array = np.vstack(all_points)
        all_point_array = order_lexicographically(all_point_array)
        all_kvs = self.kspace.convert_to_KVectors(all_point_array, 1., 1.)
        weighting_array = np.array(weighting)
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
        r = self.kspace.fermi_radius
        n_points = 12*(1+n_refinements)

        vec1 = np.tile(self.lattice.vectors.vec1, n_points).reshape(n_points,3)
        vec2 = np.tile(self.lattice.vectors.vec2, n_points).reshape(n_points,3)
        opening_angle = self.kspace.symmetry.get_symmetry_cone_angle()
        woods_kvs = []

        #print(vec1, vec2)
        vlength = self.lattice.vectors.length1
        phis = np.linspace(0, np.pi*2., n_points+1)[:-1]

        circ_x = r*np.cos(phis)
        circ_y = r*np.sin(phis)
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
            woods_points = shifted_circ[ origin_distance<=r, :]
            if woods_points.shape[0] == 0:
                continue
            woods_points = order_lexicographically(woods_points)
            woods_kvs.append(self.kspace.convert_to_KVectors(woods_points, 1., 1.))
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

    def sample_bloch_families(self, constraint=None, shifted=False,
                              use_symmetry=True, cutoff_tol=1e-5,
                              restrict_to_sym_cone=False):
        """
        Return a point sampling of k-space in bloch families

        Parameters
        ----------
        constraint: dict
            constraints for determining the number of points in the sampling
        shifted: bool
            Shift the grid so the Gamma point is excluded from sampling
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
        families, all_points = self._bloch_fam_sampler(constraint, shifted,
                                                       use_symmetry, cutoff_tol,
                                                       restrict_to_sym_cone)
        return families

    def sample(self, constraint=None, shifted=False,
                              use_symmetry=True, cutoff_tol=1e-5,
                              restrict_to_sym_cone=False):
        families, all_points = self._bloch_fam_sampler(constraint, shifted,
                                                       use_symmetry, cutoff_tol,
                                                       restrict_to_sym_cone)
        return all_points

    def _bloch_fam_sampler(self, constraint, shifted, use_symmetry, cutoff_tol,
                           restrict_to_sym_cone):
        """
        Return a point sampling of k-space in bloch families

        Parameters
        ----------
        constraint: dict
            constraints for determining the number of points in the sampling
        shifted: bool
            Shift the grid so the Gamma point is excluded from sampling
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
        """
        sampling = self.lattice.unit_cell.sample(constraint=constraint,
                                                 shifted=shifted,
                                                 use_symmetry=use_symmetry)

        angle0 = 0.0
        opening_angle = self.kspace.symmetry.get_symmetry_cone_angle()
        n_sample_points = sampling.shape[0]
        """
        longest_vector = 0.0
        for key, val in self.lattice.unit_cell.special_points.items():
            if np.linalg.norm(val) > longest_vector:
                longest_vector = np.linalg.norm(val)
        """
        n_unit_cells1 = int(np.ceil(self.kspace.fermi_radius/(0.5*self.lattice.vectors.length1)))
        range1 = range(-n_unit_cells1, n_unit_cells1+1)
        n_unit_cells2 = int(np.ceil(self.kspace.fermi_radius/(0.5*self.lattice.vectors.length2)))
        range2 = range(-n_unit_cells2, n_unit_cells2+1)
        all_points = []
        bloch_families = {}
        symmetryFamilies = {}

        vec1 = self.lattice.vectors.vec1
        vec2 = self.lattice.vectors.vec2
        counter = 0
        for i_family in range(n_sample_points):
            central_point = sampling[i_family]
            bloch_family = []
            lattice_orders = []
            for nx in range1:
                for ny in range2:
                    trial_point = nx*vec1 + ny*vec2 + central_point

                    length = np.linalg.norm(trial_point)
                    if length > self.kspace.fermi_radius*(1-cutoff_tol):
                        continue

                    if not use_symmetry:
                        # This takes very long
                        if test_for_duplicates(all_points, trial_point):
                            continue

                    if restrict_to_sym_cone:
                        if self.test_outside_symmetry_cone(trial_point,
                                                          opening_angle):
                            continue




                    all_points.append(trial_point)
                    bloch_family.append(trial_point)
                    lattice_orders.append([nx, ny])
                    counter += 1

            if len(bloch_family) > 0:
                bloch_array = np.vstack(bloch_family)
                lattice_orders = np.vstack(lattice_orders)
                kv_group = self.kspace.convert_to_KVectors(bloch_array, 1., 1)
                bloch_fam = BlochFamily.from_kvector_group(kv_group)
                bloch_fam.set_orders(lattice_orders[:,0], lattice_orders[:,1])
                bloch_families[i_family] = bloch_fam
        all_point_array = np.vstack(all_points)
        all_point_array = order_lexicographically(all_point_array)
        all_kvs = self.kspace.convert_to_KVectors(all_point_array, 1., 1.)

        return bloch_families, all_kvs



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
