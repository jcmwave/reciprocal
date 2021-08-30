import numpy as np
from reciprocal.utils import (apply_symmetry_operators, lies_on_vertex, lies_on_poly,
                              name_vertices, lies_on_sym_line, rotation2D, rotation3D,
                              order_lexicographically)
from reciprocal.symmetry import Symmetry, SpecialPoint, PointSymmetry
from reciprocal.utils import BravaisLattice
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import itertools
from collections import OrderedDict
def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)



def make_perpendicular_lines(vec1, vec2):
    """
    return vertices of lines perpendicular to combinations of two vectors.

    Iterate over combinations of the two vectors. Perpendicular lines intersect
    the mid point of the vectors. The lines are ten times longer than the
    distance to the mid point (to ensure all nearby intersections of the lines
    can be found).

    Parameters
    ----------
    vec1 (3,) <np.double>np.array
        the first lattice vector
    vec2 (3,) <np.double>np.array
        the second lattice vector

    Returns
    -------
    list of (2,) tuple containing (float, (2,2) <np.double>np.array)
        the distance to the midpoint and vertices of the perpendicular line
    """
    # Create lines perpendicular to vectors pointing to closest lattice cites
    vertices = []
    for i_first in range(-1, 2):
        for i_second in range(-1, 2):
            if i_first == 0 and i_second == 0:
                continue

            vertex = i_first*vec1*0.5 + i_second*vec2*0.5
            distance = np.linalg.norm(vertex)
            perpendicular = np.cross(vertex, np.array([0., 0., 1.]))*10
            my_line = np.squeeze(np.array([[vertex+perpendicular],[vertex-perpendicular]]))
            vertices.append((distance, my_line))
    return vertices

def find_intersections(vertices, n_closest):
    """
    find intersections of lines defined through their vertices

    the parameter n_closest determines how many intersections to return. Only
    the intersections closest to the origin are returned.

    Parameters
    ----------
    vertices: list of (2,) tuple containing (float, (2,2) <np.double>np.array)
        vertices of the lines to calculated intersections for
    n_closest: int in range (1, 2)

    Return
    ------
    list of (2,) tuple containing (float, (2,2) <np.double>np.array)
        the intersection points between the lines
    """
    closest = np.full(n_closest, np.inf)
    intersections = []
    for vertex1, vertex2 in itertools.combinations(vertices, 2):
        L1 = line(vertex1[1][0, :], vertex1[1][1, :])
        L2 = line(vertex2[1][0, :], vertex2[1][1, :])
        inter = intersection(L1, L2)
        if inter is not False:
            my_intersection = np.array([inter[0], inter[1], 0.])
            distance = np.linalg.norm(my_intersection)
            for ic in range(n_closest):
                if distance < closest[ic]:
                    closest[ic] = distance
                    break
                if np.isclose(distance, closest[ic]):
                    break

            if np.all(distance > closest+1e-6):
                   continue
            intersections.append([distance, my_intersection])
    return intersections, closest

def split_intersections(intersections, closest):
    """
    split intersections into a set of closest points and all other points

    the parameter closest determines how many intersections to return. Only
    intersection points which are less than or equal to the distances in
    closest are returned.

    Parameters
    ----------
    intersections: list of (2,) tuple containing (float, (2,2) <np.double>np.array)
        distance and intersection position
    closest: (2,) or (1,) list
        unique distances to the closest points

    Return
    ------
    (N,2) <np.double>np.array
        the intersection points
    """
    final_intersections = []
    keep_intersections = []
    for i_inter, inter in enumerate(intersections):
        distance = inter[0]
        if np.any(distance> closest+1e-6):
            keep_intersections.append(inter)
            continue
        final_intersections.append(inter[1])

    return np.array(final_intersections)

def calc_max_extent(vertices):
    max_extent = 0.0
    for row in range(vertices.shape[0]):
        vertex = vertices[row, :]
        norm = np.linalg.norm(vertex)
        if norm > max_extent:
            max_extent = norm
    return max_extent

class UnitCell():
    """
    Defines the a unit cell of a 2D lattice

    Attributes
    ----------
    weigner_seitz: bool
        whether to construct a Wigner Seitz cell or not
    vectors: LatticeVectorssymmetry_points
        lattice vectors defining the unit cell
    vertices: (N,2)<np.double>np.array
        array of verex positions of the unit cell
    max_extent: float
        the largest distance between any vertex and the origin
    special_points: OrderedDict
        names and positions of special symmetry points

    irreducible: (N,2)<np.double>np.array
        vertices of the irreducible unit cell


    Methods
    -------
    area(self)
        return the area
    make_vertices(self)
        set the vertices attribute
    """
    def __init__(self, lattice, WignerSeitz=True):
        self.wigner_seitz = WignerSeitz
        self.lattice = lattice
        #self.vertices = vertices
        #is self.vertices is None:
        self.vertices = self.make_vertices()
        #if self.wigner_seitz:
        self.special_points = self.make_special_points()
        self.irreducible = self.make_irreducible_polygon()
        #else:
        #self.shape = 'general'
        #    self.vectors = None
        #    self.vertices = vertices
        self.max_extent = calc_max_extent(self.vertices)
        ##self.bravais_lattice = None
        #self.special_points = None

    @classmethod
    def from_vertices(unit_cell, vertices):
        """
        Return Unit Cell constructed from vertices

        typical usage:
        unit_cell = UnitCell.from_vertices(vertices)

        Parameters
        ----------
        vertices: (N,2) <np.double> np.array
            vertices of the unit cell

        Returns
        -------
        UnitCell
        """
        unit_cell.vertices = vertices
        return unit_cell


    @property
    def vectors(self):
        try:
            vec = self.lattice.vectors
        except:
            vec = self._vectors
        return vec

    @vectors.setter
    def vectors(self, vec):
        self._vectors = vec

    def area(self):
        """Return the area of the unit cell"""
        x = self.vertices[:,0]
        y = self.vertices[:,1]
        area =  0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
        return area

    def make_vertices(self):
        """Return the vertices of the unit cell

        Returns
        -------
        vertices: (N,2) <np.double> np.array
        """
        if self.wigner_seitz:
            vertices = self._make_wigner_seitz_cell()
        else:
            vertices = self._make_primitive_cell()
        return vertices

    def _make_primitive_cell(self):
        """
        Return the vertices of the primitive cell.

        Returns
        -------
        (N,3)<np.double>np.array
        """
        vertices = []
        vec1 = self.vectors.vec1
        vec2 = self.vectors.vec2
        vertices.append((vec1-vec2)*0.5)
        vertices.append((vec1+vec2)*0.5)
        vertices.append((-vec1+vec2)*0.5)
        vertices.append((-vec1-vec2)*0.5)
        vertices = np.array(vertices)
        angle_b1 = np.angle(vec1[0]+1j*vec1[1])
        vertices = order_lexicographically(vertices, start=-angle_b1)
        return vertices

    def _make_wigner_seitz_cell(self):
        """
        Return the vertices of the Wigner Seitz cell.

        Returns
        -------
        (N,3)<np.double>np.array
        """
        #vertices = []
        vec1 = self.vectors.vec1
        vec2 = self.vectors.vec2
        angle = self.vectors.angle
        vertices = make_perpendicular_lines(vec1, vec2)
        if np.isclose(self.vectors.length1, self.vectors.length2):
            n_closest = 1
        else:
            n_closest = 2
        intersections, closest = find_intersections(vertices, n_closest)
        intersections = split_intersections(intersections, closest)
        angle_b1 = np.angle(vec1[0]+1j*vec1[1])
        intersections = order_lexicographically(intersections, start=-angle_b1)
        return intersections

    def make_special_points(self):
        """
        determine the symmetry points of the unit cell

        requires the bravais lattice of parent lattice of to be set.

        Returns
        -------
        OrderedDict
            The special points and their positions
        """
        special_points = OrderedDict()
        special_points[SpecialPoint.GAMMA] = np.array([0., 0., 0.])
        vec1 = self.vectors.vec1
        vec2 = self.vectors.vec2
        length1 = self.vectors.length1
        length2 = self.vectors.length2
        angle = self.vectors.angle
        co_angle = 180. - angle
        #GM = np.linalg.norm(vec1)*0.5
        if self.lattice.bravais is BravaisLattice.HEXAGON:
            GM = length1*0.5
            special_points[SpecialPoint.K] = np.array([GM/np.cos(np.pi/6.0),0., 0.])
            special_points[SpecialPoint.M] = np.array([GM*np.cos(np.pi/6.0),
                                                   GM*np.sin(np.pi/6.0), 0.])
        elif self.lattice.bravais is BravaisLattice.SQUARE:
            special_points[SpecialPoint.X] = 0.5*vec1
            special_points[SpecialPoint.M] = 0.5*(vec1+vec2)
        elif self.lattice.bravais is  BravaisLattice.RECTANGLE:
            special_points[SpecialPoint.X] = 0.5*vec1
            special_points[SpecialPoint.M] = 0.5*(vec1+vec2)
            special_points[SpecialPoint.Y] = 0.5*vec2
        elif self.lattice.bravais is BravaisLattice.OBLIQUE:
            if self.wigner_seitz:
                special_points[SpecialPoint.X] = 0.5*vec1
                special_points[SpecialPoint.H1] = self.vertices[0]
                special_points[SpecialPoint.C] = 0.5*(vec1+vec2)
                special_points[SpecialPoint.H2] = self.vertices[1]
                special_points[SpecialPoint.Y1] = 0.5*vec2
                special_points[SpecialPoint.Y2] = -0.5*vec2
                special_points[SpecialPoint.H3] = self.vertices[-1]
            else:
                special_points[SpecialPoint.X] = 0.5*vec1
                special_points[SpecialPoint.H1] = 0.5*(vec1+vec2)
                special_points[SpecialPoint.Y1] = 0.5*vec2
                special_points[SpecialPoint.Y2] = -0.5*vec2
                special_points[SpecialPoint.H2] = 0.5*(vec1-vec2)
        else:
            raise ValueError("bravais lattice {}".format(self.lattice.bravais)+
                             " invalid")

        return special_points

    def make_irreducible_polygon(self):
        """
        return the vertices of the irreducible Brillouin zone

        Returns
        -------
        (N,3) <np.double> np.array
        """
        ipoly = []
        for key, val in self.special_points.items():
            if (self.lattice.bravais is BravaisLattice.OBLIQUE
                and key is SpecialPoint.GAMMA):
                continue
            ipoly.append(val)
        ipoly = np.vstack(ipoly)
        return ipoly

    def sample(self, constraint=None, shifted=False, use_symmetry=True):
        """
        Return a point sampling of the unit cell

        Parameters
        ----------
        constraint: dict
            constraints for determining the number of points in the sampling
        shifted: bool
            Shift the grid so the Gamma point is excluded from sampling
        use_symmetry: bool
            make sample based on symmetry of unit cell

        Returns
        -------
        (N,3) <np.double> np.array
        """
        if use_symmetry:
            return self._sample_using_symmetry(constraint, shifted)
        else:
            return self._sample_no_symmetry(constraint, shifted)

    def _sample_no_symmetry(self, constraint, shifted):
        unit_cell_path = Polygon(self.vertices[:,:2],
                                   closed=True).get_path()

        n_grid_points = self._npoints_from_constraint(constraint)

        if n_grid_points == 1:
            vec1 = np.array([0.0, 0.0, 0.0])
            vec2 = np.array([0.0, 0.0, 0.0])
        else:
            vec1 = np.array(self.vectors.vec1)
            vec2 = np.array(self.vectors.vec2)
            vec1 = (0.5/(n_grid_points[0]-1))*(vec1)
            vec2 = (0.5/(n_grid_points[1]-1))*(vec2)

        range_lim1 = n_grid_points[0]+int(n_grid_points[0]/2.0)
        range_lim2 = n_grid_points[1]+int(n_grid_points[1]/2.0)
        points = []
        for nx in range(-range_lim1, range_lim1):
            for ny in range(-range_lim2, range_lim2):
                trial_point = (nx*vec1 + ny*vec2)
                if shifted:
                    trial_point += (0.5*vec1 + 0.5*vec2)
                if not unit_cell_path.contains_point(trial_point,
                                                       radius=1e-7):
                    continue
                points.append(trial_point)
        points = np.vstack(points)
        return order_lexicographically(points)

        #self.sampling = ipoly_samp

    def _sample_using_symmetry(self, constraint, shifted):
        """
        Return a sampling of the unit cell while exploiting symmetry

        Parameters
        ----------
        constraint: dict
            constraints for determining the number of points in the sampling
        shifted: bool
            Shift the grid so the Gamma point is excluded from sampling

        Returns
        -------
        (N,3) <np.double> np.array
        """
        all_points = []
        symmetry_regions = self.symmetry_regions()
        irreducible_sampling = self.sample_irreducible(constraint=constraint,
                                                       shifted=shifted)

        for symmetry in symmetry_regions:
            if symmetry not in irreducible_sampling:
                continue
            for irr_point in range(irreducible_sampling[symmetry].shape[0]):
                point = irreducible_sampling[symmetry][irr_point, :2]
                symm = symmetry_regions[symmetry]
                all_points.append(apply_symmetry_operators(point, symm)[0])
        n_points = 0
        for i, *_ in enumerate(all_points):
            n_points += all_points[i].shape[0]
        all_points_array = np.zeros((n_points, 3))
        startrow = 0
        for i, *_ in enumerate(all_points):
            points = all_points[i]
            all_points_array[startrow:startrow+points.shape[0], :2] = points
            startrow += points.shape[0]
        all_points = all_points_array
        unique_points = np.unique(all_points.round(decimals=4), axis=0)
        points = order_lexicographically(unique_points)
        #point_array = np.vstack(points)
        return points
        #self.sampling = np.array(points)

    def _npoints_from_constraint(self, constraint):
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
            n_grid_points = [int(0.5*self.vectors.length1/density),
                             int(0.5*self.vectors.length2/density)]
        elif constraint['type'] == "n_points":
            try:
                n_grid_points = [constraint['value'][0], constraint['value'][1]]
            except:
                n_grid_points = [constraint['value'], constraint['value']]
        print("n grid points:")
        print(n_grid_points)
        if n_grid_points[0] == 0:
            n_grid_points[0] = 1
        if n_grid_points[1] == 0:
            n_grid_points[1] = 1
        return n_grid_points

    def sample_irreducible(self, constraint = None, shifted = False):
        """
        Return point sampling of the irreducible unit cell

        Parameters
        ----------
        constraint: dict
            constraints for determining the number of points in the sampling
        shifted: bool
            Shift the grid so the Gamma point is excluded from sampling

        Returns
        -------
        dict
            The symmetry regions with their associated k-space points

        """
        irreducible_path = Polygon(self.irreducible[:,:2],
                                   closed=True).get_path()


        n_grid_points = self._npoints_from_constraint(constraint)
        if n_grid_points[0] == 1:
            vec1 = np.array([0.0, 0.0, 0.0])
        else:
            vec1 = np.array(self.vectors.vec1)
            vec1 = (0.5/(n_grid_points[0]-1))*(vec1)

        if n_grid_points[1] == 1:
            vec2 = np.array([0.0, 0.0, 0.0])
        else:
            vec2 = np.array(self.vectors.vec2)
            vec2 = (0.5/(n_grid_points[1]-1))*(vec2)

        special_points = list(self.special_points.keys())
        special_points += [SpecialPoint.AXIS, SpecialPoint.INTERIOR]
        ipoly_samp = {}
        for special_point in special_points:
            ipoly_samp[special_point] = []

        range_lim1 = n_grid_points[0]+int(n_grid_points[0]/2.0)
        range_lim2 = n_grid_points[1]+int(n_grid_points[1]/2.0)

        for nx in range(-range_lim1, range_lim1):
            for ny in range(-range_lim2, range_lim2):
                trial_point = (nx*vec1 + ny*vec2)
                if shifted:
                    trial_point += (0.5*vec1 + 0.5*vec2)
                if not irreducible_path.contains_point(trial_point,
                                                       radius=1e-7):
                    continue
                on_vertex, special_point = lies_on_vertex(trial_point,
                                                           self.special_points)
                if on_vertex:
                    if  not special_point is SpecialPoint.Y2:
                        ipoly_samp[special_point].append(trial_point)
                elif lies_on_poly(trial_point, self.vertices):
                    ipoly_samp[SpecialPoint.AXIS].append(trial_point)
                else:
                    ipoly_samp[SpecialPoint.INTERIOR].append(trial_point)


        for special_point in special_points:
            if len(ipoly_samp[special_point]) > 0:
                ipoly_samp[special_point] = np.array(ipoly_samp[special_point])
            else:
                del ipoly_samp[special_point]


        return ipoly_samp
        #self.sampling = ipoly_samp

    def symmetry_regions(self):
        """
        Return the point symmetry of the special points of the unit cell

        Returns
        -------
        dict
            mapping of special point to point symmetry
        """
        symmetries = {}
        symmetries[SpecialPoint.GAMMA] = {BravaisLattice.HEXAGON:PointSymmetry.C_INF,
                                          BravaisLattice.SQUARE:PointSymmetry.C_INF,
                                          BravaisLattice.RECTANGLE:PointSymmetry.C_INF,
                                          BravaisLattice.OBLIQUE: PointSymmetry.C_INF}
        symmetries[SpecialPoint.K] = {BravaisLattice.HEXAGON:PointSymmetry.C2}
        symmetries[SpecialPoint.M]  = {BravaisLattice.HEXAGON:PointSymmetry.C3,
                                       BravaisLattice.SQUARE:PointSymmetry.C1,
                                       BravaisLattice.RECTANGLE:PointSymmetry.C1}
        symmetries[SpecialPoint.X] = {BravaisLattice.SQUARE:PointSymmetry.SIGMA_D,
                                      BravaisLattice.RECTANGLE:PointSymmetry.C1,
                                      BravaisLattice.OBLIQUE:PointSymmetry.C2}
        symmetries[SpecialPoint.Y] = {BravaisLattice.RECTANGLE:PointSymmetry.C1}
        symmetries[SpecialPoint.Y1] = {BravaisLattice.OBLIQUE: PointSymmetry.C1}
        symmetries[SpecialPoint.Y2] = {BravaisLattice.OBLIQUE: PointSymmetry.C1}
        symmetries[SpecialPoint.H1] = {BravaisLattice.OBLIQUE:PointSymmetry.C1}
        symmetries[SpecialPoint.H2] = {BravaisLattice.OBLIQUE:PointSymmetry.C1}
        symmetries[SpecialPoint.H3] = {BravaisLattice.OBLIQUE:PointSymmetry.C1}
        symmetries[SpecialPoint.C] = {BravaisLattice.OBLIQUE:PointSymmetry.C1}

        symmetries[SpecialPoint.AXIS] = {BravaisLattice.HEXAGON:PointSymmetry.C6,
                                         BravaisLattice.SQUARE:PointSymmetry.C4,
                                         BravaisLattice.RECTANGLE:PointSymmetry.C2,
                                         BravaisLattice.OBLIQUE:PointSymmetry.C1}
        symmetries[SpecialPoint.INTERIOR] = {BravaisLattice.HEXAGON:PointSymmetry.D6,
                                             BravaisLattice.SQUARE:PointSymmetry.D4,
                                             BravaisLattice.RECTANGLE:PointSymmetry.D2,
                                             BravaisLattice.OBLIQUE:PointSymmetry.C2}

        symmetry_regions = {}
        special_points = list(self.special_points.keys())
        special_points += [SpecialPoint.AXIS, SpecialPoint.INTERIOR]
        for s_point in special_points:
            if self.lattice.bravais not in symmetries[s_point]:
                continue
            symmetry_regions[s_point] = Symmetry(symmetries[s_point][self.lattice.bravais])
        return symmetry_regions



class BrillouinZone(UnitCell):
    """
    Extends the UnitCell class to describe a Brillouin Zone, i.e. the first
    unit cell of a reciprocal lattice.
    """
    def __init__(self, lattice_vectors):
        super(BrillouinZone, self).__init__(lattice_vectors, WignerSeitz=True)
        #self.symmetry_points = None
        self.irreducible = None
        #self.polygon = None
        #self.irreducible_sampling = None
        #self.sampling = None
        self.make_symmetry_points()
        self.make_ibzone()
        self.make_sampling()


    def make_ibzone(self):
        if self.symmetry_points is None:
            self.make_symmetry_points()

        ipoly = []
        for key_val in self.symmetry_points:
            if self.bravais_lattice == 'oblique' and key_val[0] == 'G':
                continue
            ipoly.append(key_val[1])
        ipoly = np.array(ipoly)
        self.irreducible = UnitCell(vertices=ipoly)
        self.irreducible.symmetry_points = self.symmetry_points
        self.irreducible.vectors = self.vectors
        self.irreducible.make_sampling()

    # def make_polygon(self):
    #     if self.symmetry_points is None:
    #         self.make_symmetry_points()
    #
    #     ext_sym_points = {}
    #     for keyVal in self.symmetry_points:
    #         if keyVal[0] == 'G':
    #             continue
    #         sym_points = apply_symmetry_operators(keyVal[1],self.symmetry)[0]
    #         ext_sym_points[keyVal[0]] = sym_points
    #
    #     BZ = []
    #     for key in ext_sym_points.keys():
    #         if key == 'G':
    #             continue
    #         for iP in range(len(ext_sym_points[key])):
    #             BZ.append(ext_sym_points[key][iP])
    #     BZ = np.array(BZ)
    #     sortVal = np.arctan2(BZ[:,1],BZ[:,0])
    #     sortKey = np.argsort(sortVal)
    #     BZ = BZ[sortKey, :]
    #     self.polygon = BZ


    # def make_irreducible_sampling(self, constraint):
    #     if self.irreducible_polygon is None:
    #         self.make_irreducible_polygon()
    #
    #     irreducible_path = Polygon(self.irreducible_polygon,
    #                                closed=True).get_path()
    #
    #     #create array of points in the 1st BZ
    #     #normalise reciprocal vectors to the 1st BZ
    #     if constraint['type'] == "density":
    #         density = constraint['value']
    #         vec1 = (density*(self.vectors.vec1[:2])/
    #                 np.linalg.norm(self.vectors.vec1[:2]))
    #         vec2 = (density*(self.vectors.vec2[:2])/
    #                 np.linalg.norm(self.vectors.vec2[:2]))
    #         n_grid_points = int(0.5*np.linalg.norm(self.vectors.vec1[:2])/density)
    #     elif constraint['type'] == "n_points":
    #         n_grid_points = constraint['value']
    #
    #     if n_grid_points == 0:
    #         n_grid_points = 1
    #
    #
    #     if n_grid_points == 1:
    #         vec1 = np.array([0.0, 0.0])
    #         vec2 = np.array([0.0, 0.0])
    #     else:
    #         vec1 = (0.5/(n_grid_points-1))*(self.vectors.vec1[:2])
    #         vec2 = (0.5/(n_grid_points-1))*(self.vectors.vec2[:2])
    #
    #     ipoly_samp = {}
    #     ipoly_samp['G'] = []
    #     ipoly_samp['M'] = []
    #     ipoly_samp['X'] = []
    #     ipoly_samp['Y'] = []
    #     ipoly_samp['K'] = []
    #     ipoly_samp['onSymmetryAxis'] = []
    #     ipoly_samp['interior'] = []
    #     for nx in range(0, n_grid_points+int(n_grid_points/2.0)):
    #         for ny in range(0, n_grid_points+int(n_grid_points/2.0)):
    #             trial_point = nx*vec1 + ny*vec2
    #             if not irreducible_path.contains_point(trial_point,
    #                                                    radius=1e-7):
    #                 continue
    #             on_vertex, symmetry_point = lies_on_vertex(trial_point,
    #                                                        self.symmetry_points)
    #             if on_vertex:
    #                 ipoly_samp[symmetry_point].append(trial_point)
    #             elif lies_on_poly(trial_point, self.irreducible_polygon):
    #                 ipoly_samp['onSymmetryAxis'].append(trial_point)
    #             else:
    #                 ipoly_samp['interior'].append(trial_point)
    #
    #     point_types = ['G', 'M', 'K', 'X', 'Y', 'onSymmetryAxis', 'interior']
    #     for point_type in point_types:
    #         if len(ipoly_samp[point_type]) > 0:
    #             ipoly_samp[point_type] = np.array(ipoly_samp[point_type])
    #         else:
    #             del ipoly_samp[point_type]
    #     self.irreducible_sampling = ipoly_samp

    def make_sampling(self):
        all_points = []
        symmetry_regions = self.symmetry_regions()
        for symmetry in symmetry_regions:
            if symmetry not in self.irreducible.sampling:
                continue
            for irr_point in range(self.irreducible.sampling[symmetry].shape[0]):

                point = self.irreducible.sampling[symmetry][irr_point, :]
                symm = symmetry_regions[symmetry]
                all_points.append(apply_symmetry_operators(point, symm)[0])
        n_points = 0
        for i, *_ in enumerate(all_points):
            n_points += all_points[i].shape[0]
        all_points_array = np.zeros((n_points, 2))
        startrow = 0
        for i, *_ in enumerate(all_points):
            points = all_points[i]
            all_points_array[startrow:startrow+points.shape[0], :] = points
            startrow += points.shape[0]
        all_points = all_points_array
        unique_points = np.unique(all_points.round(decimals=4), axis=0)
        points = order_lexicographically(unique_points)
        #unique_rows = np.unique(all_points.round(decimals=4), axis=0)
        #sortkey = np.arctan2(unique_rows[:,1], unique_rows[:,0])
        #negatives = sortkey < 0
        #sortkey[negatives] += 2*np.pi
        #sort_indices = np.argsort(sortkey)
        #points = unique_rows[sort_indices]

        #BZPath = Polygon(self.BZ,closed=True).get_path()
        #for point in points:
        #    angle = np.arctan2(point[1], point[0])

        self.sampling = np.array(points)
