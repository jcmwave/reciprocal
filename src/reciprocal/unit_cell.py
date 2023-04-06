import numpy as np
from reciprocal.utils import (apply_symmetry_operators, lies_on_vertex, lies_on_poly,
                              name_vertices, lies_on_sym_line, rotation2D, rotation3D,
                              order_lexicographically)
from reciprocal.symmetry import Symmetry, SpecialPoint, PointSymmetry, symmetry_from_type
from reciprocal.utils import BravaisLattice
#import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import itertools
from itertools import tee
from collections import OrderedDict
import copy
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
    list of (2,) tuple containing (float, (2,3) <np.double>np.array)
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
            distance = np.round(np.linalg.norm(my_intersection), 9)
            for ic in range(n_closest):
                if np.isclose(distance, closest[ic], rtol=1e-6):
                    break
                if distance < closest[ic]:
                    closest[ic] = distance
                    break
            if np.all(distance > closest*(1.)):
                continue
            valid = True
            for i_inter in range(len(intersections)):
                if (np.isclose(my_intersection[0], intersections[i_inter][1][0]) and
                    np.isclose(my_intersection[1], intersections[i_inter][1][1])):
                        valid = False
                        break
            if not valid:
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
        if np.any(distance> closest*(1+1e-9)):
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
        #x = self.vertices[:,0]
        #y = self.vertices[:,1]
        x = self.lattice.vectors.vec1
        y = self.lattice.vectors.vec2
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

    def extend_special_points(self):
        if self.special_points is None:
            raise ValueError("special points must first be set")
        t_syms = self.translational_symmetries()
        extended_special_points = OrderedDict()        
        for special_point, symmetry in self.special_points.items():
            if special_point not in t_syms:
                continue            
            point = self.special_points[special_point]
            extended_points = t_syms[special_point].apply_symmetry_operators(point)
            extended_special_points[special_point] = np.atleast_2d(extended_points)
        return extended_special_points

    # def vertex_special_points(self, extended_special_points):
    #     vertex_special_points = OrderedDict()
    #     vertex_points = {BravaisLattice.HEXAGON:[SpecialPoint.K],
    #                      BravaisLattice.SQUARE:[SpecialPoint.M],
    #                      BravaisLattice.RECTANGLE:[SpecialPoint.M],
    #                      BravaisLattice.OBLIQUE:[SpecialPoint.H1, SpecialPoint.H2,
    #                                              SpecialPoint.H3]}

        
        
    #     for name, point in extended_special_points.items():
    #         if name in vertex_points[self.lattice.bravais]:                
    #             vertex_special_points[name] = point
    #     return vertex_special_points
        

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

        if n_grid_points[0] == 1:
            vec1 = np.array([0.0, 0.0, 0.0])
            vec2 = np.array([0.0, 0.0, 0.0])
        else:
            vec1 = np.array(self.vectors.vec1)
            vec2 = np.array(self.vectors.vec2)
            vec1 = (0.5/(n_grid_points[0]-1))*(vec1)
            vec2 = (0.5/(n_grid_points[1]-1))*(vec2)

        #range_lim1 = n_grid_points[0]+int(n_grid_points[0]/2.0)
        #range_lim2 = n_grid_points[1]+int(n_grid_points[1]/2.0)
        #range1 = np.linspace(-(n_grid_points[0]-1), n_grid_points[0]-1, n_grid_points[0]*2-1, dtype=np.int64)
        #range2 = np.linspace(-(n_grid_points[1]-1), n_grid_points[1]-1, n_grid_points[1]*2-1, dtype=np.int64)
        range_lim1 = n_grid_points[0]+int(n_grid_points[0]/2.0)
        range_lim2 = n_grid_points[1]+int(n_grid_points[1]/2.0)
        if n_grid_points[0] == 1:
            range1 = np.arange(0, range_lim1, 1, dtype=np.int64)
            range2 = np.arange(0, range_lim2, 1, dtype=np.int64)
        else:
            range1 = np.arange(-range_lim1, range_lim1, 1, dtype=np.int64)
            range2 = np.arange(-range_lim2, range_lim2, 1, dtype=np.int64)
        
        
        points = []
        for nx in range1:
            for ny in range2:
                trial_point = (nx*vec1 + ny*vec2)
                if shifted:
                    trial_point += (0.5*vec1 + 0.5*vec2)
                if not unit_cell_path.contains_point(trial_point,
                                                       radius=1e-7):
                    continue
                points.append(trial_point)
        points = np.vstack(points)
        points = order_lexicographically(points)
        weights = self.weight_bz_sampling(points)
        int_element = self.integration_element(constraint)/self.area()
        return points, weights, int_element

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
        sample = self.sample_irreducible(constraint=constraint,
                                         shifted=shifted)
        irreducible_sampling = sample[0]
        weighting = sample[1]
        int_element = sample[2]
        sym_ops = sample[3]
        #print("sym ops: {}".format(sym_ops))
        all_points = []
        for row in range(irreducible_sampling.shape[0]):
            point = np.concatenate([irreducible_sampling[row, :], np.array([0.])])            
            point_symmetry = sym_ops[row][0]
            translation_symmetry = sym_ops[row][1]            
            remaining_symmetry = self.symmetry()-point_symmetry
            refl_rot_points = remaining_symmetry.apply_symmetry_operators(point)
            
            to_keep = np.ones(refl_rot_points.shape[0],dtype=bool)
            if translation_symmetry is not None:
                for row2 in range(refl_rot_points.shape[0]):
                    if to_keep[row2] is False:
                        continue
                    point2 = refl_rot_points[row2, :]
                    translated = translation_symmetry.apply_symmetry_operators(point2)
                    for row3 in range(translated.shape[0]):
                        if row3 == 0:
                            continue
                        point3 = translated[row3, :]
                        for row4 in range(refl_rot_points.shape[0]):
                            point4 = refl_rot_points[row4, :]
                            if np.all(np.isclose(point4, point3)):
                                print(point4)
                                to_keep[row4] =False
            all_points.append(refl_rot_points[to_keep, :])
        n_points = 0
        for i, *_ in enumerate(all_points):
            n_points += all_points[i].shape[0]
        all_points_array = np.zeros((n_points, 3))
        startrow = 0
        for i, *_ in enumerate(all_points):
            points = all_points[i]
            all_points_array[startrow:startrow+points.shape[0], :] = points
            startrow += points.shape[0]
        all_points = all_points_array
        unique_points = np.unique(all_points.round(decimals=4), axis=0)
        points = order_lexicographically(unique_points)
        #weights = self.weight_bz_sampling(points)
        weights = np.ones(points.shape[0])*int_element
        #point_array = np.vstack(points)
        return points, weights, int_element
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
        if constraint['type'] == "max_length":
            max_length = constraint['value']
            n_grid_points = [int(0.5*self.vectors.length1/max_length),
                             int(0.5*self.vectors.length2/max_length)]
        elif constraint['type'] == "n_points":
            try:
                n_grid_points = [constraint['value'][0], constraint['value'][1]]
            except:
                n_grid_points = [constraint['value'], constraint['value']]
        if n_grid_points[0] == 0:
            n_grid_points[0] = 1
        if n_grid_points[1] == 0:
            n_grid_points[1] = 1
        #print(n_grid_points)
        return n_grid_points

    def _max_lengths_from_constraint(self, constraint):
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
        if constraint['type'] == "max_length":
            max_lengths = np.array([constraint['value'], constraint['value']])
        elif constraint['type'] == "n_points":
            try:
                n_grid_points = [constraint['value'][0], constraint['value'][1]]
            except:
                n_grid_points = [constraint['value'], constraint['value']]
            # if n_grid_points[0] == 0:
            #     n_grid_points[0] = 1
            # if n_grid_points[1] == 0:
            #     n_grid_points[1] = 1
            if n_grid_points[0] == 1:
                max_length1 = self.vectors.length1
            else:
                max_length1 = self.vectors.length1/(n_grid_points[0]*2-2)
            if n_grid_points[1] == 1:
                max_length2 = self.vectors.length2
            else:
                max_length2 = self.vectors.length2/(n_grid_points[1]*2-2)                
            max_lengths = np.array([max_length1, max_length2])
        #print(n_grid_points)
        return max_lengths

    def integration_element(self, constraint):
        max_lengths= self._max_lengths_from_constraint(constraint)
        x = self.lattice.vectors.vec1*(max_lengths[0]/self.lattice.vectors.length1)
        y = self.lattice.vectors.vec2*(max_lengths[1]/self.lattice.vectors.length2)
        area =  0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
        return area

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

        irreducible_vertices = np.hstack([irreducible_path.vertices, np.zeros((irreducible_path.vertices.shape[0], 1))])
        n_grid_points = self._npoints_from_constraint(constraint)                                   
        max_lengths = self._max_lengths_from_constraint(constraint)
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
        special_points += [SpecialPoint.AXIS, SpecialPoint.EXTERIOR, SpecialPoint.INTERIOR]
        ipoly_samp = {}
        for special_point in special_points:
            ipoly_samp[special_point] = []

        #range1 = np.linspace(-(n_grid_points[0]-1), n_grid_points[0]-1, n_grid_points[0]*2-1, dtype=np.int64)
        #range2 = np.linspace(-(n_grid_points[1]-1), n_grid_points[1]-1, n_grid_points[1]*2-1, dtype=np.int64)

        range_lim1 = n_grid_points[0]
        range_lim2 = n_grid_points[1]
        
        extended_range_lim1 = range_lim1+int(n_grid_points[0]/2.0)
        extended_range_lim2 = range_lim2+int(n_grid_points[1]/2.0)

        range1 = np.arange(-extended_range_lim1, extended_range_lim1, 1, dtype=np.int64)
        range2 = np.arange(-extended_range_lim2, extended_range_lim2, 1, dtype=np.int64)        

        int_element = self.integration_element(constraint)/self.area()
        #all_points = []
        #print(range1, range2)
        #print(vec1, vec2)
        #for nx in range(-range_lim1, range_lim1):
        #    for ny in range(-range_lim2, range_lim2):
        #weighting = []
        for nx in range1:
            for ny in range2:        
                #print("nx {}, ny {}".format(nx, ny))
                trial_point = (nx*vec1 + ny*vec2)
                #print("trial point: {}".format(trial_point))
                if shifted:
                    trial_point += ((1./2.)*vec1 + (1./2.)*vec2)
                if not irreducible_path.contains_point(trial_point,
                                                       radius=1e-7):
                    continue
                on_vertex, special_point = lies_on_vertex(trial_point,
                                                           self.special_points)
                if on_vertex:
                    if not special_point is SpecialPoint.Y2:
                        ipoly_samp[special_point].append(trial_point)
                elif lies_on_poly(trial_point, self.vertices):
                    ipoly_samp[SpecialPoint.EXTERIOR].append(trial_point)                        
                elif lies_on_poly(trial_point, irreducible_vertices):
                    ipoly_samp[SpecialPoint.AXIS].append(trial_point)
                else:
                    ipoly_samp[SpecialPoint.INTERIOR].append(trial_point)
                #all_points.append(trial_point)
        #print(ipoly_samp)
        for special_point in special_points:
            if len(ipoly_samp[special_point]) > 0:
                ipoly_samp[special_point] = np.array(ipoly_samp[special_point])
            else:
                del ipoly_samp[special_point]
        #all_points = np.vstack(all_points)
        all_points = []
        all_sym_ops = []
        all_weights = []
        refl_rot_syms = self.refl_rot_symmetries()
        trans_syms = self.translational_symmetries()
        total_sym = self.symmetry()
        #print(trans_syms)
        #print(ipoly_samp)
        for symmetry in refl_rot_syms:
            if symmetry not in ipoly_samp:
                continue
            if symmetry in trans_syms:
                bloch_symmetries_in_bz = trans_syms[symmetry].get_n_symmetry_ops()
            else:
                bloch_symmetries_in_bz = 1
            bloch_weight = 1./bloch_symmetries_in_bz
            symm = refl_rot_syms[symmetry]
            n_refl_rot = (total_sym-symm).get_n_symmetry_ops()
            total_weight =bloch_weight*n_refl_rot/total_sym.get_n_symmetry_ops()

            for row in range(ipoly_samp[symmetry].shape[0]):                
                point = ipoly_samp[symmetry][row, :2]
                all_points.append(point)
                all_weights.append(total_weight)
                #remaining_symmetry = self.symmetry()-symm
                #print("remaining_symmetry: {}".format(remaining_symmetry))
                if symmetry in trans_syms:
                    all_sym_ops.append((symm, trans_syms[symmetry]))
                else:
                    all_sym_ops.append((symm, None))
        all_points = np.vstack(all_points)
        all_weights = np.array(all_weights)
        return all_points, all_weights, int_element, all_sym_ops
        #self.sampling = ipoly_samp

    def weight_sym_ops(self, sym_ops):
        weighting = np.zeros(len(sym_ops))
        #max_sym = self.symmetry()
        for ii, sym in enumerate(sym_ops):
            #weight = sym.get_n_symmetry_ops()/max_sym
            remaining_sym = self.symmetry()-sym
            #if np.isinf(weight):
            #    weight = 1.0/(max_sym*2)
            #print(remaining_sym)
            weighting[ii] = (remaining_sym.get_n_symmetry_ops()/self.symmetry().get_n_symmetry_ops())
        #weighting/np.sum(weighting)
        return weighting

    def weight_bz_sampling(self, points):
        #symmetry_regions = self.symmetry_regions()
        t_syms = self.translational_symmetries()
        ext_sp_points = self.extend_special_points()
        #vertex_points = self.vertex_special_points(ext_sp_points)
        weights = np.zeros(points.shape[0])
        for row in range(points.shape[0]):
            trial_point = points[row, :]            
            on_vertex, special_point = lies_on_vertex(trial_point,
                                                      ext_sp_points)
            if on_vertex:
                n_sym_ops = t_syms[special_point].get_n_symmetry_ops()
                weights[row] = 1./n_sym_ops
            elif lies_on_poly(trial_point, self.vertices):
                n_sym_ops = t_syms[SpecialPoint.EXTERIOR].get_n_symmetry_ops()
                weights[row] = 1./n_sym_ops
            else:
                weights[row] = 1.0
        #weights /= weights.size
        return weights
                
    def weight_irreducible_sampling(self, ipoly_sample):
        n_points_total = 0
        weights = []
        all_points = []
        max_sym = self.symmetry().get_n_symmetry_ops()
        sym_regions = self.symmetry_regions()
        for region, points in ipoly_sample.items():
            n_points = points.shape[0]
            n_points_total += n_points
            point_symmetry = sym_regions[region]
            weight = point_symmetry.get_n_symmetry_ops()/max_sym            
            if np.isinf(weight):
                weight = 1.0/(max_sym*2)
            #print(point_symmetry, point_symmetry.get_n_symmetry_ops(), weight)
            weights += (weight*np.ones(n_points)).tolist()
        weights = np.array(weights)/n_points
        return weights

    def simplify_irreducible_sampling(self, ipoly_sample):
        all_k = []
        for name, k in ipoly_sample.items():
            all_k.append(k)
        all_k = np.vstack(all_k)
        return all_k

    def symmetry(self):
        """
        Return the maximum point symmetry of the unit cell

        Returns
        -------
        PointSymmetry
            point symmetry of the unit cell
        """
        sigma_h = symmetry_from_type(PointSymmetry.SIGMA_H)
        c1 = symmetry_from_type(PointSymmetry.C1)
        c2 = symmetry_from_type(PointSymmetry.C2)
        c3 = symmetry_from_type(PointSymmetry.C3)
        c4 = symmetry_from_type(PointSymmetry.C4)
        c6 = symmetry_from_type(PointSymmetry.C6)
        d2 = sigma_h + c2
        d3 = sigma_h + c3
        d4 = sigma_h + c4
        d6 = sigma_h + c6
        symmetries = {BravaisLattice.HEXAGON:d6,
                      BravaisLattice.SQUARE:d4,
                      BravaisLattice.RECTANGLE:d2,
                      BravaisLattice.OBLIQUE:c2}



        if self.lattice.bravais not in symmetries:
            raise ValueError("could not determine symmetry of lattice "+
                             "{}".format(self.lattice.bravais))
        return symmetries[self.lattice.bravais]

    def refl_rot_symmetries(self):
        """
        Return the point symmetry of the special points of the unit cell

        Returns
        -------
        dict
            mapping of special point to point symmetry
        """
        symmetries = {}
        
        # symmetries[SpecialPoint.GAMMA] = {BravaisLattice.HEXAGON:PointSymmetry.C1,
        #                                   BravaisLattice.SQUARE:PointSymmetry.C1,
        #                                   BravaisLattice.RECTANGLE:PointSymmetry.C1,
        #                                   BravaisLattice.OBLIQUE:PointSymmetry.C1}
        # symmetries[SpecialPoint.K] = {BravaisLattice.HEXAGON:PointSymmetry.C2}
        # symmetries[SpecialPoint.M]  = {BravaisLattice.HEXAGON:PointSymmetry.C3,
        #                                BravaisLattice.SQUARE:PointSymmetry.C1,
        #                                BravaisLattice.RECTANGLE:PointSymmetry.C1}
        # symmetries[SpecialPoint.X] = {BravaisLattice.SQUARE:PointSymmetry.SIGMA_D,
        #                               BravaisLattice.RECTANGLE:PointSymmetry.C1,
        #                               BravaisLattice.OBLIQUE:PointSymmetry.C2}
        # symmetries[SpecialPoint.Y] = {BravaisLattice.RECTANGLE:PointSymmetry.C1}
        # symmetries[SpecialPoint.Y1] = {BravaisLattice.OBLIQUE: PointSymmetry.C1}
        # symmetries[SpecialPoint.Y2] = {BravaisLattice.OBLIQUE: PointSymmetry.C1}
        # symmetries[SpecialPoint.H1] = {BravaisLattice.OBLIQUE:PointSymmetry.C1}
        # symmetries[SpecialPoint.H2] = {BravaisLattice.OBLIQUE:PointSymmetry.C1}
        # symmetries[SpecialPoint.H3] = {BravaisLattice.OBLIQUE:PointSymmetry.C1}
        # symmetries[SpecialPoint.C] = {BravaisLattice.OBLIQUE:PointSymmetry.C1}

        # symmetries[SpecialPoint.AXIS] = {BravaisLattice.HEXAGON:PointSymmetry.C6,
        #                                  BravaisLattice.SQUARE:PointSymmetry.C4,
        #                                  BravaisLattice.RECTANGLE:PointSymmetry.C2,
        #                                  BravaisLattice.OBLIQUE:PointSymmetry.C1}
        # symmetries[SpecialPoint.INTERIOR] = {BravaisLattice.HEXAGON:PointSymmetry.D6,
        #                                      BravaisLattice.SQUARE:PointSymmetry.D4,
        #                                      BravaisLattice.RECTANGLE:PointSymmetry.D2,
        #                                      BravaisLattice.OBLIQUE:PointSymmetry.C2}
        sigma_h = symmetry_from_type(PointSymmetry.SIGMA_H)
        c1 = symmetry_from_type(PointSymmetry.C1)
        c2 = symmetry_from_type(PointSymmetry.C2)
        c3 = symmetry_from_type(PointSymmetry.C3)
        c4 = symmetry_from_type(PointSymmetry.C4)
        c6 = symmetry_from_type(PointSymmetry.C6)
        d1 = sigma_h + c1        
        d2 = sigma_h + c2
        d3 = sigma_h + c3
        d4 = sigma_h + c4
        d6 = sigma_h + c6        
        
        symmetries[SpecialPoint.GAMMA] = {BravaisLattice.HEXAGON:d6,
                                          BravaisLattice.SQUARE:d4,
                                          BravaisLattice.RECTANGLE:d2,
                                          BravaisLattice.OBLIQUE:c2}
        
        symmetries[SpecialPoint.K] = {BravaisLattice.HEXAGON:sigma_h}
        
        symmetries[SpecialPoint.M]  = {BravaisLattice.HEXAGON:sigma_h,
                                       BravaisLattice.SQUARE:sigma_h,
                                       BravaisLattice.RECTANGLE:c1}
        
        symmetries[SpecialPoint.X] = {BravaisLattice.SQUARE:sigma_h,
                                      BravaisLattice.RECTANGLE:sigma_h,
                                      BravaisLattice.OBLIQUE:c1}
        
        symmetries[SpecialPoint.Y] = {BravaisLattice.RECTANGLE:sigma_h}
        
        symmetries[SpecialPoint.Y1] = {BravaisLattice.OBLIQUE:c1}
        symmetries[SpecialPoint.Y2] = {BravaisLattice.OBLIQUE:c1}
        symmetries[SpecialPoint.H1] = {BravaisLattice.OBLIQUE:c1}
        symmetries[SpecialPoint.H2] = {BravaisLattice.OBLIQUE:c1}
        symmetries[SpecialPoint.H3] = {BravaisLattice.OBLIQUE:c1}
        symmetries[SpecialPoint.C] = {BravaisLattice.OBLIQUE:c1}

        symmetries[SpecialPoint.AXIS] = {BravaisLattice.HEXAGON:sigma_h,
                                         BravaisLattice.SQUARE:sigma_h,
                                         BravaisLattice.RECTANGLE:sigma_h,
                                         BravaisLattice.OBLIQUE:c2}
        symmetries[SpecialPoint.EXTERIOR] = {BravaisLattice.HEXAGON:c1,
                                             BravaisLattice.SQUARE:c1,
                                             BravaisLattice.RECTANGLE:c1,
                                             BravaisLattice.OBLIQUE:c1}
        symmetries[SpecialPoint.INTERIOR] = {BravaisLattice.HEXAGON:c1,
                                             BravaisLattice.SQUARE:c1,
                                             BravaisLattice.RECTANGLE:c1,
                                             BravaisLattice.OBLIQUE:c1}
        
        symmetry_regions = {}
        special_points = list(self.special_points.keys())
        special_points += [SpecialPoint.AXIS, SpecialPoint.EXTERIOR, SpecialPoint.INTERIOR]
        for s_point in special_points:
            if self.lattice.bravais not in symmetries[s_point]:
                continue
            symmetry_regions[s_point] = symmetries[s_point][self.lattice.bravais]
        return symmetry_regions

    def translational_symmetries(self):
        trans1 = symmetry_from_type(PointSymmetry.T1)
        trans1.vector = -self.lattice.vectors.vec1

        trans2 = symmetry_from_type(PointSymmetry.T2)
        trans2.vector = -self.lattice.vectors.vec2

        trans3 = symmetry_from_type(PointSymmetry.T1)
        trans3.vector = -self.lattice.vectors.vec1 -self.lattice.vectors.vec2

        trans12 = trans1 + trans2
        symmetries = {}
        symmetries[SpecialPoint.K] = {BravaisLattice.HEXAGON:trans12}
        
        symmetries[SpecialPoint.M]  = {BravaisLattice.HEXAGON:trans1,
                                       BravaisLattice.SQUARE:trans12,
                                       BravaisLattice.RECTANGLE:trans12}
        
        symmetries[SpecialPoint.X] = {BravaisLattice.SQUARE:trans1,
                                      BravaisLattice.RECTANGLE:trans1,
                                      BravaisLattice.OBLIQUE:trans3}
        
        symmetries[SpecialPoint.Y] = {BravaisLattice.RECTANGLE:trans2}
        
        symmetries[SpecialPoint.Y1] = {BravaisLattice.OBLIQUE:trans2}
        #symmetries[SpecialPoint.Y2] = {BravaisLattice.OBLIQUE: PointSymmetry.C1}
        symmetries[SpecialPoint.H1] = {BravaisLattice.OBLIQUE:trans3}
        symmetries[SpecialPoint.H2] = {BravaisLattice.OBLIQUE:trans3}
        symmetries[SpecialPoint.H3] = {BravaisLattice.OBLIQUE:trans3}
        symmetries[SpecialPoint.C] = {BravaisLattice.OBLIQUE:trans3}

        symmetries[SpecialPoint.EXTERIOR] = {BravaisLattice.HEXAGON:trans1,
                                             BravaisLattice.SQUARE:trans1,
                                             BravaisLattice.RECTANGLE:trans1,
                                             BravaisLattice.OBLIQUE:trans1}

        
        trans_symmetries = {}
        #special_points = list(self.special_points.keys())
        #special_points += [SpecialPoint.EXTERIOR]
        for s_point in symmetries.keys():
            if self.lattice.bravais not in symmetries[s_point]:
                continue
            trans_symmetries[s_point] = symmetries[s_point][self.lattice.bravais]
        return trans_symmetries
