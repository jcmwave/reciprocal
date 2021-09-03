import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from reciprocal.unit_cell import UnitCell
from reciprocal.utils import rotation2D
from reciprocal.unit_cell import order_lexicographically
from reciprocal.utils import BravaisLattice
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def make_lattice_points(first_orders, second_orders, vktr1, vktr2):
    """Return a list of lattice points.

    The lattice point with order N1 == N2 == 0 is excluded.

    Parameters
    ----------
    first_orders: range
        the integer orders of v1 to iterate over
    second_orders: range
        the integer orders of v2 to iterate over
    vktr1: (2,)<np.double> np.array
        the first lattice vector
    vktr2: (2,)<np.double> np.array
        the second lattice vector

    Returns
    -------
    list of (2,)<np.double> np.array
    """
    lattice_points = []
    for xi in first_orders:
        for yi in second_orders:
            if xi == 0 and yi == 0:
                continue
            pos = vktr1*xi+vktr2*yi
            lattice_points.append(pos)
    return lattice_points

def get_unique_lengths(lattice_points):
    """Return the unique distance from the origin to a set of lattice points.

    Parameters
    ----------
    lattice_points: list of (2,)<np.double> np.array
        the lattice points used to calculate unique distances

    Returns
    -------
    list of floats
        the unique distances from the origin to the lattice points

    """
    lengths = []
    for point in lattice_points:
        dist = np.linalg.norm(point)
        lengths.append(dist)
    lengths = np.array(lengths)
    lengths = np.sort(np.unique(lengths))
    return lengths

def get_n_shortest(lattice_points, n_shortest, unique_lengths):
    """Return N lattice points with smallest unique distance to the origin.

    Parameters
    ----------
    lattice_points: (N,2)<np.double> np.ndarray
        array of lattice points
    n_shortest: 0<int<=2
        how many lattice points to return
    unique_lengths: list of floats
        the unqiue distance to the origin of the lattice points

    Returns
    -------
    list of (N,2)<np.double> np.ndarray
        the lattice points with unique smallest distance to the origin
    """
    start_row = 0
    n_found_vectors = 0
    vectors = []
    plot_n = 0
    for length in unique_lengths:
        for row in range(start_row, lattice_points.shape[0]):
            pos = lattice_points[row, :]
            dist = np.linalg.norm(pos)
            conditions = np.isclose(dist, length)

            if n_found_vectors == 1:
                """the second vector needs to fulfill extra conditions"""
                arg = np.clip(vectors[0].dot(pos)/
                              (np.linalg.norm(vectors[0])*dist), -1.0, 1.0)
                sep_angle = np.arccos(arg)
                cross_prod = np.cross(vectors[0], pos)
                obtuse = sep_angle >= np.pi*0.5
                not_reflex = cross_prod[2] > 0.
                linearly_independent = np.linalg.norm(cross_prod)/np.linalg.norm(vectors[0]) > 1e-3
                conditions = conditions and obtuse
                conditions = conditions and linearly_independent
                conditions = conditions and not_reflex

            if conditions:
                vectors.append(pos)
                n_found_vectors += 1
                start_row = row
                if n_found_vectors == n_shortest:
                    return vectors

    raise ValueError("not enough vectors found")

def make_vectors(length1, length2, angle):
    """Returns two vectors defined using lengths and angle between.

    The first vector is assumed to lie along the x axis. Both vectors lie in the
    x-y plane.
    """
    vec1 = length1*np.array([1.0, 0.0, 0.0])
    vec2 = length2*np.array([np.cos(np.radians(angle)),
                                       np.sin(np.radians(angle)), 0.0])
    return vec1, vec2

class LatticeVectors():
    """
    Defines two independent basis vectors of a lattice.

    Use the class methods from_vectors, from_lengths_angle to construct.

    Attribues
    ---------
    length1: float
        length of the first lattice vector
    length2: float
        length of the second lattice vector
    angle: float
        angle between the lattice vectors in degrees
    vec1: (3,)<float>np.array
        first lattice vector
    vec2: (3,)<float>np.array
        second lattice vector

    Methods
    -------
    from_vectors(vector1, vector2) [Constructor]


    make_vectors(self)
        sets vector class attributes from lengths and angle
    reciprocal_vectors(self)
        returns reciprocal lattice vectors
    get_shortest_vectors(self)
        return the shortest possible lattice vectors
    """

    def __init__(self, vector1, vector2):
        """
        initialize LatticeVectors object.

        Parameters
        ----------
        vector1: (3,)<np.double>np.array
            first lattice vector
        vector2: (3,)<np.double>np.array
            second lattice vector

        Returns
        -------
        LatticeVectors
        """
        self.vec1 = vector1
        self.vec2 = vector2
        self.angle = np.degrees(angle_between(vector1, vector2))
        self.length1 = np.linalg.norm(vector1)
        self.length2 = np.linalg.norm(vector2)

    def __repr__(self):
        return f"LatticeVectors({self.vec1}, {self.vec2})"

    @classmethod
    def from_lengths_angle(lat_vec, length1, length2, angle):
        """Return LatticeVectors from lengths and angle between.

        typical usage:
        lat_vec = LatticeVectors.from_lengths_angle(length1, length2, angle)

        Parameters
        ----------
        lat_vec: LatticeVector
            an instance of this class
        length1: float
            length of the first lattice vector
        length2: float
            length of the second lattice vector
        angle: float
            angle between vectors in degrees

        Returns
        -------
        LatticeVectors
        """
        vectors = make_vectors(length1, length2, angle)
        return lat_vec(vectors[0], vectors[1])

    def reciprocal_vectors(self):
        """Return reciprocal lattice vectors

        Returns
        -------
        LatticeVectors
        """
        if self.vec1 is None or self.vec2 is None:
            self.make_vectors()

        R = rotation2D(90)
        a1 = self.vec1[0:2]
        a2 = self.vec2[0:2]
        if np.any( a2 == float('inf')) :
            a2_ = np.array([0.0,1.0])
            b1 = (2*np.pi*R.dot(a2_)/ np.dot(a1,R.dot(a2_)))
            b2 = (2*np.pi*R.dot(a1)/ np.dot(a2,R.dot(a1)))
        else:
            b1 = (2*np.pi*R.dot(a2)/ np.dot(a1,R.dot(a2)))
            b2 = (2*np.pi*R.dot(a1)/ np.dot(a2,R.dot(a1)))
        b1 = np.array([b1[0],b1[1],0.0])
        b2 = np.array([b2[0],b2[1],0.0])

        b1_norm = np.linalg.norm(b1)
        b2_norm = np.linalg.norm(b2)
        rl = LatticeVectors(vector1=b1, vector2=b2)
        b1_new, b2_new = rl.get_shortest_vectors()
        rl = LatticeVectors(vector1=b1_new, vector2=b2_new)
        return rl

    def get_shortest_vectors(self):
        """Return arrays of lattice vectors that are as short as possible.

        Returns
        -------
        (2,) tuple of (3,)<np.double> np.array
        """
        n_shortest = 1
        if np.isclose(self.length1, self.length2):
            first = 2
            second = 2
            n_shortest = 2
        if self.length1 > self.length2:
            second = int(np.ceil(self.length1/self.length2))
            first = 2
        else:
            first = int(np.ceil(self.length2/self.length1))
            second = 2

        first_orders = range(-first, first+1)
        second_orders = range(-second, second+1)

        lattice_points = make_lattice_points(first_orders, second_orders,
                                             self.vec1, self.vec2)
        unique_lengths = get_unique_lengths(lattice_points)
        lattice_points = np.array(lattice_points)
        lattice_points = order_lexicographically(lattice_points,
                                                 start=0.5*np.pi-1e-1)

        vectors = []
        vectors = get_n_shortest(lattice_points, 2, unique_lengths)
        vectors = np.concatenate([vectors])
        return vectors[0, :], vectors[1, :]




class Lattice():
    """
    Defines a periodic lattice in 2D

    Attributes
    ----------
    lattice_type: str
        real-space or reciprocal
    vectors: LatticeVectors
        the lattice vectors defining the lattice
    bravais: BravaisLattice
        bravias lattice type (see class BravaisLattice)
    unit_cell: UnitCell
        the unit cell of the lattice
    brillouin_zone: UnitCell
        alias for unit_cell

    Methods
    -------
    from_lat_vec_args(lattice, kwargs)
        constructs a Lattice object using kwargs for a LatticeVector object
    make_reciprocal(self)
        returns a Lattice object defined by the reciprocal of this lattice

    """
    def __init__(self, lattice_vectors, lattice_type='real_space'):
        """
        initialize Lattice object.

        The standard constructor for this class takes a LatticeVectors argument.
        Alternatively the class method from_lat_vec_args may be used.

        Parameters
        ----------
        lattice_vectors LatticeVectors
            the lattice vectors

        Returns
        -------
        Lattice
        """
        valid_lattice_types = ("real_space", "reciprocal")
        if lattice_type not in valid_lattice_types:
            raise ValueError("lattice type {}".format(lattice_type) +
                             " not understood. lattice type must be in range "+
                             "({})".format(valid_lattice_types))
        self.lattice_type = lattice_type
        if lattice_vectors is None:
            lattice_vectors = LatticeVectors(**lv_args)
        else:
            lattice_vectors = lattice_vectors
        self.vectors = lattice_vectors
        self.bravais =  self.determine_bravais_lattice()
        if self.lattice_type == 'real_space':
            self.unit_cell = UnitCell(self, WignerSeitz=False)
        else:
            self.unit_cell = UnitCell(self, WignerSeitz=True)

    def __repr__(self):
        return f"Lattice({self.vectors}, {self.lattice_type})"


    @property
    def brillouin_zone(self):
        return self.unit_cell

    @brillouin_zone.setter
    def brillouin_zone(self, unit_cell):
        self.unit_cell = unit_cell

    @classmethod
    def from_lat_vec_args(lattice, **kwargs):
        """
        construct a Lattice using kwargs for a LatticeVectors object.

        This methods constructs the LatticeVectors object needed to construct
        a Lattice. This requires keyword argments to determine the correct
        LatticeVector constructor to use. kwargs must contain either vector1 and
        vector2, or length1, length2 and angle.

        Parameters
        ----------
        lattice: Lattice
            A Lattice object
        vector1: (2,)<np.double>np.array
            The first lattice vector
        vector2: (2,)<np.double>np.array
            The second lattice vector
        length1: float
            The length of the first lattice vector
        length2: float
            The length of the second lattice vector
        angle: float
            The angle between lattice vectors in degrees

        Returns
        -------
        Lattice
        """
        lat_vec = None
        try:
            if "vector1" in kwargs and "vector2" in kwargs:
                lat_vec = LatticeVectors(kwargs['vector1'], kwargs['vector2'])
        except:
            pass
        try:
            if ("length1" in kwargs and "length2" in kwargs
                and "angle" in kwargs):
                lat_vec = LatticeVectors.from_lengths_angle(kwargs['length1'],
                                                            kwargs['length2'],
                                                            kwargs['angle'])
        except:
            pass
        if lat_vec is None:
            raise ValueError("could not construct LatticeVectors from args: " +
                             "{}".format(kwargs))
        return lattice(lat_vec)

    def make_reciprocal(self):
        """Return a Lattice object defined by the reciprocal of this lattice.

        Returns
        -------
        Lattice
        """
        r_vectors = self.vectors.reciprocal_vectors()
        return Lattice(r_vectors, lattice_type='reciprocal')

    def determine_bravais_lattice(self):
        """
        determines the 2D bravais lattice of the unit cell

        Uses the vector lengths and angles of the lattice vectors to determine
        the bravais lattice.

        Returns
        -------
        BravaisLattice
        """
        length1 = self.vectors.length1
        length2 = self.vectors.length2
        angle = self.vectors.angle
        co_angle = 180. - angle
        if np.isclose(length1, length2) and np.isclose(angle, 120.):
            bv_lat = BravaisLattice.HEXAGON
        elif np.isclose(length1, length2) and np.isclose(angle, 90.):
            bv_lat = BravaisLattice.SQUARE
        elif (np.isclose(angle, 90.) or
              np.isclose(length2*np.cos(np.radians(co_angle)), length1)):
            bv_lat = BravaisLattice.RECTANGLE
        else:
            bv_lat = BravaisLattice.OBLIQUE
        return bv_lat

    def orders_by_distance(self, max_order):
        """
        Returns lattices orders grouped by equal distance

        Parameters
        ----------
        max_order: int
            the maximum order of lattice vector to consider

        Returns
        -------
        list of (N,2) <np.int> np.array
            the lattice orders
        (M,1) <np.double> np.array
            the associated distances
        """
        vec1 = self.vectors.vec1
        vec2 = self.vectors.vec2
        n_rows = (2*max_order+1)**2
        order_table = np.zeros((n_rows, 3))
        row = 0
        for order1 in range(-max_order, max_order+1):
            for order2 in range(-max_order, max_order+1):
                distance = np.linalg.norm(vec1*order1 + vec2*order2)
                order_table[row,:] = np.array([order1, order2, distance])
                row += 1
        sort_indices = np.argsort(order_table[:,2])
        order_table = order_table[sort_indices, :][1:, :]
        order_table[:,2] = np.round(order_table[:,2], 5)
        unique_distances = np.unique(order_table[:,2])
        orders_list = []
        for unique_dist in unique_distances:
            equal_to_distance = np.isclose(order_table[:,2],unique_dist)
            order_array = order_table[equal_to_distance, :2]
            order_array = order_array.astype('int64')
            orders_list.append(order_array)
        return orders_list, unique_distances
