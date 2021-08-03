import numpy as np
import matplotlib.pyplot as plt
from reciprocal.unit_cell import UnitCell, BrillouinZone
from reciprocal.utils import rotation2D
from reciprocal.unit_cell import order_lexicographically

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


class LatticeVectors():
    """
    Defines two independent basis vectors of a lattice.

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
    make_vectors(self)
        sets vector class attributes from lengths and angle
    reciprocal_vectors(self)
        returns reciprocal lattice vectors
    get_shortest_vectors(self)
        return the shortest possible lattice vectors
    """

    def __init__(self, length1=None, length2=None, angle=None,
                 vector1=None, vector2=None):
        """
        initialize LatticeVectors object.

        This class can be initialized via two mutually exclusive argument
        combinations. Either:

        length1, length2 and angle

        or

        vector1 and vector2
        """
        valid_inputs = False
        if length1 is not None and length2 is not None and angle is not None:
            valid_inputs = True
            input_type = 1
        elif vector1 is not None and vector2 is not None:
            valid_inputs = True
            input_type = 2
        if valid_inputs is False:
            raise ValueError("inputs must either be two vector lengths and "+
                             "lattice angle or the lattice vectors")

        if input_type == 1:
            self.length1 = length1
            self.length2 = length2
            self.angle = angle
            self.make_vectors()
        elif input_type == 2:
            self.vec1 = vector1
            self.vec2 = vector2
            self.angle = np.degrees(angle_between(vector1, vector2))
            self.length1 = np.linalg.norm(vector1)
            self.length2 = np.linalg.norm(vector2)

    def make_vectors(self):
        """Set vector attributes using lengths and angle.

        The first vector is assumed to lie along the x axis.
        """
        self.vec1 = self.length1*np.array([1.0, 0.0, 0.0])
        self.vec2 = self.length2*np.array([np.cos(np.radians(self.angle)),
                                           np.sin(np.radians(self.angle)), 0.0])


    def reciprocal_vectors(self):
        """Return recprocal lattice vectors based on this lattice."""
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
        """Return arrays of lattice vectors that are as short as possible."""
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
    Defines a periodic lattice in 2D based on two independent lattice vectors
    defined by their lengths and separation angle.
    """
    def __init__(self, length1, length2, angle):
        lattice_vectors = LatticeVectors(length1=length1,
                                         length2=length2,
                                         angle=angle)
        self.vectors = lattice_vectors
        self.unit_cell = None
        self.make_unit_cell()

    def make_unit_cell(self):
        self.unit_cell = UnitCell(self.vectors)

    def make_reciprocal(self):
        r_vectors = self.vectors.reciprocal_vectors()
        return ReciprocalLattice(r_vectors)

class ReciprocalLattice(Lattice):
    """
    Extends the lattice class to define a reciprocal lattice. A reciprocal
    lattice has a BrillouinZone as its unit_cell.
    """

    def __init__(self, vectors):
        self.vectors = vectors
        self.bzone = None
        self.make_unit_cell()
        self.unit_cell = self.bzone

    def make_unit_cell(self):
        self.bzone = BrillouinZone(self.vectors)
