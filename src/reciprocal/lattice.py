import numpy as np
import matplotlib.pyplot as plt
from reciprocal.primitive import Primitive, BrillouinZone
from reciprocal.utils import rotation2D
from reciprocal.primitive import order_lexicographically

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

def make_lattice_points(first_orders, second_orders, v1, v2):
    lattice_points = []
    #shortest = np.inf
    for xi in first_orders:
        for yi in second_orders:
            if xi == 0 and yi == 0:
                continue
            pos = v1*xi+v2*yi
            #dist = np.linalg.norm(pos)
            lattice_points.append(pos)
            #if dist < shortest:
                #shortest = dist
            #plt.scatter(pos[0], [pos[1]])
    return lattice_points

def get_unique_lengths(lattice_points):
    lengths = []
    for point in lattice_points:
        dist = np.linalg.norm(point)
        lengths.append(dist)
    lengths = np.array(lengths)
    lengths = np.sort(np.unique(lengths))
    return lengths

def get_n_shortest(lattice_points, n_shortest, unique_lengths):
    start_row = 0
    n_found_vectors = 0
    vectors = []
    plot_n = 0
    for length in unique_lengths:
        for row in range(start_row, lattice_points.shape[0]):
            pos = lattice_points[row, :]
            dist = np.linalg.norm(pos)
            #print(row, pos, dist)
            conditions = np.isclose(dist, length)

            # if conditions:
            #     hw = dist*0.05
            #     width = dist*0.02
            #     plt.arrow(0.0, 0.0, pos[0], pos[1],
            #               head_width=hw,
            #               width=width,
            #               color='r',
            #               ec='r',
            #               length_includes_head=True)
            #     plt.text(pos[0]*0.5, pos[1]*0.5, plot_n)
            #     plot_n += 1

            if n_found_vectors == 1:
                arg = np.clip(vectors[0].dot(pos)/(np.linalg.norm(vectors[0])*dist), -1.0, 1.0)
                sep_angle = np.arccos(arg)
                cross_prod = np.cross(vectors[0], pos)
                obtuse = sep_angle >= np.pi*0.5
                not_reflex = cross_prod[2] > 0.
                linearly_independent = np.linalg.norm(cross_prod)/np.linalg.norm(vectors[0]) > 1e-3
                #print(row, conditions, obtuse, linearly_independent, not_reflex)
                conditions = conditions and obtuse
                conditions = conditions and linearly_independent
                conditions = conditions and not_reflex

            if conditions:
                #print("added to vectors")
                vectors.append(pos)
                n_found_vectors += 1
                start_row = row
                if n_found_vectors == n_shortest:
                    return vectors

    raise ValueError("not enough vectors found")


class LatticeVectors():

    def __init__(self, length1=None, length2=None, angle=None,
                 vector1=None, vector2=None):
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
        self.vec1 = self.length1*np.array([1.0, 0.0, 0.0])
        self.vec2 = self.length2*np.array([np.cos(np.radians(self.angle)),
                                           np.sin(np.radians(self.angle)), 0.0])


    def reciprocal_vectors(self):
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
        vec1 = self.vec1
        vec2 = self.vec2
        lv1 = self.length1
        lv2 = self.length2
        #print(lv1, lv2)
        #equal_lengths = np.isclose(lv1, lv2)
        n_shortest = 1
        if np.isclose(lv1, lv2):
            first = 2
            second = 2
            n_shortest = 2
        if lv1 > lv2:
            second = int(np.ceil(lv1/lv2))
            first = 2
        else:
            first = int(np.ceil(lv2/lv1))
            second = 2

        first_orders = range(-first, first+1)
        second_orders = range(-second, second+1)
        #MOrders = len(first_orders)*len(second_orders)
        v1 = vec1
        v2 = vec2

        lattice_points = make_lattice_points(first_orders, second_orders, v1, v2)
        unique_lengths = get_unique_lengths(lattice_points)
        lattice_points = np.array(lattice_points)
        lattice_points = order_lexicographically(lattice_points,
                                                 start=0.5*np.pi-1e-1)


        vectors = []
        vectors = get_n_shortest(lattice_points, 2, unique_lengths)
        #if len(vectors) == 2:
        vectors = np.concatenate([vectors])
        return vectors[0, :], vectors[1, :]
        #else:
        #    return v1, v2









class Lattice():

    def __init__(self, length1, length2, angle):
        lattice_vectors = LatticeVectors(length1=length1,
                                         length2=length2,
                                         angle=angle)
        self.vectors = lattice_vectors
        self.primitive = None
        self.make_primitive()

    def make_primitive(self):
        self.primitive = Primitive(self.vectors)

    def make_reciprocal(self):
        r_vectors = self.vectors.reciprocal_vectors()
        return ReciprocalLattice(r_vectors)

class ReciprocalLattice(Lattice):

    def __init__(self, vectors):
        self.vectors = vectors
        self.bzone = None
        self.make_primitive()
        self.primitive = self.bzone

    def make_primitive(self):
        self.bzone = BrillouinZone(self.vectors)
