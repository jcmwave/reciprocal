import numpy as np
from reciprocal.utils import (apply_symmetry_operators, lies_on_vertex, lies_on_poly,
                              name_vertices, lies_on_sym_line, rotation2D)
from reciprocal.symmetry import Symmetry
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import itertools
import scipy.spatial

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

def overlap(p1, p2):
    if np.all(np.isclose(p1,p2)):
        return True
    else:
        return False

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def order_lexicographically(points, start=0.0):
    #angle = np.arctan2( points[:,1], points[:,0])
    angle = np.angle( (points[:,0]+ 1j*points[:,1])*np.exp(1j*(np.pi+1e-9+start)))
    #print(np.sort(angle))
    sortKey = np.argsort(angle)
    return points[sortKey, :]

def sumtriangles(xy, z, triangles ):
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


class Primitive():

    def __init__(self, vectors=None, vertices=None, WignerSeitz=False):
        if vectors is None and vertices is None:
            raise ValueError("one of vectors or vertices must be defined")
        self.wigner_seitz = WignerSeitz
        if vectors is not None:
            #self.shape = vectors.shape
            self.vectors = vectors
            self.vertices = None
            self.make_vertices()
        else:
            #self.shape = 'general'
            self.vectors = None
            self.vertices = vertices
        max_extent = 0.0
        for row in range(self.vertices.shape[0]):
            vertex = self.vertices[row, :]
            norm = np.linalg.norm(vertex)
            if norm > max_extent:
                max_extent = norm
        self.max_extent = max_extent
        self.symmetry_points = None
        self.symmetric_sampling = None
        self.sampling = None
        # self.symmetry_points = None
        # self.irreducible_polygon = None
        # self.polygon = None
        # self.irreducible_sampling = None
        # self.sampling = None


    def area(self):
        x = self.vertices[:,0]
        y = self.vertices[:,1]
        area =  0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
        return area

    def make_vertices(self):
        if self.wigner_seitz:
            self.make_wigner_seitz_cell()
        else:
            self._quick_vertices()

    def _quick_vertices(self):
        vertices = []
        vec1 = self.vectors.vec1
        vec2 = self.vectors.vec2
        vertices.append((vec1-vec2)*0.5)
        vertices.append((vec1+vec2)*0.5)
        vertices.append((-vec1+vec2)*0.5)
        vertices.append((-vec1-vec2)*0.5)
        vertices = np.array(vertices)
        self.vertices = vertices

    def make_wigner_seitz_cell(self):
        vertices = []
        vec1 = self.vectors.vec1
        vec2 = self.vectors.vec2
        angle = self.vectors.angle
        # Create lines perpendicular to vectors pointing to closest lattice cites
        for i_first in range(-1, 2):
            for i_second in range(-1, 2):
                if i_first == 0 and i_second == 0:
                    continue
                elif i_first == 1 and i_second == -1:
                    pass
                elif i_first == -1 and i_second == 1:
                    pass
                # elif i_first == 1 and i_second == 1 and angle-90 < 90.:
                #     continue
                # elif i_first == -1 and i_second == -1 and angle-90 < 90.:
                #     continue

                vertex = i_first*vec1*0.5 + i_second*vec2*0.5
                distance = np.linalg.norm(vertex)
                perpendicular = np.cross(vertex, np.array([0., 0., 1.]))*10
                my_line = np.squeeze(np.array([[vertex+perpendicular],[vertex-perpendicular]]))
                vertices.append((distance, my_line))
                #plt.scatter(vertex[0], vertex[1], color='k', s= 100)
                #plt.plot(my_line[:, 0], my_line[:, 1])
        if np.isclose(self.vectors.length1, self.vectors.length2):
            closest = np.array([np.inf])
        else:
            closest = np.array([np.inf, np.inf])
        # Find all intersection points of the lines
        intersections = []
        for vertex1, vertex2 in itertools.combinations(vertices, 2):
            L1 = line(vertex1[1][0, :], vertex1[1][1, :])
            L2 = line(vertex2[1][0, :], vertex2[1][1, :])
            inter = intersection(L1, L2)
            if inter is not False:
                my_intersection = np.array(inter)
                distance = np.linalg.norm(my_intersection)
                for ic in range(closest.size):
                    if distance < closest[ic]:
                        closest[ic] = distance
                        break
                    if np.isclose(distance, closest[ic]):
                        break
                # print("intersection at {}".format(my_intersection) +
                #        ", distance: {}".format(distance) +
                #        ", closest:{}".format(closest))
                if np.all(distance > closest+1e-6):
                       continue
                #plt.scatter(my_intersection[0], my_intersection[1], s=100)
                intersections.append([distance, my_intersection])
                #plt.scatter(vertex[0], vertex[1], s=100)
                #plt.plot(my_line[:, 0], my_line[:, 1])
        # split intersections into a set of closest points and all other points
        final_intersections = []
        # circ = Circle(xy=[0., 0.], radius=closest[-1], facecolor=[0.0, 0.0, 0.0, 0.0],
        #         edgecolor='k')
        # plt.gca().add_artist(circ)
        keep_intersections = []
        for i_inter, inter in enumerate(intersections):
            #print(inter)
            distance = inter[0]
            if np.any(distance> closest+1e-6):
                keep_intersections.append(inter)
                continue
            # print("intersection at {}".format(inter[1]) +
            #       ", distance: {}".format(inter[0]) +
            #       ", closest:{}".format(closest))
            final_intersections.append(inter[1])
            #plt.scatter(inter[1][0], inter[1][1], s=100)

        intersections = np.array(final_intersections)
        angle_b1 = np.angle(vec1[0]+1j*vec1[1])
        intersections = order_lexicographically(intersections, start=-angle_b1)
        self.vertices = intersections

    def make_sampling(self, constraint = None):
        irreducible_path = Polygon(self.vertices,
                                   closed=True).get_path()

        if constraint is None:
            constraint = {'type':'n_points', 'value':5}
        #create array of points in the 1st BZ
        #normalise reciprocal vectors to the 1st BZ
        if constraint['type'] == "density":
            density = constraint['value']
            vec1 = (density*(self.vectors.vec1[:2])/
                    np.linalg.norm(self.vectors.vec1[:2]))
            vec2 = (density*(self.vectors.vec2[:2])/
                    np.linalg.norm(self.vectors.vec2[:2]))
            n_grid_points = int(0.5*np.linalg.norm(self.vectors.vec1[:2])/density)
        elif constraint['type'] == "n_points":
            n_grid_points = constraint['value']

        if n_grid_points == 0:
            n_grid_points = 1

        if n_grid_points == 1:
            vec1 = np.array([0.0, 0.0])
            vec2 = np.array([0.0, 0.0])
        else:
            vec1 = np.array(self.vectors.vec1[:2])
            vec2 = np.array(self.vectors.vec2[:2])
            #angle = angle_between(vec1,vec2)
            rot1 = rotation2D(30.)
            rot2 = rotation2D(-60.)            
            vec1 = rot1.dot(vec1)
            vec2 = rot2.dot(vec2)            
            #vec2 = np.array([1.0, 0.0])*
            #print("[{} - {}]".format(vec1,self.vectors.vec2[:2]))

            #vec1 = rot.dot(vec1)
            #print(vec2)
            vec1 = ((1/np.sqrt(3))/(n_grid_points-1))*(vec1)            
            vec2 = (0.5/(n_grid_points-1))*(vec2)
            #plt.plot([0.,vec1[0]],[0.,vec1[1]],color='b')
            #plt.plot([0.,0.],vec2,color='b')            
            #print("{} - {}".format(vec1, vec2))

        ipoly_samp = {}
        ipoly_samp['G'] = []
        ipoly_samp['M'] = []
        ipoly_samp['X'] = []
        ipoly_samp['Y'] = []
        ipoly_samp['K'] = []
        ipoly_samp['C'] = []
        ipoly_samp['Y1'] = []
        ipoly_samp['Y2'] = []
        ipoly_samp['H1'] = []
        ipoly_samp['H2'] = []
        ipoly_samp['H3'] = []
        ipoly_samp['onSymmetryAxis'] = []
        ipoly_samp['interior'] = []

        range_lim = n_grid_points+int(n_grid_points/2.0)

        #named_vertices = name_vertices(self.vertices, self.symmetry_points )
        point_list = []
        for nx in range(-range_lim, range_lim):
            for ny in range(-range_lim, range_lim):
                trial_point = nx*vec1 + ny*vec2
                if not irreducible_path.contains_point(trial_point,
                                                       radius=1e-7):
                    continue
                point_list.append(trial_point)
                on_vertex, symmetry_point = lies_on_vertex(trial_point,
                                                           self.symmetry_points)
                #on_sym_line, point12 = lies_on_sym_line(trial_point,
                #                                        named_vertices)
                if on_vertex:
                    if  not symmetry_point == "Y2":
                        ipoly_samp[symmetry_point].append(trial_point)
                elif lies_on_poly(trial_point, self.vertices):
                    ipoly_samp['onSymmetryAxis'].append(trial_point)
                    #elif on_sym_line:
                    #ipoly_samp['onSymmetryAxis'].append(trial_point)
                else:
                    ipoly_samp['interior'].append(trial_point)
        for row in range(self.vertices.shape[0]):
            vertex = self.vertices[row,:]
            included = False
            for point in point_list:
                if overlap(vertex, point):
                    included = True
                    break
            if not included:
                point_list.append(vertex)
                on_vertex, symmetry_point = lies_on_vertex(vertex,
                                                           self.symmetry_points)
                ipoly_samp[symmetry_point].append(vertex)
        point_types = ['G', 'M', 'K', 'X', 'Y', 'Y1', 'Y2', 'C',
                       'H1', 'H2', 'H3',
                       'onSymmetryAxis', 'interior']
        for point_type in point_types:
            if len(ipoly_samp[point_type]) > 0:
                ipoly_samp[point_type] = np.array(ipoly_samp[point_type])
            else:
                del ipoly_samp[point_type]
        self.sampling = order_lexicographically(np.array(point_list))
        self.symmetric_sampling = ipoly_samp

    def integrate_sampled_function(self, function_vals):
        tri = scipy.spatial.Delaunay(self.sampling)
        z_sum, area_sum = sumtriangles(self.sampling, function_vals, tri.simplices)
        return z_sum


class BrillouinZone(Primitive):

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

    def make_symmetry_points(self):
        symmetry_points = []
        symmetry_points.append(('G', np.array([0.,0.])))
        vec1 = self.vectors.vec1[:2]
        vec2 = self.vectors.vec2[:2]
        length1 = self.vectors.length1
        length2 = self.vectors.length2
        angle = self.vectors.angle
        co_angle = 180. - angle
        #GM = np.linalg.norm(vec1)*0.5
        if np.isclose(length1, length2) and np.isclose(angle, 120.):
            #Hexagonal
            self.group = 'hexagon'
            GM = length1*0.5
            symmetry_points.append(('K', np.array([GM/np.cos(np.pi/6.0),0.0])))
            symmetry_points.append(('M', np.array([GM*np.cos(np.pi/6.0),
                                                   GM*np.sin(np.pi/6.0)])))
        elif np.isclose(length1, length2) and np.isclose(angle, 90.):
            #square
            self.group = 'square'
            symmetry_points.append(('X', 0.5*vec1))
            symmetry_points.append(('M', 0.5*(vec1+vec2)))
        elif (np.isclose(angle, 90.) or
              np.isclose(length2*np.cos(np.radians(co_angle)), length1)):
            #rectangle
            self.group = 'rectangle'
            symmetry_points.append(('X', 0.5*vec1))
            symmetry_points.append(('M', 0.5*(vec1+vec2)))
            symmetry_points.append(('Y', 0.5*vec2))
        else:
            #general base of a monoclinic type lattice
            self.group = 'monoclinic'
            symmetry_points.append(('X', 0.5*vec1))
            symmetry_points.append(('H1', self.vertices[0]))
            symmetry_points.append(('C', 0.5*(vec1+vec2)))
            symmetry_points.append(('H2', self.vertices[1]))
            symmetry_points.append(('Y1', 0.5*vec2))
            symmetry_points.append(('Y2', -0.5*vec2))
            symmetry_points.append(('H3', self.vertices[-1]))

        self.symmetry_points = symmetry_points

    def make_ibzone(self):
        if self.symmetry_points is None:
            self.make_symmetry_points()

        ipoly = []
        for key_val in self.symmetry_points:
            if self.group == 'monoclinic' and key_val[0] == 'G':
                continue
            ipoly.append(key_val[1])
        ipoly = np.array(ipoly)
        self.irreducible = Primitive(vertices=ipoly)
        self.irreducible.symmetry_points = self.symmetry_points
        self.irreducible.vectors = self.vectors
        self.irreducible.make_sampling()



    def make_sampling(self):
        all_points = []
        symmetry_regions = self.symmetry_regions()
        for symmetry in symmetry_regions:
            if symmetry not in self.irreducible.symmetric_sampling:
                continue
            for irr_point in range(self.irreducible.symmetric_sampling[symmetry].shape[0]):

                point = self.irreducible.symmetric_sampling[symmetry][irr_point, :]
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

    def symmetry_regions(self):
        symmetries = {}
        symmetries['G'] = {'hexagon':'inf',
                           'square':'inf',
                           'rectangle':'inf',
                           'monoclinic': 'inf'}
        symmetries['K'] = {'hexagon':'C6'}
        symmetries['M']  = {'hexagon':'C6',
                            'square':'inf',
                            'rectangle':'inf'}
        symmetries['X'] = {'square':'XY',
                           'rectangle':'inf',
                           'monoclinic':'C2'}
        symmetries['Y'] = {'rectangle':'inf'}
        symmetries['Y1'] = {'monoclinic': 'inf'}
        symmetries['Y2'] = {'monoclinic': 'inf'}
        symmetries['H1'] = {'monoclinic':'inf'}
        symmetries['H2'] = {'monoclinic':'inf'}
        symmetries['H3'] = {'monoclinic':'inf'}
        symmetries['C'] = {'monoclinic':'inf'}
        #symmetries['symmetry_points'] = {'hexagon':'C3','rectangle':'C2'}
        symmetries['onSymmetryAxis'] = {'hexagon':'C6',
                                        'square':'C4',
                                        'rectangle':'C2',
                                        'monoclinic': 'inf'}
        symmetries['interior'] = {'hexagon':'D6',
                                  'square':'D4',
                                  'rectangle':'D2',
                                  'monoclinic':'C2'}

        symmetry_regions = {}
        for sym in ['G', 'K', 'M', 'X', 'Y', 'Y1', 'Y2', 'H1', 'H2',
                    'H3', 'C', 'onSymmetryAxis', 'interior']:
            if self.group not in symmetries[sym]:
                continue
            symmetry_regions[sym] = Symmetry(symmetries[sym][self.group])
        return symmetry_regions

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
