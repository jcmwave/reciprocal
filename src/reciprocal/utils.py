import numpy as np
from enum import Enum
import pprint
def rotation2D(theta):
    theta = np.radians(theta)
    c, s = np.cos(theta), np.sin(theta)
    tol = 1e-6
    if np.abs(c) < tol:
        c = 0.0
    if np.abs(s) < tol:
        s = 0.0
    return np.array([[c,-s, 0.], [s, c, 0.], [0., 0., 1.]])


def rotation3D(theta,axis):
    axes = ['X','Y','Z']
    assert axis in axes
    theta = np.radians(theta)
    c, s = np.cos(theta), np.sin(theta)
    tol = 1e-6
    if np.abs(c) < tol:
        c = 0.0
    if np.abs(s) < tol:
        s = 0.0

    if axis == axes[0]:
        return np.array([ [1. , 0. ,0.],
                          [0. , c, -s],
                          [0. , s, c]])
    elif axis == axes[1]:
        return np.array([ [c , 0. ,-s],
                          [0. , 1., 0.],
                          [s , 0, c]])
    elif axis == axes[2]:
        return np.array([ [c, -s, 0.],
                          [s, c, 0.],
                          [0. ,0. , 1.]])


def reflection2D(axis):
    if axis == 'x':
        return np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
    elif axis == 'y':
        return np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    elif axis == 'xy':
        return np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
    return None

def translation2D(vector):
    t = np.array([[1., 0., vector[0]],
                  [0., 1., vector[1]],
                  [0., 0., 1.]])
    return t


def is_between(a, c, b, rel_tol=1e-3):
    return np.isclose(np.linalg.norm(a-c) + np.linalg.norm(c-b),
                      np.linalg.norm(a-b), rtol=rel_tol)


def lies_on_poly(point, poly, closed=True, rel_tol=1e-6):
    if closed:
        num_vert = poly.shape[0]
    else:
        num_vert = poly.shape[0]-1
    for i_vert in range(num_vert):
        vertex = poly[i_vert, :]
        if i_vert == poly.shape[0]-1:
            next_vert = poly[0, :]
        else:
            next_vert = poly[i_vert+1, :]
        if is_between(vertex, point, next_vert, rel_tol):
            return True
    return False

def lies_on_sym_line(point, named_vertices, closed=True, rel_tol=1e-3):
    if closed:
        num_vert = len(named_vertices)
    else:
        num_vert = len(named_vertices)-1

    for i_vert in range(num_vert):
        named_vert = named_vertices[i_vert]
        if i_vert == len(named_vertices)-1:
            next_named_vert = named_vertices[0]
        else:
            next_named_vert = named_vertices[i_vert+1]

        if (np.isclose(named_vert[1][0], next_named_vert[1][0], rtol=rel_tol) and
            np.isclose(named_vert[1][1], next_named_vert[1][1], rtol=rel_tol)):
            return (True, (named_vert[0], next_named_vert[0]))
    return (False, ("", ""))

def name_vertices(vertices, named_points):
    named_vertices = []
    for row in range(vertices.shape[0]):
        vertex = vertices[row, :]
        for name_point in named_points:
            name = name_point[0]
            point = name_point[1]
            if (np.isclose(vertex[0], point[0]) and
                np.isclose(vertex[1], point[1])):
                named_vertices.append((name, vertex))
    assert len(named_vertices) == vertices.shape[0]
    return named_vertices


def lies_on_vertex(point, named_vertices, rel_tol=1e-3):
    for special_point, vertex in named_vertices.items():
        #vertex = named_vertices[i_vert][1]
        vertex = np.atleast_2d(vertex)
        for row in range(vertex.shape[0]):
            if (np.isclose(vertex[row, 0], point[0], rtol=rel_tol) and
                np.isclose(vertex[row, 1], point[1], rtol=rel_tol)):
                return (True, special_point)
            
    return (False, '')

def order_lexicographically(points, start=0.0, return_sort_indices=False):
    angle = np.angle( (points[:,0]+1j*points[:,1])*np.exp(1j*(np.pi+1e-3+start)))
    angle = np.round(angle, 3)
    radius = np.linalg.norm(points, axis=1)
    angle[np.isclose(radius, 0.)] = -np.pi
    sort_indices = np.lexsort((radius, angle))
    #sort_indices = np.argsort(angle)
    all_data = np.round(np.vstack([points.T, angle, radius]).T, 3)
    #pp = pprint.PrettyPrinter(indent=4, width=120)
    #pp.pprint("x, y, z, angle, radius")
    #pp.pprint(all_data)
    #pp.pprint("x, y, z, angle, radius")
    #pp.pprint(all_data[sort_indices, :])
    if return_sort_indices:
        return points[sort_indices, :], sort_indices
    else:
        return points[sort_indices, :]

class BravaisLattice(Enum):
    HEXAGON = 0
    SQUARE = 1
    RECTANGLE = 2
    OBLIQUE = 3
    RHOMBUS = 4
