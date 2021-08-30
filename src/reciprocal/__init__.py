import numpy as np
import reciprocal.lattice
import reciprocal.kspace
import reciprocal.kvector
import reciprocal.primitive
import reciprocal.utils
from reciprocal.symmetry import Symmetry
__version__ = "0.0.1"
__all__ = ["Symmetry"]


class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


def rotation2D(theta):
    theta = np.radians(theta)
    c, s = np.cos(theta), np.sin(theta)
    tol = 1e-6
    if np.abs(c) < tol:
        c = 0.0
    if np.abs(s) < tol:
        s = 0.0
    return np.array([[c,-s], [s, c]])


def rotation3D(theta,axis):
    axes = ['X','Y','Z']
    assert axis in axes

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
        return np.array([[1.0,0.0], [0.0,-1.0]])
    elif axis == 'y':
        return np.array([[-1.0,0.0], [0.0,1.0]])
    elif axis == 'xy':
        return np.array([[0.0,1.0], [1.0,0.0]])
    return None


def apply_symmetry_operators(point,symmetry):
    points = []
    refYOperators = []
    rotOperators = []
    refXYOperators = []
    s = symmetry
    nRot = s.getNRotations()
    nRefY = s.getNReflectionsY()
    nRefXY = s.getNReflectionsXY()

    for i in range(1,nRot):
        rotOperators.append(rotation2D(i*360.0/nRot))

    for i in range(nRefY):
        refYOperators.append(reflection2D('y'))

    for i in range(nRefXY):
        refXYOperators.append(reflection2D('xy'))

    operatorStack = [np.eye(2)]

    for i in range(len(rotOperators)):
        operatorStack.append( rotOperators[i])
    nOps = len(operatorStack)

    for iOp in range(nOps):
        for iRefY in range(nRefY):
            operatorStack.append( refYOperators[iRefY].dot(operatorStack[iOp]))
    nOps = len(operatorStack)

    for iOp in range(nOps):
        for iRefXY in range(nRefXY):
            operatorStack.append( refXYOperators[iRefXY].dot(operatorStack[iOp]))
    points = []
    for op in operatorStack:
        points.append( op.dot(point))

    points = np.array(points)

    return points,operatorStack

def is_between(a,c,b,rel_tol=1e-3):
    return np.isclose(np.linalg.norm(a-c) + np.linalg.norm(c-b), np.linalg.norm(a-b),rtol=rel_tol)


def liesOnPoly(point,poly,closed=True,rel_tol=1e-3):
    if closed:
        nV = poly.shape[0]
    else:
        nV = poly.shape[0]-1
    for iV in range(nV):
        vertex = poly[iV,:]
        if iV == poly.shape[0]-1:
            nextVert = poly[0,:]
        else:
            nextVert = poly[iV+1,:]
        if is_between( vertex,point,nextVert,rel_tol):
            return True
    return False

def liesOnVertex(point,namedVertices,rel_tol=1e-3):
    for iV in range(len(namedVertices)):
        vertex = namedVertices[iV][1]
        if (np.isclose(vertex[0],point[0],rtol=rel_tol) and
            np.isclose(vertex[1],point[1],rtol=rel_tol)):
            return (True,namedVertices[iV][0])
    return (False,'')



if __name__ == '__main__':
    pass
