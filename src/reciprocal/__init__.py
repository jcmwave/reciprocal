from importlib.metadata import PackageNotFoundError, version

import numpy as np

import reciprocal.lattice
import reciprocal.kspace
import reciprocal.kvector
import reciprocal.utils
from reciprocal.symmetry import Symmetry

try:
    __version__ = version("reciprocal")
except PackageNotFoundError:  # Source checkout used without installation.
    __version__ = "0+unknown"

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
        super().__init__(message)
        self.expression = expression
        self.message = message


def rotation2D(theta):
    theta = np.radians(theta)
    c, s = np.cos(theta), np.sin(theta)
    if np.abs(c) < 1e-6:
        c = 0.0
    if np.abs(s) < 1e-6:
        s = 0.0
    return np.array([[c, -s], [s, c]])


def rotation3D(theta, axis):
    axes = ["X", "Y", "Z"]
    if axis not in axes:
        raise ValueError("axis must be one of 'X', 'Y', or 'Z'")
    c, s = np.cos(theta), np.sin(theta)
    if np.abs(c) < 1e-6:
        c = 0.0
    if np.abs(s) < 1e-6:
        s = 0.0
    if axis == "X":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == "Y":
        return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def reflection2D(axis):
    operators = {
        "x": np.array([[1.0, 0.0], [0.0, -1.0]]),
        "y": np.array([[-1.0, 0.0], [0.0, 1.0]]),
        "xy": np.array([[0.0, 1.0], [1.0, 0.0]]),
    }
    try:
        return operators[axis]
    except KeyError as exc:
        raise ValueError("axis must be one of 'x', 'y', or 'xy'") from exc


def apply_symmetry_operators(point, symmetry):
    rotations = [
        rotation2D(i * 360.0 / symmetry.getNRotations())
        for i in range(1, symmetry.getNRotations())
    ]
    operators = [np.eye(2), *rotations]
    base_count = len(operators)
    for operator in operators[:base_count]:
        for _ in range(symmetry.getNReflectionsY()):
            operators.append(reflection2D("y").dot(operator))
    base_count = len(operators)
    for operator in operators[:base_count]:
        for _ in range(symmetry.getNReflectionsXY()):
            operators.append(reflection2D("xy").dot(operator))
    return np.array([operator.dot(point) for operator in operators]), operators


def is_between(a, c, b, rel_tol=1e-3):
    return np.isclose(
        np.linalg.norm(a - c) + np.linalg.norm(c - b),
        np.linalg.norm(a - b),
        rtol=rel_tol,
    )


def liesOnPoly(point, poly, closed=True, rel_tol=1e-3):
    num_vertices = poly.shape[0] if closed else poly.shape[0] - 1
    for index in range(num_vertices):
        next_index = 0 if index == poly.shape[0] - 1 else index + 1
        if is_between(poly[index, :], point, poly[next_index, :], rel_tol):
            return True
    return False


def liesOnVertex(point, namedVertices, rel_tol=1e-3):
    for name, vertex in namedVertices:
        if np.all(np.isclose(vertex[:2], point[:2], rtol=rel_tol)):
            return True, name
    return False, ""
