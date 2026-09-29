"""Construction of primitive, conventional, and Wigner-Seitz cells."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from reciprocal.bravais import BravaisLattice
from reciprocal.numerics import DEFAULT_TOLERANCES, Tolerances

from .model import PolygonDomain, UnitCell, UnitCellKind


class VectorPair(Protocol):
    vec1: NDArray[np.float64]
    vec2: NDArray[np.float64]


def _as_three(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.asarray(vector, dtype=float)
    if result.shape not in ((2,), (3,)) or not np.all(np.isfinite(result)):
        raise ValueError("lattice vectors must be finite vectors of length two or three")
    if result.shape == (3,) and result[2] != 0.0:
        raise ValueError("lattice vectors must be planar")
    return np.append(result, 0.0) if len(result) == 2 else result.copy()


def reduce_basis(vectors: VectorPair) -> NDArray[np.float64]:
    """Return a deterministic two-dimensional Gauss-reduced row basis."""
    first, second = _as_three(vectors.vec1), _as_three(vectors.vec2)
    original_orientation = np.sign(first[0] * second[1] - first[1] * second[0])
    for _ in range(128):
        first_key = (np.dot(first, first), first[0], first[1])
        second_key = (np.dot(second, second), second[0], second[1])
        if second_key < first_key:
            first, second = second, first
        coefficient = int(np.rint(np.dot(first, second) / np.dot(first, first)))
        if coefficient == 0:
            break
        second = second - coefficient * first
    else:
        raise RuntimeError("could not reduce lattice basis")
    orientation = first[0] * second[1] - first[1] * second[0]
    if orientation * original_orientation < 0:
        second = -second
    return np.vstack((first, second))


def _centered_parallelogram(basis: NDArray[np.float64]) -> PolygonDomain:
    first, second = basis
    return PolygonDomain(
        np.vstack(
            (
                (-first - second) / 2,
                (first - second) / 2,
                (first + second) / 2,
                (-first + second) / 2,
            )
        )
    )


def make_primitive_cell(vectors: VectorPair) -> UnitCell:
    basis = reduce_basis(vectors)
    return UnitCell(basis, _centered_parallelogram(basis), [[0.0, 0.0]], UnitCellKind.PRIMITIVE)


def make_conventional_cell(vectors: VectorPair, bravais: BravaisLattice) -> UnitCell:
    basis = reduce_basis(vectors)
    points = np.array([[0.0, 0.0]])
    if bravais is BravaisLattice.CENTERED_RECTANGULAR:
        first, second = basis
        plus, minus = first + second, first - second
        if abs(np.dot(plus[:2], minus[:2])) > 1e-8 * np.linalg.norm(plus) * np.linalg.norm(minus):
            raise ValueError("centered-rectangular primitive vectors must have equal length")
        basis = np.vstack((plus, minus))
        if basis[0, 0] * basis[1, 1] - basis[0, 1] * basis[1, 0] < 0:
            basis[1] *= -1
        points = np.array([[0.0, 0.0], [0.5, 0.5]])
    return UnitCell(basis, _centered_parallelogram(basis), points, UnitCellKind.CONVENTIONAL)


def _clip_half_plane(
    vertices: NDArray[np.float64], normal: NDArray[np.float64], offset: float, tolerance: float
) -> NDArray[np.float64]:
    clipped: list[NDArray[np.float64]] = []
    for index, start in enumerate(vertices):
        end = vertices[(index + 1) % len(vertices)]
        start_distance = np.dot(start, normal) - offset
        end_distance = np.dot(end, normal) - offset
        start_inside, end_inside = start_distance <= tolerance, end_distance <= tolerance
        if start_inside != end_inside:
            direction = end - start
            clipped.append(
                start + (offset - np.dot(start, normal)) / np.dot(direction, normal) * direction
            )
        if end_inside:
            clipped.append(end)
    if not clipped:
        raise RuntimeError("lattice half-planes produced an empty Wigner-Seitz cell")
    return np.asarray(clipped)


def make_wigner_seitz_cell(
    vectors: VectorPair, *, tolerances: Tolerances = DEFAULT_TOLERANCES
) -> UnitCell:
    if not isinstance(tolerances, Tolerances):
        raise TypeError("tolerances must be a Tolerances instance")
    basis = reduce_basis(vectors)
    first, second = basis[:, :2]
    radius = 2.0 * (np.linalg.norm(first) + np.linalg.norm(second))
    vertices = np.array(
        [[-radius, -radius], [radius, -radius], [radius, radius], [-radius, radius]]
    )
    translations = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            if i or j:
                translations.append(i * first + j * second)
    translations.sort(key=lambda item: float(np.dot(item, item)))
    for translation in translations:
        squared = float(np.dot(translation, translation))
        vertices = _clip_half_plane(
            vertices, translation, squared / 2, tolerances.relative * squared + tolerances.absolute
        )
    tolerance = (
        np.finfo(float).eps
        * max(float(np.max(np.linalg.norm(vertices, axis=1))), np.finfo(float).tiny)
        * 128
    )
    keep = np.linalg.norm(vertices - np.roll(vertices, 1, axis=0), axis=1) > tolerance
    vertices = vertices[keep]
    domain = PolygonDomain(np.column_stack((vertices, np.zeros(len(vertices)))))
    return UnitCell(basis, domain, [[0.0, 0.0]], UnitCellKind.WIGNER_SEITZ)
