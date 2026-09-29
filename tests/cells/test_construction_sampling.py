import numpy as np
import pytest

from reciprocal.bravais import BravaisLattice
from reciprocal.cells import (
    CellSampler,
    UnitCellKind,
    make_conventional_cell,
    make_primitive_cell,
    make_wigner_seitz_cell,
)
from reciprocal.lattice import Lattice, LatticeVectors


@pytest.mark.parametrize(
    "length1,length2,angle,bravais",
    [
        (1.0, 1.3, 70.0, BravaisLattice.OBLIQUE),
        (1.0, 2.0, 90.0, BravaisLattice.RECTANGULAR),
        (1.0, 1.0, 70.0, BravaisLattice.CENTERED_RECTANGULAR),
        (1.0, 1.0, 90.0, BravaisLattice.SQUARE),
        (1.0, 1.0, 60.0, BravaisLattice.HEXAGONAL),
        (1.0, 1.0, 120.0, BravaisLattice.HEXAGONAL),
    ],
)
def test_cell_area_and_multiplicity_invariants(length1, length2, angle, bravais):
    vectors = LatticeVectors.from_lengths_angle(length1, length2, angle)
    primitive = make_primitive_cell(vectors)
    conventional = make_conventional_cell(vectors, bravais)
    wigner_seitz = make_wigner_seitz_cell(vectors)

    expected_area = abs(np.linalg.det(np.vstack((vectors.vec1[:2], vectors.vec2[:2]))))
    assert primitive.kind is UnitCellKind.PRIMITIVE
    assert primitive.area == pytest.approx(expected_area)
    assert wigner_seitz.area == pytest.approx(expected_area)
    assert conventional.primitive_area == pytest.approx(expected_area)
    expected_multiplicity = 2 if bravais is BravaisLattice.CENTERED_RECTANGULAR else 1
    assert conventional.multiplicity == expected_multiplicity


def test_cell_sampler_uses_normalized_weights_and_physical_area():
    cell = make_wigner_seitz_cell(LatticeVectors.from_lengths_angle(1.0, 1.0, 60.0))
    result = CellSampler().sample(cell, {"type": "n_points", "value": 9})

    assert np.sum(result.weights) == pytest.approx(1.0)
    assert result.integration_element == pytest.approx(cell.area)
    assert np.dot(
        result.weights, np.ones(len(result.points))
    ) * result.integration_element == pytest.approx(cell.area)


@pytest.mark.parametrize(
    "parameters",
    [(1.0, 1.0, 90.0), (1.0, 1.0, 60.0), (1.0, 2.0, 90.0), (1.0, 1.0, 70.0)],
)
def test_bravais_classification_is_invariant_under_unimodular_basis_change(parameters):
    original = LatticeVectors.from_lengths_angle(*parameters)
    original_basis = np.vstack((original.vec1, original.vec2))
    expected = Lattice(original).bravais
    transformation = np.array([[2, 1], [1, 1]])
    transformed = transformation @ original_basis

    assert Lattice(LatticeVectors(transformed[0], transformed[1])).bravais is expected
