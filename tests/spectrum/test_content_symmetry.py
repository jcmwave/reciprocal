from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Circle, Rectangle, Wedge

from reciprocal import KSpace, MaxSpacing, OrbitIntegrationCell
from reciprocal.canvas import Canvas
from reciprocal.spectrum import (
    AxialVectorRepresentation,
    CartesianGrid,
    CircularGrid,
    ContentSymmetry,
    FieldSamples,
    PolarGrid,
    PolarVectorRepresentation,
    PropagationDisk,
    ScalarRepresentation,
    SPPolarizationRepresentation,
    SymmetryCone,
    TensorRepresentation,
)


def _space():
    return KSpace.propagating(2 * np.pi)


@pytest.mark.parametrize("grid", [CartesianGrid((9, 9)), CircularGrid(5), PolarGrid(4, 16)])
def test_d4_reduction_preserves_full_domain_weights_identities_and_physical_orbits(grid):
    sampled = _space().sample_domain(
        grid,
        content_symmetry=ContentSymmetry("D4"),
        symmetry_cone="irreducible",
        include_cells=True,
    )
    mapping = sampled.symmetry_reduction
    assert mapping is not None
    target = mapping.target
    assert np.all(SymmetryCone(0, 45).contains(sampled.points))
    assert len(sampled.points) < len(target.points)
    assert sampled.integration_element == target.integration_element == pytest.approx(np.pi)
    assert mapping.speedup == pytest.approx(len(target.points) / len(sampled.points))
    expected = np.bincount(mapping.full_to_reduced, weights=target.physical_weights)
    np.testing.assert_allclose(sampled.physical_weights, expected)
    assert len(np.unique(target.point_ids)) == len(target.points)
    assert set(sampled.point_ids).issubset(set(target.point_ids))
    assert not mapping.full_to_reduced.flags.writeable
    assert not mapping.source_points.flags.writeable
    for point, index, operations in zip(target.points, mapping.full_to_reduced, mapping.operations):
        for operation in operations:
            np.testing.assert_allclose(
                operation.apply(sampled.points[index])[:2], point, atol=1e-14
            )
    for cell, weight in zip(sampled.integration_cells, sampled.physical_weights):
        assert isinstance(cell, OrbitIntegrationCell)
        assert cell.area == pytest.approx(weight)
    with_kz = sampled.with_kz(1.0, -1)
    assert with_kz.symmetry_reduction is mapping


def test_origin_and_axis_degeneracies_are_not_multiplied_by_eight():
    sampled = _space().sample_domain(
        CartesianGrid((9, 9)), content_symmetry="D4", symmetry_cone=(0, 45)
    )
    target = sampled.symmetry_reduction.target
    sizes = np.bincount(sampled.symmetry_reduction.full_to_reduced)
    for point, size, weight in zip(sampled.points, sizes, sampled.normalized_weights):
        if np.allclose(point, 0):
            assert size == 1
        elif abs(point[1]) < 1e-12 or np.isclose(point[0], point[1]):
            assert size == 4
        else:
            assert size == 8
        assert weight == pytest.approx(size / len(target.points))


@pytest.mark.parametrize(
    "name,angle", [("C1", 0), ("C4", 13), ("D2", 10), ("D3", 0), ("D4", 17), ("D6", 30)]
)
def test_rotated_and_cyclic_content_groups_reconstruct_scalar_fields(name, angle):
    sampled = _space().sample_domain(
        CircularGrid(5), content_symmetry=ContentSymmetry(name, angle), symmetry_cone="irreducible"
    )
    data = FieldSamples(sampled, np.sum(sampled.points**2, axis=1) + 1j, ScalarRepresentation())
    full = sampled.symmetry_reduction.expand(data)
    np.testing.assert_allclose(full.values, np.sum(full.sampling.points**2, axis=1) + 1j)
    assert full.sampling.kz is not None
    np.testing.assert_allclose(data.integrate(), full.integrate())
    np.testing.assert_allclose(data.average(), full.average())


def test_cartesian_vectors_and_tensors_expand_covariantly_and_integrate_correctly():
    sampled = _space().sample_domain(
        CircularGrid(5), content_symmetry="D4", symmetry_cone="irreducible"
    )
    vectors = np.column_stack((sampled.points, np.ones(len(sampled.points))))
    data = FieldSamples(sampled, vectors, PolarVectorRepresentation())
    full = sampled.symmetry_reduction.expand(data)
    np.testing.assert_allclose(
        full.values, np.column_stack((full.sampling.points, np.ones(len(full.values)))), atol=1e-14
    )
    np.testing.assert_allclose(data.integrate(), [0, 0, np.pi], atol=1e-14)
    tensors = np.einsum("ni,nj->nij", vectors, vectors)
    tensor_data = FieldSamples(sampled, tensors, TensorRepresentation())
    tensor_full = sampled.symmetry_reduction.expand(tensor_data)
    np.testing.assert_allclose(
        tensor_full.values, np.einsum("ni,nj->nij", full.values, full.values), atol=1e-14
    )


def test_axial_vectors_include_the_reflection_determinant():
    sampled = _space().sample_domain(
        CircularGrid(5), content_symmetry="D4", symmetry_cone="irreducible"
    )
    x, y = sampled.points.T
    vectors = np.column_stack((np.zeros(len(x)), np.zeros(len(x)), x * y * (x * x - y * y)))
    data = FieldSamples(sampled, vectors, AxialVectorRepresentation())
    full = sampled.symmetry_reduction.expand(data)
    x, y = full.sampling.points.T
    expected = np.column_stack((np.zeros(len(x)), np.zeros(len(x)), x * y * (x * x - y * y)))
    np.testing.assert_allclose(full.values, expected, atol=1e-14)
    np.testing.assert_allclose(data.integrate(), [0, 0, 0], atol=1e-14)


def test_polarization_channels_follow_local_frame_transformations():
    sampled = _space().sample_domain(
        PolarGrid(3, 12), content_symmetry="D4", symmetry_cone="irreducible"
    )
    values = np.tile([0.0, 1.0], (len(sampled.points), 1))
    data = FieldSamples(sampled, values, SPPolarizationRepresentation(1.0))
    full = sampled.symmetry_reduction.expand(data)
    np.testing.assert_allclose(full.values, np.tile([0, 1], (len(full.values), 1)), atol=1e-13)


def test_inconsistent_symmetry_boundary_response_and_wrong_source_are_rejected():
    sampled = _space().sample_domain(
        CircularGrid(3), content_symmetry="D4", symmetry_cone="irreducible"
    )
    bad = FieldSamples(
        sampled, np.tile([1, 0, 0], (len(sampled.points), 1)), PolarVectorRepresentation()
    )
    with pytest.raises(ValueError, match="inconsistent"):
        sampled.symmetry_reduction.expand(bad)
    data = FieldSamples(sampled, np.ones(len(sampled.points)), ScalarRepresentation())
    wrong = replace(sampled, point_ids=sampled.point_ids + 1000)
    with pytest.raises(ValueError, match="does not match"):
        sampled.symmetry_reduction.expand(FieldSamples(wrong, data.values, data.representation))


def test_incompatible_explicit_grids_raise_or_are_adapted_explicitly():
    space = _space()
    for grid in [CartesianGrid((7, 9)), PolarGrid(3, 10), CircularGrid(5, 26)]:
        with pytest.raises(ValueError, match="symmetry"):
            space.sample_domain(grid, content_symmetry="D4", symmetry_cone="irreducible")
        adapted = space.sample_domain(
            grid, content_symmetry="D4", symmetry_cone="irreducible", adapt_symmetry=True
        )
        assert adapted.symmetry_reduction is not None
    with pytest.raises(ValueError, match="grid does not preserve"):
        space.sample_domain(
            CartesianGrid((9, 9)),
            content_symmetry=ContentSymmetry("D4", 17),
            symmetry_cone="irreducible",
            adapt_symmetry=True,
        )


def test_cone_coverage_and_domain_compatibility_are_checked():
    space = _space()
    with pytest.raises(ValueError, match="requires content_symmetry"):
        space.sample_domain(CircularGrid(5), symmetry_cone=(0, 45))
    with pytest.raises(ValueError, match="every physical orbit"):
        space.sample_domain(CircularGrid(5), content_symmetry="D4", symmetry_cone=(10, 20))
    with pytest.raises(ValueError, match="domain does not preserve"):
        space.sample_domain(
            CartesianGrid((9, 9)),
            domain=PropagationDisk(1, (0.1, 0)),
            content_symmetry="D4",
            symmetry_cone="irreducible",
        )
    sampled = space.sample_domain(CircularGrid(5), content_symmetry="D4")
    assert sampled.symmetry_reduction is None
    assert sampled.metadata["content_symmetry"].name == "D4"


def test_maximum_spacing_is_preserved_by_symmetry_compatible_circular_counts():
    sampled = _space().sample_domain(
        "circular",
        constraint=MaxSpacing((0.2, 0.1)),
        content_symmetry="D4",
        symmetry_cone="irreducible",
        include_cells=True,
    )
    for cell in sampled.symmetry_reduction.target.integration_cells[1:]:
        radius = (cell.inner_radius + cell.outer_radius) / 2
        assert radius * (cell.theta_end - cell.theta_start) <= 0.1 * (1 + 1e-12)
        assert cell.outer_radius - cell.inner_radius <= 0.2 * (1 + 1e-12)


@pytest.mark.parametrize(
    "grid,types", [(CartesianGrid((9, 9)), (Rectangle,)), (CircularGrid(5), (Circle, Wedge))]
)
def test_representative_artists_show_footprints_and_keep_orbit_allocations(grid, types):
    sampled = _space().sample_domain(
        grid, content_symmetry="D4", symmetry_cone="irreducible", include_cells=True
    )
    figure, axes = plt.subplots()
    try:
        artists = Canvas(axes).plot_integration_cells(sampled)
        assert len(artists) == len(sampled.points)
        assert all(isinstance(artist, types) for artist in artists)
        figure.canvas.draw()
    finally:
        plt.close(figure)


def test_declared_orientation_cone_and_operation_arrays_are_immutable():
    symmetry = ContentSymmetry("D4", 30)
    assert symmetry.cone == SymmetryCone(30, 75)
    assert len(symmetry.operations) == 8
    with pytest.raises(ValueError):
        symmetry.operations[0].cartesian.setflags(write=True)
    assert SymmetryCone(350, 370).contains([[1, 0], [0, 0]]).all()
    with pytest.raises(ValueError):
        ContentSymmetry("D0")
    with pytest.raises(ValueError):
        SymmetryCone(45, 0)


def test_associating_content_symmetry_checks_the_full_grid_even_without_reduction():
    with pytest.raises(ValueError, match="grid does not preserve"):
        _space().sample_domain(CartesianGrid((7, 9)), content_symmetry="D4")


@pytest.mark.parametrize("scheme", ["cartesian", "polar", "circular"])
def test_spacing_constraints_choose_compatible_counts_automatically(scheme):
    result = _space().sample_domain(
        scheme,
        constraint=MaxSpacing((0.17, 0.23)),
        content_symmetry="D4",
        symmetry_cone="irreducible",
    )
    assert result.symmetry_reduction is not None
    assert result.integrate(np.ones(len(result.points))) == pytest.approx(np.pi)


@pytest.mark.parametrize("orientation", [-10, 23, 321, 370])
def test_wrapping_orientations_and_single_sector_circular_cells(orientation):
    result = _space().sample_domain(
        CircularGrid(2, 1),
        content_symmetry=ContentSymmetry("C1", orientation),
        symmetry_cone="irreducible",
        include_cells=True,
    )
    assert len(result.points) == 2
    assert result.integrate(np.ones(2)) == pytest.approx(np.pi)
    field = FieldSamples(result, np.ones(2), ScalarRepresentation())
    assert field.integrate() == pytest.approx(np.pi)


def test_invalid_field_integration_reports_invalid_samples_before_transforming_them():
    sampled = _space().sample_domain(
        CircularGrid(3), content_symmetry="D4", symmetry_cone="irreducible"
    )
    data = FieldSamples(
        sampled,
        np.tile([1, 0, 0], (len(sampled.points), 1)),
        PolarVectorRepresentation(),
        valid=np.zeros(len(sampled.points), dtype=bool),
    )
    with pytest.raises(ValueError, match="invalid field"):
        data.integrate()
