from dataclasses import FrozenInstanceError, replace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Rectangle, Wedge

from reciprocal import CartesianIntegrationCell, KSpace, PolarIntegrationCell
from reciprocal.canvas import Canvas
from reciprocal.cells import PolygonDomain
from reciprocal.spectrum import (
    CartesianGrid,
    IntersectionDomain,
    NonPeriodicSampler,
    PointCounts,
    PolarGrid,
    PolygonKDomain,
    PropagationDisk,
    PupilDomain,
)


@pytest.mark.parametrize("grid", [CartesianGrid((6, 7)), PolarGrid(4, 12), PointCounts(5)])
def test_retained_cells_are_aligned_optional_and_preserved_with_kz(grid):
    space = KSpace.propagating(2 * np.pi)
    plain = space.sample_domain(grid)
    sampling = space.sample_domain(grid, include_cells=True)
    assert plain.integration_cells is None
    assert isinstance(sampling.integration_cells, tuple)
    assert len(sampling.integration_cells) == len(sampling.points)
    np.testing.assert_array_equal(sampling.points, plain.points)
    np.testing.assert_array_equal(sampling.physical_weights, plain.physical_weights)
    np.testing.assert_allclose(
        [cell.area for cell in sampling.integration_cells], sampling.physical_weights
    )
    changed = sampling.with_kz(1.0, direction=-1)
    assert changed.integration_cells == sampling.integration_cells
    np.testing.assert_array_equal(changed.kz, -sampling.kz)
    with pytest.raises(FrozenInstanceError):
        if isinstance(sampling.integration_cells[0], CartesianIntegrationCell):
            sampling.integration_cells[0].area = 0.0
        else:
            sampling.integration_cells[0].inner_radius = 0.0
    with pytest.raises(ValueError, match="one cell per point"):
        replace(sampling, integration_cells=sampling.integration_cells[:-1])
    with pytest.raises(ValueError, match="match physical_weights"):
        first = sampling.integration_cells[0]
        if isinstance(first, CartesianIntegrationCell):
            first = replace(first, area=first.area * 2)
        else:
            first = replace(first, outer_radius=first.outer_radius * 2)
        replace(sampling, integration_cells=(first, *sampling.integration_cells[1:]))


def test_pupil_wrapper_retains_shifted_annular_sectors():
    space = KSpace.propagating(2 * np.pi)
    sampling = space.sample_pupil(
        PolarGrid(3, 7), domain=PupilDomain(1.0, 0.3, (0.1, -0.2)), include_cells=True
    )
    cells = sampling.integration_cells
    assert all(isinstance(cell, PolarIntegrationCell) for cell in cells)
    assert cells[0].center == (0.1, -0.2)
    assert cells[0].inner_radius == pytest.approx(0.3)
    assert cells[-1].outer_radius == pytest.approx(1.0)
    assert cells[-1].theta_end == pytest.approx(2 * np.pi)
    numeric = space.sample_pupil(PolarGrid(2, 4), numerical_aperture=0.5, include_cells=True)
    assert numeric.integration_cells[-1].outer_radius == pytest.approx(0.5)


_POLYGON = PolygonKDomain(PolygonDomain([[-0.9, -0.8], [0.9, -0.7], [0.2, 0.9]]))


@pytest.mark.parametrize(
    "domain",
    [
        PropagationDisk(1.0),
        PropagationDisk(2e6, (1e6, -2e6)),
        PupilDomain(1.0, 0.65),
        PupilDomain(1.0, 0.2, (0.3, -0.1)),
        _POLYGON,
        IntersectionDomain((_POLYGON, PupilDomain(1.0, 0.3))),
    ],
)
@pytest.mark.parametrize("shape", [(3, 3), (6, 7)])
def test_cartesian_artists_are_full_uniform_rectangles_centered_on_points(domain, shape):
    sampling = NonPeriodicSampler().sample(domain, CartesianGrid(shape), include_cells=True)
    figure, axes = plt.subplots()
    try:
        artists = Canvas(axes).plot_integration_cells(sampling, facecolor="lightgray")
        assert len(artists) == len(sampling.points)
        for artist, cell, point, weight in zip(
            artists, sampling.integration_cells, sampling.points, sampling.physical_weights
        ):
            assert isinstance(cell, CartesianIntegrationCell)
            assert isinstance(artist, Rectangle)
            assert artist.axes is axes
            x0, x1, y0, y1 = cell.bounds
            np.testing.assert_allclose(point, [(x0 + x1) / 2, (y0 + y1) / 2], atol=1e-14)
            assert artist.get_width() == pytest.approx(
                (domain.bounds()[1] - domain.bounds()[0]) / shape[0]
            )
            assert artist.get_height() == pytest.approx(
                (domain.bounds()[3] - domain.bounds()[2]) / shape[1]
            )
            assert cell.geometric_area == pytest.approx(artist.get_width() * artist.get_height())
            assert cell.area == pytest.approx(weight)
            assert weight == pytest.approx(domain.area / len(sampling.points))
        assert axes.get_xlim()[0] <= domain.bounds()[0]
        assert axes.get_xlim()[1] >= domain.bounds()[1]
        figure.canvas.draw()
    finally:
        plt.close(figure)


@pytest.mark.parametrize("azimuthal", [1, 7])
def test_polar_plot_returns_fresh_styleable_wedges(azimuthal):
    sampling = KSpace.propagating(2 * np.pi).sample_domain(
        PolarGrid(3, azimuthal), include_cells=True
    )
    first, first_axes = plt.subplots()
    second, second_axes = plt.subplots()
    try:
        artists = Canvas(first_axes).plot_integration_cells(sampling, edgecolor="red", linewidth=2)
        other = Canvas(second_axes).plot_integration_cells(sampling)
        for artist, new, cell in zip(artists, other, sampling.integration_cells):
            assert isinstance(artist, Wedge)
            assert artist is not new
            assert artist.axes is first_axes
            assert new.axes is second_axes
            assert artist.get_linewidth() == 2
            assert artist.r == pytest.approx(cell.outer_radius)
            assert artist.width == pytest.approx(cell.outer_radius - cell.inner_radius)
    finally:
        plt.close(first)
        plt.close(second)


def test_polar_polygon_fallback_retains_cartesian_cells():
    sampling = NonPeriodicSampler().sample(_POLYGON, PolarGrid(3, 8), include_cells=True)
    assert all(isinstance(cell, CartesianIntegrationCell) for cell in sampling.integration_cells)


def test_plotter_explains_missing_cells_without_changing_axes():
    figure, axes = plt.subplots()
    try:
        sampling = KSpace.propagating(2 * np.pi).sample_domain(PolarGrid(2, 4))
        with pytest.raises(ValueError, match="include_cells=True"):
            Canvas(axes).plot_integration_cells(sampling)
        assert not axes.patches
    finally:
        plt.close(figure)


@pytest.mark.parametrize("radius", [1e-6, 1.0, 1e6])
@pytest.mark.parametrize("shape", [(1, 1), (5, 7), (6, 7)])
def test_tangent_rectangle_intersection_areas_partition_the_disk(radius, shape):
    from reciprocal.spectrum.domains import intersection_area

    domain = PropagationDisk(radius)
    x_edges = np.linspace(-radius, radius, shape[0] + 1)
    y_edges = np.linspace(-radius, radius, shape[1] + 1)
    area = sum(
        intersection_area(domain, PolygonDomain([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
        for x0, x1 in zip(x_edges[:-1], x_edges[1:])
        for y0, y1 in zip(y_edges[:-1], y_edges[1:])
    )
    assert area == pytest.approx(domain.area, rel=1e-12)


def test_cartesian_boundary_cell_keeps_its_full_rectangle():
    domain = PropagationDisk(1.0)
    sampling = NonPeriodicSampler().sample(domain, CartesianGrid((9, 9)), include_cells=True)
    assert any(
        not np.all(domain.contains([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
        for x0, x1, y0, y1 in (cell.bounds for cell in sampling.integration_cells)
    )
    assert sampling.integration_cells[0].geometric_area != pytest.approx(
        sampling.integration_cells[0].area
    )
