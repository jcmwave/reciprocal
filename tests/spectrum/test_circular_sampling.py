from dataclasses import FrozenInstanceError

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Circle, Wedge

from reciprocal import CircularIntegrationCell, KSpace, PolarIntegrationCell
from reciprocal.canvas import Canvas
from reciprocal.cells import PolygonDomain
from reciprocal.spectrum import (
    CircularGrid,
    EvanescentDisk,
    NonPeriodicSampler,
    PolygonKDomain,
    PropagationDisk,
    PupilDomain,
)


@pytest.mark.parametrize("radial", [1, 2, 5, 12])
@pytest.mark.parametrize("radius", [1e-6, 1.0, 1e6])
def test_circular_grid_has_one_origin_fixed_ring_spacing_and_comparable_arc_spacing(radial, radius):
    sampling = NonPeriodicSampler().sample(PropagationDisk(radius), CircularGrid(radial))
    lengths = np.linalg.norm(sampling.points, axis=1)
    assert np.count_nonzero(lengths == 0.0) == 1
    np.testing.assert_array_equal(sampling.points[0], [0.0, 0.0])
    assert sampling.integration_cells is None
    spacing = radius / (radial - 0.5)
    np.testing.assert_allclose(lengths / spacing, np.rint(lengths / spacing), atol=1e-12)
    counts = []
    for ring in range(1, radial):
        points = sampling.points[np.isclose(lengths / spacing, ring)]
        count = len(points)
        counts.append(count)
        assert 2 * np.pi * ring / count == pytest.approx(1.0, rel=0.06)
        angles = np.sort(np.arctan2(points[:, 1], points[:, 0]) % (2 * np.pi))
        gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
        np.testing.assert_allclose(gaps, 2 * np.pi / count)
        assert not np.any(np.all(np.isclose(points / radius, 0.0), axis=1))
    assert sum(counts) + 1 == len(sampling.points)
    assert all(first < second for first, second in zip(counts[:-1], counts[1:]))
    assert np.all(lengths <= radius)


@pytest.mark.parametrize("radial", [1, 2, 5, 12])
def test_circle_and_wedges_tile_disk_and_keep_weights_aligned(radial):
    radius = 2.0
    domain = PropagationDisk(radius)
    sampler = NonPeriodicSampler()
    sampling = sampler.sample(domain, CircularGrid(radial), include_cells=True)
    plain = sampler.sample(domain, CircularGrid(radial))
    np.testing.assert_array_equal(sampling.points, plain.points)
    np.testing.assert_array_equal(sampling.normalized_weights, plain.normalized_weights)
    cells = sampling.integration_cells
    assert isinstance(cells, tuple)
    assert len(cells) == len(sampling.points)
    assert isinstance(cells[0], CircularIntegrationCell)
    assert cells[0].radius == pytest.approx(radius / (2 * radial - 1))
    assert cells[0].center == (0.0, 0.0)
    assert sum(cell.area for cell in cells) == pytest.approx(domain.area)
    np.testing.assert_allclose([cell.area for cell in cells], sampling.physical_weights)
    assert sampling.integrate(np.ones(len(cells))) == pytest.approx(domain.area)
    assert sampling.with_kz(1.0).integration_cells == cells
    with pytest.raises(FrozenInstanceError):
        cells[0].radius = 0.0
    for point, cell in zip(sampling.points[1:], cells[1:]):
        assert isinstance(cell, PolarIntegrationCell)
        assert np.linalg.norm(point) == pytest.approx((cell.inner_radius + cell.outer_radius) / 2)
        middle_angle = (cell.theta_start + cell.theta_end) / 2
        np.testing.assert_allclose(
            point / np.linalg.norm(point), [np.cos(middle_angle), np.sin(middle_angle)], atol=1e-14
        )
    if radial > 1:
        rings = sorted(set((cell.inner_radius, cell.outer_radius) for cell in cells[1:]))
        assert rings[0][0] == pytest.approx(cells[0].radius)
        assert rings[-1][1] == pytest.approx(radius)
        for inner, outer in rings:
            wedges = [cell for cell in cells[1:] if cell.inner_radius == inner]
            assert sum(cell.theta_end - cell.theta_start for cell in wedges) == pytest.approx(
                2 * np.pi
            )
            assert sum(cell.area for cell in wedges) == pytest.approx(np.pi * (outer**2 - inner**2))
        for first, second in zip(rings[:-1], rings[1:]):
            assert first[1] == pytest.approx(second[0])


def test_circular_sampler_facades_and_artists():
    space = KSpace.propagating(2 * np.pi)
    sampling = space.sample_domain(CircularGrid(5), include_cells=True, direction=-1)
    np.testing.assert_array_equal(sampling.points[0], [0.0, 0.0])
    assert sampling.kz[0] == pytest.approx(-1.0)
    pupil = space.sample_pupil(CircularGrid(5), numerical_aperture=0.5, include_cells=True)
    assert pupil.integration_cells[-1].outer_radius == pytest.approx(0.5)
    extended = KSpace.extended(2 * np.pi, 1.0, 2.0).sample_domain(CircularGrid(5))
    assert np.any(np.imag(extended.kz) > 0.0)
    first, axes = plt.subplots()
    second, other_axes = plt.subplots()
    try:
        artists = Canvas(axes).plot_integration_cells(sampling, edgecolor="red", linewidth=2.0)
        other = Canvas(other_axes).plot_integration_cells(sampling)
        assert isinstance(artists[0], Circle)
        assert artists[0].radius == pytest.approx(sampling.integration_cells[0].radius)
        assert all(isinstance(artist, Wedge) for artist in artists[1:])
        for artist, fresh in zip(artists, other):
            assert artist.axes is axes
            assert fresh.axes is other_axes
            assert artist is not fresh
            assert artist.get_linewidth() == 2.0
        first.canvas.draw()
    finally:
        plt.close(first)
        plt.close(second)


@pytest.mark.parametrize(
    "domain", [PropagationDisk(1.0), EvanescentDisk(1.0, 2.0), PupilDomain(1.0)]
)
def test_circular_sampling_supports_complete_disk_domains(domain):
    sampling = NonPeriodicSampler().sample(domain, CircularGrid(4))
    assert sampling.integrate(np.ones(len(sampling.points))) == pytest.approx(domain.area)


@pytest.mark.parametrize(
    "domain",
    [
        PupilDomain(1.0, 0.3),
        PropagationDisk(1.0, (0.1, 0.0)),
        PupilDomain(1.0, center=(0.0, 0.1)),
        PolygonKDomain(PolygonDomain([[-1, -1], [1, -1], [1, 1], [-1, 1]])),
    ],
)
def test_circular_grid_rejects_domains_that_cannot_tile_origin_centered_cells(domain):
    with pytest.raises(ValueError, match="origin-centered disk"):
        NonPeriodicSampler().sample(domain, CircularGrid())


@pytest.mark.parametrize("radial", [0, -1, 1.5, np.nan, np.inf])
def test_circular_grid_requires_positive_integer_radial_count(radial):
    with pytest.raises(ValueError, match="positive integer"):
        CircularGrid(radial)


def test_circular_grid_rejects_boolean_counts():
    with pytest.raises(TypeError, match="integer"):
        CircularGrid(True)
