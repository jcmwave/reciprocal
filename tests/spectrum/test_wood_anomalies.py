from dataclasses import FrozenInstanceError

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Circle

from reciprocal import KSpace, Lattice, MaxSpacing, PointCounts
from reciprocal.canvas import Canvas
from reciprocal.spectrum import (
    CircleArc,
    PropagationDisk,
    WoodAnomalyCircle,
    WoodAnomalySet,
    circle_disk_intersection,
)


@pytest.mark.parametrize("scale", [1e-6, 1, 1e6])
def test_circle_edge_intersections_handle_full_partial_tangent_and_enclosing_cases(scale):
    domain = PropagationDisk(scale)
    arc, contact = circle_disk_intersection([0, 0], scale / 2, domain)
    assert arc.closed and contact is None
    assert arc.length == pytest.approx(np.pi * scale)
    arc, contact = circle_disk_intersection([scale, 0], scale, domain)
    assert contact is None
    assert arc.length == pytest.approx(2 * np.pi * scale / 3)
    assert np.all(domain.contains(arc.points_at(np.linspace(arc.theta_start, arc.theta_end, 101))))
    arc, contact = circle_disk_intersection([2 * scale, 0], scale, domain)
    assert arc is None
    np.testing.assert_allclose(contact, [scale, 0], atol=scale * 1e-14)
    assert circle_disk_intersection([3 * scale, 0], scale, domain) == (None, None)
    # Filled circles overlap, but their edges do not enter the propagating disk.
    assert circle_disk_intersection([0, 0], 2 * scale, domain) == (None, None)
    assert circle_disk_intersection([0.2 * scale, 0], 2 * scale, domain) == (None, None)


def _anomalies():
    space = KSpace.propagating(2 * np.pi)
    lattice = Lattice.from_lengths_angle(2 * np.pi, 2 * np.pi, 90)
    return space.with_periodic_structure(lattice).wood_anomalies()


def test_diffraction_order_sign_shell_metadata_and_tangent_circles():
    anomalies = _anomalies()
    assert len(anomalies.circles) == 12
    assert set(anomalies.circles_by_shell) == {1, 2, 3}
    for circle in anomalies.circles:
        np.testing.assert_allclose(circle.center, -np.asarray(circle.reciprocal_order), atol=1e-14)
        assert circle.radius == 1
        if circle.arc is not None:
            points = circle.arc.points_at(
                np.linspace(circle.arc.theta_start, circle.arc.theta_end, 101)
            )
            np.testing.assert_allclose(np.linalg.norm(points + circle.reciprocal_order, axis=1), 1)
            assert np.all(anomalies.propagating_domain.contains(points))
    assert sum(circle.tangent_point is not None for circle in anomalies.circles) == 4
    assert len(_anomalies().sample_edges().tangent_points) == 4
    with pytest.raises(FrozenInstanceError):
        anomalies.circles[0].radius = 2


@pytest.mark.parametrize("constraint", [PointCounts(1), PointCounts(7), MaxSpacing(0.07)])
def test_arc_sampling_weights_spacing_segments_and_provenance(constraint):
    anomalies = _anomalies()
    sampled = anomalies.sample_edges(constraint)
    assert np.all(anomalies.propagating_domain.contains(sampled.points))
    length = sum(circle.arc.length for circle in anomalies.circles if circle.arc is not None)
    assert sampled.integrate(np.ones(len(sampled.points))) == pytest.approx(length)
    assert sampled.normalized_weights.sum() == pytest.approx(1)
    assert sampled.average(np.ones(len(sampled.points))) == pytest.approx(1)
    for first, last in sampled.segment_boundaries:
        index = sampled.circle_indices[first]
        assert np.all(sampled.circle_indices[first:last] == index)
        circle = anomalies.circles[index]
        np.testing.assert_array_equal(
            sampled.reciprocal_orders[first:last],
            np.tile(circle.reciprocal_order, (last - first, 1)),
        )
        assert sampled.physical_weights[first:last].sum() == pytest.approx(circle.arc.length)
        if isinstance(constraint, PointCounts):
            assert last - first == constraint.first
        else:
            points = sampled.points[first:last] - circle.center
            angles = np.unwrap(np.arctan2(points[:, 1], points[:, 0]))
            assert np.max(np.diff(angles)) * circle.radius <= constraint.value * (1 + 1e-12)
    assert not sampled.points.flags.writeable
    assert not sampled.physical_weights.flags.writeable


def test_closed_curve_omits_duplicate_closing_endpoint_and_integrates_vector_values():
    arc = CircleArc((0, 0), 1, 0, 2 * np.pi)
    anomalies = WoodAnomalySet((WoodAnomalyCircle((0, 0), 1, (0, 0), 1, arc),), PropagationDisk(1))
    sampled = anomalies.sample_edges(PointCounts(16))
    assert len(np.unique(sampled.points, axis=0)) == 16
    np.testing.assert_allclose(sampled.integrate(sampled.points), [0, 0], atol=1e-14)
    np.testing.assert_allclose(sampled.physical_weights, 2 * np.pi / 16)


def test_empty_and_tangent_only_results_report_contacts_without_normalizing_zero_length():
    empty = WoodAnomalySet((), PropagationDisk(1)).sample_edges()
    assert empty.points.shape == (0, 2)
    assert empty.normalized_weights.shape == (0,)
    assert empty.integrate([]) == 0
    with pytest.raises(ValueError, match="zero-length"):
        empty.average([])
    tangent = WoodAnomalyCircle((2, 0), 1, (-2, 0), 1, None, (1, 0))
    sampled = WoodAnomalySet((tangent,), PropagationDisk(1)).sample_edges()
    np.testing.assert_array_equal(sampled.tangent_points, [[1, 0]])
    assert len(sampled.points) == 0


def test_canvas_draws_clipped_circles_and_accepts_curve_samples():
    anomalies = _anomalies()
    sampled = anomalies.sample_edges(PointCounts(7))
    figure, axes = plt.subplots()
    try:
        canvas = Canvas(axes)
        artists = canvas.plot_wood_anomalies(anomalies, linewidth=2)
        assert len(artists) == len(anomalies.circles)
        for artist, circle in zip(artists, anomalies.circles):
            assert isinstance(artist, Circle)
            assert artist.axes is axes
            assert artist.center == circle.center
            assert artist.radius == circle.radius
            assert artist.get_clip_path() is not None
        points = canvas.plot_point_sampling_weighted(sampled, physical=True)
        assert len(points.get_offsets()) == len(sampled.points)
        figure.canvas.draw()
    finally:
        plt.close(figure)


def test_invalid_shells_and_curve_constraints_are_rejected():
    space = KSpace.propagating(2 * np.pi)
    periodic = space.with_periodic_structure(Lattice.from_lengths_angle(2 * np.pi, 2 * np.pi, 90))
    for shells in [(0,), (-1,), (1.5,), (True,), (1, 1)]:
        with pytest.raises(ValueError, match="shell"):
            periodic.wood_anomalies(shells=shells)
    with pytest.raises(ValueError, match="dimensions"):
        _anomalies().sample_edges(PointCounts(5, 5))
