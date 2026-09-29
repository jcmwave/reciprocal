import warnings

import matplotlib.pyplot as plt
import numpy as np

from reciprocal import KSpace, Lattice
from reciprocal.canvas import Canvas
from reciprocal.spectrum import (
    FieldSamples,
    PointCounts,
    PolarGrid,
    ScalarRepresentation,
    interpolate,
)


def _objects():
    space = KSpace.propagating(2.0 * np.pi, 1.0)
    direct = Lattice.from_lengths_angle(2.0 * np.pi, 2.0 * np.pi, 90.0)
    reciprocal = direct.make_reciprocal()
    plan = space.with_periodic_structure(direct).plan(
        source="ibz", target="bz", constraint=PointCounts(5)
    )
    pupil = space.sample_pupil(PolarGrid(3, 12))
    field = FieldSamples(
        pupil,
        pupil.points[:, 0] + 1j * pupil.points[:, 1],
        ScalarRepresentation(),
    )
    return space, direct, reciprocal, plan, pupil, field


def test_canvas_uses_immutable_lattice_and_zone_objects_without_legacy_cell():
    _space, direct, reciprocal, plan, _pupil, _field = _objects()
    figure, axes = plt.subplots()
    canvas = Canvas(axes)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        canvas.plot_tessellation(direct, orders=np.array([[0, 0], [1, 0]]))
        canvas.plot_tessellation(reciprocal, orders=np.array([[0, 0]]))
        canvas.plot_unit_cell(direct)
        canvas.plot_unit_cell(reciprocal)
        canvas.plot_bzone(plan.zone)
        canvas.plot_ibzone(plan)
        special = canvas.plot_special_points(reciprocal)

    assert len(special) == len(plan.zone.special_points)
    plt.close(figure)


def test_canvas_accepts_sampling_weights_fields_and_expansion_plans():
    space, _direct, _reciprocal, plan, pupil, field = _objects()
    figure, axes = plt.subplots()
    canvas = Canvas(axes)

    weighted = canvas.plot_point_sampling_weighted(pupil)
    families = canvas.plot_bloch_families(plan)
    field_artist = canvas.plot_field(field)
    spectrum = canvas.plot_spectrum(space)
    target_spectrum = canvas.plot_spectrum(plan)

    assert len(weighted.get_offsets()) == len(pupil.points)
    assert len(families) == len(plan.representatives.points)
    assert len(field_artist.get_offsets()) == len(field.values)
    assert len(spectrum) == 1
    assert len(target_spectrum) == 1
    plt.close(figure)


def test_canvas_plots_interpolation_result_with_current_scipy_api():
    _space, _direct, _reciprocal, _plan, pupil, field = _objects()
    result = interpolate(field, pupil.points, method="nearest", extrapolation="nearest")
    figure, axes = plt.subplots()
    canvas = Canvas(axes)

    artist = canvas.plot_interpolation(result)

    assert artist.axes is axes
    plt.close(figure)


def test_extended_spectrum_draws_propagating_and_outer_boundaries():
    space = KSpace.extended(2.0 * np.pi, 1.0, 2.0)
    figure, axes = plt.subplots()

    artists = Canvas(axes).plot_spectrum(space)

    assert len(artists) == 2
    plt.close(figure)
