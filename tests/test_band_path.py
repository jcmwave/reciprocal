import matplotlib.pyplot as plt
import numpy as np
import pytest

from reciprocal import Lattice
from reciprocal.band_path import make_high_symmetry_path, set_band_path_axis
from reciprocal.canvas import Canvas


@pytest.mark.parametrize(
    ("length1", "length2", "angle", "labels"),
    [
        (1.0, 1.0, 90.0, ("Γ", "X", "M", "Γ")),
        (1.5, 1.0, 90.0, ("Γ", "X", "S", "Y", "Γ")),
        (1.0, 1.0, 70.0, ("Γ", "X", "H1", "S", "Γ")),
        (1.0, 1.0, 60.0, ("Γ", "M", "K", "Γ")),
        (1.3, 1.0, 73.0, ("Γ", "X", "C", "Y", "Γ")),
    ],
)
def test_default_paths_for_all_bravais_families(length1, length2, angle, labels):
    zone = Lattice.from_lengths_angle(length1, length2, angle).make_reciprocal().brillouin_zone
    path = make_high_symmetry_path(zone, points_per_segment=5)
    assert path.labels == labels
    assert np.all(np.diff(path.distance) >= 0.0)
    np.testing.assert_allclose(
        path.fractional_points @ path.reciprocal_basis[:, :2],
        path.points,
        atol=1e-12,
    )
    for label, index in zip(path.labels, path.node_indices):
        np.testing.assert_allclose(path.points[index], zone.special_points[label].cartesian[:2])


def test_spacing_custom_branches_and_axis_helpers():
    zone = Lattice.from_lengths_angle(1.0, 1.0, 90.0).make_reciprocal().brillouin_zone
    path = make_high_symmetry_path(
        zone,
        (("Γ", "X"), ("M", "Γ")),
        max_spacing=0.3,
    )
    increments = np.diff(path.distance)
    assert np.all(increments <= 0.3 + 1e-12)
    assert len(path.break_indices) == 1
    assert increments[path.break_indices[0] - 1] == pytest.approx(0.0)

    figure, axes = plt.subplots()
    Canvas(axes).plot_high_symmetry_path(path)
    set_band_path_axis(axes, path)
    assert len(axes.get_xticks()) == len(path.labels)
    plt.close(figure)


def test_centered_rectangular_default_path_stays_in_the_canonical_ibz():
    zone = Lattice.from_lengths_angle(500.0, 500.0, 70.0).make_reciprocal().brillouin_zone
    path = make_high_symmetry_path(zone, points_per_segment=5)

    assert path.labels == ("Γ", "X", "H1", "S", "Γ")
    assert np.all(zone.irreducible_domain.contains(path.points))


def test_path_resolution_validation():
    zone = Lattice.from_lengths_angle(1.0, 1.0, 90.0).make_reciprocal().brillouin_zone
    with pytest.raises(ValueError, match="either"):
        make_high_symmetry_path(zone, points_per_segment=3, max_spacing=0.1)
    with pytest.raises(ValueError, match="unknown"):
        make_high_symmetry_path(zone, ("Γ", "missing"))
