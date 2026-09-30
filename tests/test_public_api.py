import warnings
from importlib import reload

import matplotlib.pyplot as plt
import numpy as np
import pytest

import reciprocal
from reciprocal.bravais import BravaisLattice
from reciprocal.canvas import Canvas


def test_documented_public_api_is_stable():
    expected = {
        "BlochFamily",
        "BoundaryGrid",
        "BravaisLattice",
        "BrillouinZone",
        "CanonicalPlacementError",
        "DEFAULT_TOLERANCES",
        "GridCentering",
        "GridSymmetry",
        "HighSymmetryPath",
        "IncompatibleGridSymmetryError",
        "KSpace",
        "KVector",
        "KVectorGroup",
        "Lattice",
        "LatticeVectors",
        "MaxSpacing",
        "MeshReduction",
        "MonkhorstPackGrid",
        "PointCounts",
        "PointSymmetry",
        "RepresentativeMap",
        "RepresentativePlacement",
        "SpecialPoint",
        "SamplingDomain",
        "SamplingResult",
        "Symmetry",
        "SymmetryCombination",
        "Tolerances",
        "UnitCell",
        "ZoneGrid",
        "ZoneReduction",
        "ZoneRegion",
        "ZoneSamplingError",
        "__version__",
        "make_high_symmetry_path",
        "grid_symmetry",
        "reduce_brillouin_zone",
        "reduce_monkhorst_pack",
        "sample_monkhorst_pack",
        "sample_brillouin_zone",
        "set_band_path_axis",
    }
    assert set(reciprocal.__all__) == expected
    for name in expected:
        assert hasattr(reciprocal, name)
    assert reciprocal.BravaisLattice is BravaisLattice


def test_import_does_not_install_global_warning_filters():
    before = list(warnings.filters)
    reload(reciprocal)
    assert warnings.filters == before


def test_canvas_uses_the_supplied_axes_without_changing_current_axes():
    figure, (first, second) = plt.subplots(1, 2)
    plt.sca(first)
    canvas = Canvas(second)
    handle = canvas.plot_point_sampling(np.array([[0.0, 0.0]]))
    assert canvas.fig is figure
    assert handle.axes is second
    assert plt.gca() is first
    plt.close(figure)


def test_misspelled_plotting_name_has_deprecation_path(monkeypatch):
    figure, axes = plt.subplots()
    canvas = Canvas(axes)
    called = []
    monkeypatch.setattr(canvas, "plot_tessellation", lambda *args, **kwargs: called.append(True))
    with pytest.warns(DeprecationWarning, match="plot_tessellation"):
        canvas.plot_tesselation(object())
    assert called == [True]
    plt.close(figure)
