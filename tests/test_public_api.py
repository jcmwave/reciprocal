import warnings
from importlib import reload

import matplotlib.pyplot as plt
import numpy as np

import reciprocal
from reciprocal.canvas import Canvas


def test_documented_public_api_is_stable():
    expected = {
        "BlochFamily", "BravaisLattice", "KSpace", "KVector",
        "KVectorGroup", "Lattice", "LatticeVectors", "PointSymmetry",
        "SpecialPoint", "Symmetry", "SymmetryCombination", "SymmetryFamily",
        "UnitCell", "__version__",
    }
    assert set(reciprocal.__all__) == expected
    for name in expected:
        assert hasattr(reciprocal, name)


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
