import numpy as np
import pytest

from reciprocal.symmetry import Symmetry
from reciprocal.utils import reflection2D, rotation3D


@pytest.mark.parametrize("name, expected_size", [("C4", 4), ("D4", 8)])
def test_symmetry_preserves_norm_and_has_expected_degeneracy(name, expected_size):
    symmetry = Symmetry.from_string(name)
    point = np.array([1.25, -0.5, 0.0])

    transformed = symmetry.apply_symmetry_operators(point)

    assert transformed.shape == (expected_size, 3)
    np.testing.assert_allclose(
        np.linalg.norm(transformed, axis=1),
        np.linalg.norm(point),
        rtol=1e-12,
        atol=1e-12,
    )


def test_unknown_symmetry_and_axes_raise_useful_errors():
    with pytest.raises(ValueError, match="unknown"):
        Symmetry.from_string("C5")
    with pytest.raises(ValueError, match="axis"):
        rotation3D(90.0, "Q")
    with pytest.raises(ValueError, match="axis"):
        reflection2D("diagonal-ish")
