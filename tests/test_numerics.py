import numpy as np
import pytest

from reciprocal.numerics import Tolerances, contains_close_point


def test_vectorized_duplicate_detection_matches_scalar_reference():
    rng = np.random.default_rng(1729)
    points = rng.normal(size=(500, 3))
    candidates = [points[123], np.array([20.0, 20.0, 20.0])]
    for candidate in candidates:
        expected = any(
            np.isclose(candidate, sampled, rtol=1e-9, atol=0.0).all() for sampled in points
        )
        assert contains_close_point(points, candidate) is expected


def test_duplicate_absolute_tolerance_is_explicit_and_overrideable():
    points = np.array([[1e-12, 0.0]])
    candidate = np.array([2e-12, 0.0])
    assert not contains_close_point(points, candidate)
    assert contains_close_point(points, candidate, absolute_tolerance=1e-12)


def test_invalid_tolerance_policy_is_rejected():
    with pytest.raises(ValueError, match="finite and non-negative"):
        Tolerances(relative=-1.0)
