import numpy as np
import pytest

from reciprocal.cells.geometry import (
    contains_points,
    crop_points,
    distance_to_boundary,
    intersect_convex_polygons,
    lies_on_boundary,
)
from reciprocal.cells.model import PolygonDomain


def test_polygon_domain_is_immutable_and_normalizes_orientation():
    source = np.array([[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 0.0]])
    domain = PolygonDomain(source)
    source[:] = -1.0

    assert domain.area == pytest.approx(4.0)
    assert not domain.vertices.flags.writeable
    with pytest.raises(ValueError):
        domain.vertices[0, 0] = 1.0


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_geometry_operations_are_scale_aware(scale):
    first = PolygonDomain(scale * np.array([[0, 0], [2, 0], [2, 2], [0, 2]]))
    second = PolygonDomain(scale * np.array([[1, 1], [3, 1], [3, 3], [1, 3]]))
    points = scale * np.array([[1, 1], [3, 1], [0, 1]])

    np.testing.assert_array_equal(contains_points(first, points), [True, False, True])
    np.testing.assert_array_equal(lies_on_boundary(first, points), [False, False, True])
    np.testing.assert_allclose(distance_to_boundary(first, points[:1]), [scale])
    cropped, mask = crop_points(first, points, return_mask=True)
    np.testing.assert_array_equal(cropped, points[mask])
    assert intersect_convex_polygons(first, second).area == pytest.approx(scale**2)
