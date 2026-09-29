"""NumPy-only polygon geometry."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reciprocal.numerics import DEFAULT_TOLERANCES

from .model import FloatArray, PolygonDomain

BoolArray = NDArray[np.bool_]


def _cross2(first: FloatArray, second: FloatArray) -> FloatArray:
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def polygon_area(vertices: ArrayLike) -> float:
    array = np.asarray(vertices, dtype=float)
    if array.ndim != 2 or array.shape[0] < 3 or array.shape[1] not in (2, 3):
        raise ValueError("vertices must have shape (N, 2) or (N, 3), N >= 3")
    relative = array[:, :2] - array[0, :2]
    x, y = relative[:, 0], relative[:, 1]
    return float(abs(0.5 * (x @ np.roll(y, -1) - y @ np.roll(x, -1))))


def _points(points: ArrayLike) -> FloatArray:
    result = np.asarray(points, dtype=float)
    if result.ndim == 1:
        result = result[None, :]
    if result.ndim != 2 or result.shape[1] not in (2, 3):
        raise ValueError("points must have shape (N, 2) or (N, 3)")
    if not np.all(np.isfinite(result)):
        raise ValueError("points must be finite")
    return result


def distance_to_boundary(domain: PolygonDomain, points: ArrayLike) -> FloatArray:
    query = _points(points)[:, :2]
    start = domain.vertices[:, :2]
    end = np.roll(start, -1, axis=0)
    edge = end - start
    relative = query[:, None, :] - start[None, :, :]
    fraction = np.sum(relative * edge[None, :, :], axis=2) / np.sum(edge * edge, axis=1)
    fraction = np.clip(fraction, 0.0, 1.0)
    closest = start[None, :, :] + fraction[:, :, None] * edge[None, :, :]
    return np.min(np.linalg.norm(query[:, None, :] - closest, axis=2), axis=1)


def lies_on_boundary(
    domain: PolygonDomain,
    points: ArrayLike,
    tolerance: float = DEFAULT_TOLERANCES.boundary,
) -> BoolArray:
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    scale = max(domain.max_extent, np.ptp(domain.vertices[:, 0]), np.ptp(domain.vertices[:, 1]))
    return distance_to_boundary(domain, points) <= tolerance * max(scale, np.finfo(float).tiny)


def contains_points(
    domain: PolygonDomain, points: ArrayLike, *, include_boundary: bool = True
) -> BoolArray:
    query = _points(points)[:, :2]
    x, y = query[:, 0], query[:, 1]
    first = domain.vertices[:, :2]
    second = np.roll(first, -1, axis=0)
    y1, y2 = first[:, 1], second[:, 1]
    crosses = (y1[None, :] > y[:, None]) != (y2[None, :] > y[:, None])
    denominator = y2 - y1
    safe_denominator = np.where(denominator == 0.0, 1.0, denominator)
    intersections = (
        first[:, 0] + (y[:, None] - y1) * (second[:, 0] - first[:, 0]) / safe_denominator
    )
    inside = np.logical_xor.reduce(crosses & (x[:, None] < intersections), axis=1)
    boundary = lies_on_boundary(domain, query)
    return (inside | boundary) if include_boundary else (inside & ~boundary)


def crop_points(
    domain: PolygonDomain, points: ArrayLike, *, return_mask: bool = False
) -> FloatArray | tuple[FloatArray, BoolArray]:
    array = _points(points)
    mask = contains_points(domain, array)
    cropped = np.array(array[mask], copy=True)
    return (cropped, mask) if return_mask else cropped


def _clip_vertices(subject: FloatArray, clip: FloatArray) -> FloatArray:
    output = np.asarray(subject[:, :2], dtype=float)
    coordinate_scale = max(
        float(np.ptp(clip[:, 0])),
        float(np.ptp(clip[:, 1])),
        np.finfo(float).tiny,
    )
    side_tolerance = np.finfo(float).eps * coordinate_scale * coordinate_scale * 128
    for index, edge_start in enumerate(clip[:, :2]):
        edge_end = clip[(index + 1) % len(clip), :2]
        edge = edge_end - edge_start

        def side(point: FloatArray) -> float:
            return float(_cross2(edge, point - edge_start))

        input_vertices = output
        output_list: list[FloatArray] = []
        if len(input_vertices) == 0:
            break
        start = input_vertices[-1]
        start_inside = side(start) >= -side_tolerance
        for end in input_vertices:
            end_inside = side(end) >= -side_tolerance
            if end_inside != start_inside:
                direction = end - start
                denominator = _cross2(edge, direction)
                if denominator != 0.0:
                    fraction = _cross2(edge, edge_start - start) / denominator
                    output_list.append(start + fraction * direction)
            if end_inside:
                output_list.append(end)
            start, start_inside = end, end_inside
        output = np.asarray(output_list, dtype=float)
    if len(output) < 3:
        raise ValueError("polygons do not have a positive-area intersection")
    consecutive = np.linalg.norm(output - np.roll(output, 1, axis=0), axis=1)
    output = output[
        consecutive
        > np.finfo(float).eps * max(float(np.max(np.abs(output))), np.finfo(float).tiny) * 64
    ]
    return np.column_stack((output, np.zeros(len(output))))


def _is_convex(domain: PolygonDomain) -> bool:
    vertices = domain.vertices[:, :2]
    edge1 = np.roll(vertices, -1, axis=0) - vertices
    edge2 = np.roll(vertices, -2, axis=0) - np.roll(vertices, -1, axis=0)
    scale = max(
        float(np.ptp(vertices[:, 0])),
        float(np.ptp(vertices[:, 1])),
        np.finfo(float).tiny,
    )
    return bool(np.all(_cross2(edge1, edge2) >= -np.finfo(float).eps * scale * scale * 128))


def clip_convex_polygon(subject: PolygonDomain, clip_domain: PolygonDomain) -> PolygonDomain:
    if not _is_convex(subject) or not _is_convex(clip_domain):
        raise ValueError("clipping requires convex polygons")
    return PolygonDomain(_clip_vertices(subject.vertices, clip_domain.vertices))


def intersect_convex_polygons(first: PolygonDomain, second: PolygonDomain) -> PolygonDomain:
    return clip_convex_polygon(first, second)
