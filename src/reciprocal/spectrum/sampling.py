"""Non-periodic samplers for bounded transverse-spectrum domains."""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay, QhullError

from reciprocal.cells.model import PolygonDomain

from .domains import (
    EvanescentDisk,
    KDomain,
    PropagationDisk,
    PupilDomain,
    intersection_area,
)
from .model import CartesianGrid, KSampling, PointCounts, PolarGrid, SamplingConstraint


def _cell_polygon(x0: float, x1: float, y0: float, y1: float) -> PolygonDomain:
    return PolygonDomain(
        np.array(
            [
                [x0, y0, 0.0],
                [x1, y0, 0.0],
                [x1, y1, 0.0],
                [x0, y1, 0.0],
            ]
        )
    )


def _representative(domain: KDomain, bounds: tuple[float, float, float, float]) -> np.ndarray:
    x0, x1, y0, y1 = bounds
    center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5])
    if domain.contains(center)[0]:
        return center
    # Boundary cells need a point inside both the cell and domain.  A small,
    # deterministic tensor search is sufficient because cells with zero-area
    # intersections have already been removed.
    coordinates = np.linspace(0.05, 0.95, 19)
    candidates = np.array(
        [
            [x0 + tx * (x1 - x0), y0 + ty * (y1 - y0)]
            for tx in coordinates
            for ty in coordinates
        ]
    )
    inside = candidates[domain.contains(candidates)]
    if len(inside):
        return inside[np.argmin(np.linalg.norm(inside - center, axis=1))]
    raise RuntimeError("could not place a representative in a positive-area boundary cell")


def _clip_half_plane(
    vertices: np.ndarray, normal: np.ndarray, offset: float, tolerance: float
) -> np.ndarray:
    output: list[np.ndarray] = []
    if len(vertices) == 0:
        return vertices
    for index, start in enumerate(vertices):
        end = vertices[(index + 1) % len(vertices)]
        first = float(np.dot(start, normal) - offset)
        second = float(np.dot(end, normal) - offset)
        first_inside = first <= tolerance
        second_inside = second <= tolerance
        if first_inside != second_inside:
            fraction = first / (first - second)
            output.append(start + fraction * (end - start))
        if second_inside:
            output.append(end)
    array = np.asarray(output, dtype=float)
    if len(array) < 2:
        return array
    scale = max(float(np.max(np.abs(array))), np.finfo(float).tiny)
    threshold = np.finfo(float).eps * scale * 128
    cleaned = [array[0]]
    for point in array[1:]:
        if np.linalg.norm(point - cleaned[-1]) > threshold:
            cleaned.append(point)
    if len(cleaned) > 1 and np.linalg.norm(cleaned[0] - cleaned[-1]) <= threshold:
        cleaned.pop()
    return np.asarray(cleaned, dtype=float)


def voronoi_physical_weights(points: np.ndarray, domain: KDomain) -> np.ndarray:
    """Clip nearest-point Voronoi cells to a bounded integration domain."""

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("points must have shape (N, 2) with N > 0")
    if len(points) == 1:
        return np.array([domain.area])
    neighbors: list[set[int]] = [set() for _ in range(len(points))]
    try:
        triangulation = Delaunay(points)
        indptr, indices = triangulation.vertex_neighbor_vertices
        for index in range(len(points)):
            neighbors[index].update(
                int(item) for item in indices[indptr[index] : indptr[index + 1]]
            )
    except QhullError:
        for index in range(len(points)):
            neighbors[index].update(item for item in range(len(points)) if item != index)

    x0, x1, y0, y1 = domain.bounds()
    rectangle = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
    scale = max(x1 - x0, y1 - y0, np.finfo(float).tiny)
    tolerance = np.finfo(float).eps * scale * scale * 256
    weights = np.zeros(len(points))
    for index, point in enumerate(points):
        vertices = rectangle.copy()
        for neighbor_index in neighbors[index]:
            neighbor = points[neighbor_index]
            normal = neighbor - point
            offset = 0.5 * (float(np.dot(neighbor, neighbor)) - float(np.dot(point, point)))
            vertices = _clip_half_plane(vertices, normal, offset, tolerance)
            if len(vertices) < 3:
                break
        if len(vertices) >= 3:
            weights[index] = intersection_area(domain, PolygonDomain(vertices))
    total = float(np.sum(weights))
    if total <= 0.0:
        raise RuntimeError("Voronoi quadrature produced zero area")
    # Convex Voronoi cells partition the domain. Correct only accumulated
    # floating-point clipping error while retaining boundary-cell ratios.
    weights *= domain.area / total
    return weights


class NonPeriodicSampler:
    """Generate weighted samples without imposing lattice periodicity."""

    def sample(
        self,
        domain: KDomain,
        grid: CartesianGrid | PolarGrid | PointCounts = CartesianGrid((16, 16)),
    ) -> KSampling:
        if isinstance(grid, PointCounts):
            grid = CartesianGrid((grid.first, grid.second))
        if isinstance(grid, CartesianGrid):
            return self.sample_cartesian(domain, grid)
        if isinstance(grid, PolarGrid):
            return self.sample_polar(domain, grid)
        raise TypeError("grid must be CartesianGrid, PolarGrid, or PointCounts")

    def sample_cartesian(self, domain: KDomain, grid: CartesianGrid) -> KSampling:
        x_min, x_max, y_min, y_max = domain.bounds()
        x_edges = np.linspace(x_min, x_max, grid.shape[0] + 1)
        y_edges = np.linspace(y_min, y_max, grid.shape[1] + 1)
        points: list[np.ndarray] = []
        areas: list[float] = []
        area_threshold = np.finfo(float).eps * domain.area * 256
        for first in range(grid.shape[0]):
            for second in range(grid.shape[1]):
                bounds = (
                    float(x_edges[first]),
                    float(x_edges[first + 1]),
                    float(y_edges[second]),
                    float(y_edges[second + 1]),
                )
                cell = _cell_polygon(*bounds)
                area = intersection_area(domain, cell)
                if area <= area_threshold:
                    continue
                points.append(_representative(domain, bounds))
                areas.append(area)
        if not points:
            raise ValueError("grid produced no points in the domain")
        raw = np.asarray(areas, dtype=float)
        # The cells tile the bounding box. Renormalization only removes
        # floating-point accumulation error in analytic clipping.
        weights = raw / np.sum(raw)
        return KSampling(np.vstack(points), weights, domain.area, domain)

    def sample_polar(self, domain: KDomain, grid: PolarGrid) -> KSampling:
        if isinstance(domain, PropagationDisk):
            inner, outer, center = 0.0, domain.radius, domain.center
        elif isinstance(domain, EvanescentDisk):
            inner, outer, center = 0.0, domain.max_parallel_wavevector, domain.center
        elif isinstance(domain, PupilDomain):
            inner, outer, center = domain.inner_radius, domain.outer_radius, domain.center
        else:
            # Polar cells are only exact for radial domains. Preserve a useful
            # deterministic fallback for arbitrary domain implementations.
            return self.sample_cartesian(domain, CartesianGrid((grid.radial * 2, grid.radial * 2)))

        radial_squared = np.linspace(inner**2, outer**2, grid.radial + 1)
        angular_edges = np.linspace(0.0, 2.0 * np.pi, grid.azimuthal + 1)
        points = []
        areas = []
        center_array = np.asarray(center)
        for radial_index in range(grid.radial):
            lower_squared = radial_squared[radial_index]
            upper_squared = radial_squared[radial_index + 1]
            radius = np.sqrt(0.5 * (lower_squared + upper_squared))
            for angular_index in range(grid.azimuthal):
                lower = angular_edges[angular_index]
                upper = angular_edges[angular_index + 1]
                angle = 0.5 * (lower + upper)
                points.append(center_array + radius * np.array([np.cos(angle), np.sin(angle)]))
                areas.append(0.5 * (upper_squared - lower_squared) * (upper - lower))
        raw = np.asarray(areas)
        return KSampling(np.vstack(points), raw / np.sum(raw), domain.area, domain)


def grid_from_constraint(
    domain: KDomain, constraint: SamplingConstraint, *, default: int = 16
) -> CartesianGrid:
    """Convert a typed or legacy resolution constraint to a Cartesian grid."""

    if constraint is None:
        return CartesianGrid((default, default))
    if isinstance(constraint, PointCounts):
        return CartesianGrid((constraint.first, constraint.second))
    if isinstance(constraint, CartesianGrid):
        return constraint
    if isinstance(constraint, dict):
        kind = constraint.get("type")
        value = constraint.get("value")
        if kind == "n_points":
            values = np.asarray(value)
            if values.ndim == 0:
                return CartesianGrid((int(values), int(values)))
            if values.shape == (2,):
                return CartesianGrid((int(values[0]), int(values[1])))
        if kind == "max_length":
            spacing = float(value)
            if not np.isfinite(spacing) or spacing <= 0.0:
                raise ValueError("max_length must be finite and positive")
            bounds = domain.bounds()
            return CartesianGrid(
                (
                    max(1, int(np.ceil((bounds[1] - bounds[0]) / spacing))),
                    max(1, int(np.ceil((bounds[3] - bounds[2]) / spacing))),
                )
            )
    from .model import MaxSpacing

    if isinstance(constraint, MaxSpacing):
        return grid_from_constraint(domain, constraint.as_legacy(), default=default)
    raise ValueError("unsupported sampling constraint")


__all__ = ["NonPeriodicSampler", "grid_from_constraint", "voronoi_physical_weights"]
