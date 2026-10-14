"""Non-periodic samplers for bounded transverse-spectrum domains."""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay, QhullError

from reciprocal.cells.model import PolygonDomain
from reciprocal.integration_cells import (
    CartesianIntegrationCell,
    CircularIntegrationCell,
    IntegrationCell,
    PolarIntegrationCell,
)
from reciprocal.sampling import MaxSpacing, normalize_constraint, resolve_point_counts

from .domains import (
    EvanescentDisk,
    KDomain,
    PropagationDisk,
    PupilDomain,
    intersection_area,
)
from .model import (
    CartesianGrid,
    CircularGrid,
    KSampling,
    PointCounts,
    PolarGrid,
    SamplingConstraint,
)


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
        grid: CartesianGrid | CircularGrid | PolarGrid | PointCounts | str = CartesianGrid((16, 16)),
        *,
        include_cells: bool = False,
        constraint: SamplingConstraint = None,
    ) -> KSampling:
        if isinstance(grid, str):
            grid = resolve_sampling_grid(domain, grid, constraint)
        elif constraint is not None:
            raise ValueError("constraint cannot accompany an explicitly resolved grid")
        if isinstance(grid, PointCounts):
            grid = CartesianGrid((grid.first, grid.second))
        if isinstance(grid, CartesianGrid):
            return self.sample_cartesian(domain, grid, include_cells=include_cells)
        if isinstance(grid, CircularGrid):
            return self.sample_circular(domain, grid, include_cells=include_cells)
        if isinstance(grid, PolarGrid):
            return self.sample_polar(domain, grid, include_cells=include_cells)
        raise TypeError("grid must be CartesianGrid, CircularGrid, PolarGrid, or PointCounts")

    def sample_cartesian(
        self, domain: KDomain, grid: CartesianGrid, *, include_cells: bool = False
    ) -> KSampling:
        """Mask a fixed Cartesian grid to the domain and give its points equal weights.

        Points stay at the centers of equally spaced bounding-box rectangles.
        Retained cell footprints are full rectangles, including at boundaries;
        each cell's effective quadrature area is ``domain.area / N``.
        """
        x_min, x_max, y_min, y_max = domain.bounds()
        x_edges = np.linspace(x_min, x_max, grid.shape[0] + 1)
        y_edges = np.linspace(y_min, y_max, grid.shape[1] + 1)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        mesh = np.meshgrid(x_centers, y_centers, indexing="ij")
        candidates = np.column_stack((mesh[0].ravel(), mesh[1].ravel()))
        keep = domain.contains(candidates)
        points = candidates[keep]
        if len(points) == 0:
            raise ValueError("grid produced no points in the domain")
        weights = np.full(len(points), 1.0 / len(points))
        cells = None
        if include_cells:
            cells = tuple(
                CartesianIntegrationCell(
                    (
                        float(x_edges[first]),
                        float(x_edges[first + 1]),
                        float(y_edges[second]),
                        float(y_edges[second + 1]),
                    ),
                    domain,
                    domain.area / len(points),
                )
                for first, second in zip(*np.nonzero(keep.reshape(grid.shape)))
            )
        return KSampling(
            points,
            weights,
            domain.area,
            domain,
            integration_cells=cells,
        )

    def sample_circular(
        self,
        domain: KDomain,
        grid: CircularGrid,
        *,
        include_cells: bool = False,
        angular_multiple: int = 1,
        angular_offset: float = 0.0,
    ) -> KSampling:
        """Sample the origin and rings with comparable radial and arc spacing.

        Requires an origin-centered disk so the circular center cell and
        annular wedges partition the integration domain exactly.
        """
        if isinstance(domain, PropagationDisk):
            outer = domain.radius
        elif isinstance(domain, EvanescentDisk):
            outer = domain.max_parallel_wavevector
        elif isinstance(domain, PupilDomain) and domain.inner_radius == 0.0:
            outer = domain.outer_radius
        else:
            raise ValueError("CircularGrid requires an origin-centered disk domain")
        if domain.center != (0.0, 0.0):
            raise ValueError("CircularGrid requires an origin-centered disk domain")

        spacing = outer / (grid.radial - 0.5)
        center_radius = spacing * 0.5
        points = [np.zeros(2)]
        areas = [np.pi * center_radius**2]
        cells: list[IntegrationCell] = []
        if include_cells:
            cells.append(CircularIntegrationCell(center_radius))
        for ring in range(1, grid.radial):
            radius = ring * spacing
            if grid.max_azimuthal_spacing is not None:
                count = max(
                    angular_multiple,
                    int(np.ceil(2 * np.pi * radius / grid.max_azimuthal_spacing / angular_multiple))
                    * angular_multiple,
                )
            else:
                nominal = (
                    (grid.azimuthal * ring / (grid.radial - 1))
                    if grid.azimuthal is not None
                    else 2 * np.pi * ring
                )
                count = max(
                    angular_multiple,
                    int(np.floor(nominal / angular_multiple + 0.5)) * angular_multiple,
                )
            angle_spacing = 2.0 * np.pi / count
            angles = np.arange(count) * angle_spacing + angular_offset
            inner = (ring - 0.5) * spacing
            upper = min((ring + 0.5) * spacing, outer)
            points.extend(np.column_stack((radius * np.cos(angles), radius * np.sin(angles))))
            area = 0.5 * (upper**2 - inner**2) * angle_spacing
            areas.extend([area] * count)
            if include_cells:
                cells.extend(
                    PolarIntegrationCell(
                        (0.0, 0.0),
                        inner,
                        upper,
                        angle - angle_spacing * 0.5,
                        angle + angle_spacing * 0.5,
                    )
                    for angle in angles
                )
        raw = np.asarray(areas)
        return KSampling(
            np.vstack(points),
            raw / np.sum(raw),
            domain.area,
            domain,
            integration_cells=tuple(cells) if include_cells else None,
        )

    def sample_polar(
        self,
        domain: KDomain,
        grid: PolarGrid,
        *,
        include_cells: bool = False,
        angular_offset: float = 0.0,
    ) -> KSampling:
        if isinstance(domain, PropagationDisk):
            inner, outer, center = 0.0, domain.radius, domain.center
        elif isinstance(domain, EvanescentDisk):
            inner, outer, center = 0.0, domain.max_parallel_wavevector, domain.center
        elif isinstance(domain, PupilDomain):
            inner, outer, center = domain.inner_radius, domain.outer_radius, domain.center
        else:
            # Polar cells are only exact for radial domains. Preserve a useful
            # deterministic fallback for arbitrary domain implementations.
            return self.sample_cartesian(
                domain,
                CartesianGrid((grid.radial * 2, grid.radial * 2)),
                include_cells=include_cells,
            )

        radial_squared = np.linspace(inner**2, outer**2, grid.radial + 1)
        angular_edges = np.linspace(0.0, 2.0 * np.pi, grid.azimuthal + 1) + angular_offset
        points = []
        areas = []
        cells = []
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
                if include_cells:
                    cells.append(
                        PolarIntegrationCell(
                            center, np.sqrt(lower_squared), np.sqrt(upper_squared), lower, upper
                        )
                    )
        raw = np.asarray(areas)
        return KSampling(
            np.vstack(points),
            raw / np.sum(raw),
            domain.area,
            domain,
            integration_cells=tuple(cells) if include_cells else None,
        )


def grid_from_constraint(
    domain: KDomain, constraint: SamplingConstraint, *, default: int = 16
) -> CartesianGrid:
    """Convert a typed or legacy resolution constraint to a Cartesian grid."""

    if isinstance(constraint, CartesianGrid):
        return constraint
    return resolve_sampling_grid(
        domain, "cartesian", normalize_constraint(constraint, default=default)
    )


def resolve_sampling_grid(domain: KDomain, scheme: str, constraint: SamplingConstraint = None):
    """Construct a spectrum grid from common physical spacing/count constraints."""
    if scheme not in ("cartesian", "circular", "polar"):
        raise ValueError("scheme must be cartesian, circular, or polar")
    typed = normalize_constraint(constraint, default=5 if scheme == "circular" else 16)
    x0, x1, y0, y1 = domain.bounds()
    if scheme == "cartesian":
        return CartesianGrid(resolve_point_counts((x1 - x0, y1 - y0), typed))
    if isinstance(domain, PropagationDisk):
        inner, outer = 0.0, domain.radius
    elif isinstance(domain, EvanescentDisk):
        inner, outer = 0.0, domain.max_parallel_wavevector
    elif isinstance(domain, PupilDomain):
        inner, outer = domain.inner_radius, domain.outer_radius
    else:
        raise ValueError("radial constraints require a disk or annular domain")
    values = typed.values
    if scheme == "circular":
        if inner != 0:
            raise ValueError("CircularGrid requires a complete disk")
        if isinstance(typed, PointCounts):
            return CircularGrid(typed.first, None if typed.dimensions == 1 else typed.second)
        radial_max = values[0]
        radial = max(1, int(np.ceil(outer / radial_max + 0.5)))
        angular_max = values[-1]
        return CircularGrid(radial, max_azimuthal_spacing=angular_max)
    if scheme == "polar":
        if isinstance(typed, PointCounts):
            return PolarGrid(typed.first, typed.second)
        # Equal-area radial bands: their largest width occurs at the inner edge.
        radial = max(
            1, int(np.ceil((outer**2 - inner**2) / (values[0] ** 2 + 2 * inner * values[0])))
        )
        angular = resolve_point_counts(
            2 * np.pi * outer, MaxSpacing(values[-1]), placement="closed"
        )[0]
        return PolarGrid(radial, angular)
    raise ValueError("scheme must be cartesian, circular, or polar")


__all__ = [
    "NonPeriodicSampler",
    "grid_from_constraint",
    "resolve_sampling_grid",
    "voronoi_physical_weights",
]
