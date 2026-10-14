"""Sampling services for immutable unit cells."""

from __future__ import annotations

import numpy as np

from reciprocal.sampling import SamplingConstraint, resolve_point_counts

from .geometry import contains_points
from .model import SamplingResult, UnitCell


def _grid_shape(cell: UnitCell, constraint: SamplingConstraint | None) -> tuple[int, int]:
    fractional = cell.vertices[:, :2] @ np.linalg.inv(cell.basis[:, :2])
    extents = np.ptp(fractional, axis=0) * np.linalg.norm(cell.basis[:, :2], axis=1)
    first, second = resolve_point_counts(extents, constraint, placement="endpoints", default=5)
    return first, second


class CellSampler:
    """Generate deterministic cell samples with normalized quadrature weights."""

    def sample(
        self,
        cell: UnitCell,
        constraint: SamplingConstraint | None = None,
        center: np.ndarray | None = None,
    ) -> SamplingResult:
        first_count, second_count = _grid_shape(cell, constraint)
        center_array = np.zeros(2) if center is None else np.asarray(center, dtype=float)
        if center_array.shape not in ((2,), (3,)) or not np.all(np.isfinite(center_array)):
            raise ValueError("center must be a finite planar point")
        if center_array.shape == (3,) and center_array[2] != 0.0:
            raise ValueError("center must lie in the x-y plane")
        center_array = center_array[:2]
        inverse_basis = np.linalg.inv(cell.basis[:, :2])
        domain_fractional = cell.vertices[:, :2] @ inverse_basis
        first_bounds = (np.min(domain_fractional[:, 0]), np.max(domain_fractional[:, 0]))
        second_bounds = (np.min(domain_fractional[:, 1]), np.max(domain_fractional[:, 1]))
        first_coordinates = (
            np.array([np.mean(first_bounds)])
            if first_count == 1
            else np.linspace(*first_bounds, first_count)
        )
        second_coordinates = (
            np.array([np.mean(second_bounds)])
            if second_count == 1
            else np.linspace(*second_bounds, second_count)
        )
        fractional = np.stack(
            np.meshgrid(first_coordinates, second_coordinates, indexing="ij"), axis=-1
        ).reshape(-1, 2)
        cartesian = fractional @ cell.basis[:, :2] + center_array
        cartesian = cartesian[contains_points(cell.domain, cartesian)]
        order = np.lexsort((cartesian[:, 1], cartesian[:, 0]))
        cartesian = cartesian[order]
        if len(cartesian) == 0:
            raise ValueError("sampling constraint produced no points in the cell")
        ownership = np.ones(len(cartesian))
        for index, point in enumerate(cartesian):
            copies = 0
            for first_shift in range(-1, 2):
                for second_shift in range(-1, 2):
                    shift = np.array([first_shift, second_shift]) @ cell.basis[:, :2]
                    copies += int(contains_points(cell.domain, [point + shift])[0])
            ownership[index] = 1.0 / copies
        weights = ownership / np.sum(ownership)
        return SamplingResult(cartesian, weights, cell.area, cell.domain)
