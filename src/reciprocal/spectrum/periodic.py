"""Periodic representative sampling and provenance-preserving expansion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from reciprocal.brillouin_zone import BrillouinZone, BrillouinZoneSampler
from reciprocal.lattice import Lattice
from reciprocal.numerics import DEFAULT_TOLERANCES, Tolerances
from reciprocal.symmetry import PointOperation

from .domains import EvanescentDisk, KDomain, PropagationDisk, brillouin_zone_domain
from .model import (
    FieldSamples,
    KSampling,
    SamplingConstraint,
    SourceRegion,
    TargetRegion,
    legacy_constraint,
)
from .sampling import voronoi_physical_weights


@dataclass(frozen=True, slots=True)
class ExpansionGenerator:
    source_index: int
    operation: PointOperation
    reciprocal_order: tuple[int, int]


@dataclass(frozen=True, slots=True, eq=False)
class ExpansionMap:
    """Map representative samples to unique physical targets."""

    source: KSampling
    target: KSampling
    primary_generators: tuple[ExpansionGenerator, ...]
    equivalent_generators: tuple[tuple[ExpansionGenerator, ...], ...]

    def __post_init__(self) -> None:
        if len(self.primary_generators) != len(self.target.points):
            raise ValueError("one primary generator is required for every target")
        if len(self.equivalent_generators) != len(self.target.points):
            raise ValueError("one generator equivalence class is required for every target")
        for primary, generators in zip(self.primary_generators, self.equivalent_generators):
            if not generators or primary not in generators:
                raise ValueError("each equivalence class must contain its primary generator")
            if any(
                generator.source_index < 0
                or generator.source_index >= len(self.source.points)
                for generator in generators
            ):
                raise ValueError("generator source index is outside the source sampling")

    def expand(self, data: FieldSamples[Any], *, verify: bool = True) -> FieldSamples[Any]:
        if not isinstance(data, FieldSamples):
            raise TypeError("data must be FieldSamples")
        if not np.array_equal(data.sampling.point_ids, self.source.point_ids) or not np.allclose(
            data.sampling.points, self.source.points, rtol=0.0, atol=0.0
        ):
            raise ValueError("field data sampling does not match the expansion source")
        values = []
        valid = []
        for target_index, (point, primary, generators) in enumerate(
            zip(self.target.points, self.primary_generators, self.equivalent_generators)
        ):
            value = data.representation.transform(
                data.values[primary.source_index],
                primary.operation,
                self.source.points[primary.source_index],
                point,
                primary.reciprocal_order,
            )
            is_valid = bool(data.valid[primary.source_index])
            if verify and is_valid:
                for alternative in generators:
                    if not data.valid[alternative.source_index]:
                        continue
                    alternative_value = data.representation.transform(
                        data.values[alternative.source_index],
                        alternative.operation,
                        self.source.points[alternative.source_index],
                        point,
                        alternative.reciprocal_order,
                    )
                    if not np.allclose(alternative_value, value, rtol=1e-8, atol=1e-11):
                        raise ValueError(
                            "equivalent symmetry generators produce inconsistent values "
                            f"at target index {target_index}"
                        )
            values.append(value)
            valid.append(is_valid)
        return FieldSamples(
            self.target,
            np.asarray(values),
            data.representation,
            np.asarray(valid),
            data.units,
        )


@dataclass(frozen=True, slots=True)
class PeriodicSamplingPlan:
    direct_lattice: Lattice
    reciprocal_lattice: Lattice
    zone: BrillouinZone
    source_region: SourceRegion
    target_region: TargetRegion
    representatives: KSampling
    expansion: ExpansionMap

    @property
    def target(self) -> KSampling:
        return self.expansion.target

    @property
    def speedup(self) -> float:
        return len(self.target.points) / len(self.representatives.points)

    def expand(self, data: FieldSamples[Any], *, verify: bool = True) -> FieldSamples[Any]:
        return self.expansion.expand(data, verify=verify)


def _identity(operations: tuple[PointOperation, ...]) -> PointOperation:
    for operation in operations:
        if np.array_equal(operation.fractional, np.eye(2, dtype=int)):
            return operation
    raise RuntimeError("point group does not contain the identity")


def _canonical_fractional(point: np.ndarray, basis: np.ndarray) -> np.ndarray:
    fractional = point @ np.linalg.inv(basis)
    return fractional - np.floor(fractional + 0.5)


def _canonicalize_sampling(sampling: KSampling, basis: np.ndarray) -> KSampling:
    keys: list[np.ndarray] = []
    points: list[np.ndarray] = []
    weights: list[float] = []
    tolerance = DEFAULT_TOLERANCES.relative
    for point, weight in zip(sampling.points, sampling.normalized_weights):
        canonical = _canonical_fractional(point, basis)
        match = next(
            (
                index
                for index, existing in enumerate(keys)
                if np.allclose(canonical, existing, rtol=0.0, atol=tolerance)
            ),
            None,
        )
        if match is None:
            keys.append(canonical)
            points.append(point)
            weights.append(float(weight))
        else:
            weights[match] += float(weight)
    result_weights = np.asarray(weights)
    result_weights /= np.sum(result_weights)
    metadata = dict(sampling.metadata)
    metadata["canonical_fractional"] = np.vstack(keys)
    return KSampling(
        np.vstack(points),
        result_weights,
        sampling.integration_element,
        sampling.domain,
        metadata=metadata,
    )


def _translation_orders(
    domain: KDomain,
    basis: np.ndarray,
    zone_extent: float,
    target_region: TargetRegion,
) -> np.ndarray:
    if target_region is TargetRegion.BZ:
        return np.zeros((1, 2), dtype=np.int64)
    x0, x1, y0, y1 = domain.bounds()
    corners = np.array(
        [
            [x0 - zone_extent, y0 - zone_extent],
            [x0 - zone_extent, y1 + zone_extent],
            [x1 + zone_extent, y0 - zone_extent],
            [x1 + zone_extent, y1 + zone_extent],
        ]
    )
    fractional = corners @ np.linalg.inv(basis)
    lower = np.floor(np.min(fractional, axis=0)).astype(int) - 1
    upper = np.ceil(np.max(fractional, axis=0)).astype(int) + 1
    first = np.arange(lower[0], upper[0] + 1, dtype=np.int64)
    second = np.arange(lower[1], upper[1] + 1, dtype=np.int64)
    mesh = np.meshgrid(first, second, indexing="ij")
    return np.column_stack((mesh[0].ravel(), mesh[1].ravel()))


def _generator_key(generator: ExpansionGenerator) -> tuple[int, int, int, tuple[int, ...]]:
    return (
        generator.source_index,
        generator.reciprocal_order[0],
        generator.reciprocal_order[1],
        tuple(int(value) for value in generator.operation.fractional.ravel()),
    )


def _group_candidates(
    points: np.ndarray,
    tolerance: float,
) -> list[list[int]]:
    if len(points) == 1:
        return [[0]]
    parent = np.arange(len(points))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(first: int, second: int) -> None:
        left, right = find(first), find(second)
        if left != right:
            parent[max(left, right)] = min(left, right)

    for first, second in cKDTree(points).query_pairs(tolerance):
        union(first, second)
    groups: dict[int, list[int]] = {}
    for index in range(len(points)):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def build_expansion_map(
    source: KSampling,
    reciprocal_lattice: Lattice,
    zone: BrillouinZone,
    target_domain: KDomain,
    source_region: SourceRegion,
    target_region: TargetRegion,
    *,
    tolerances: Tolerances = DEFAULT_TOLERANCES,
) -> ExpansionMap:
    """Construct a complete combined point-operation/translation map."""

    basis = reciprocal_lattice.vectors.basis[:, :2]
    identity = _identity(zone.point_group)
    operations = zone.point_group if source_region is SourceRegion.IBZ else (identity,)
    orders = _translation_orders(target_domain, basis, zone.max_extent, target_region)
    candidate_points: list[np.ndarray] = []
    candidate_generators: list[ExpansionGenerator] = []
    for source_index, point in enumerate(source.points):
        for operation in operations:
            rotated = operation.apply(point)[:2]
            translated = rotated + orders @ basis
            keep = target_domain.contains(translated)
            for target_point, order in zip(translated[keep], orders[keep]):
                candidate_points.append(target_point)
                candidate_generators.append(
                    ExpansionGenerator(
                        source_index,
                        operation,
                        (int(order[0]), int(order[1])),
                    )
                )
    if not candidate_points:
        raise ValueError("periodic expansion produced no points in the target domain")
    candidates = np.vstack(candidate_points)
    scale = max(
        float(np.max(np.linalg.norm(candidates, axis=1))),
        float(np.max(np.linalg.norm(basis, axis=1))),
        np.finfo(float).tiny,
    )
    groups = _group_candidates(candidates, tolerances.absolute + tolerances.relative * scale)

    records = []
    for indices in groups:
        generators = tuple(
            sorted((candidate_generators[index] for index in indices), key=_generator_key)
        )
        points = candidates[indices]
        representative = points[np.lexsort((points[:, 1], points[:, 0]))[0]]
        records.append((representative, generators))
    records.sort(key=lambda item: (item[0][0], item[0][1]))

    target_points = np.vstack([record[0] for record in records])
    physical_weights = voronoi_physical_weights(target_points, target_domain)
    weights = physical_weights / target_domain.area
    target = KSampling(
        target_points,
        weights,
        target_domain.area,
        target_domain,
        metadata={
            "source_region": source_region.value,
            "target_region": target_region.value,
            "reciprocal_basis": basis,
        },
    )
    equivalence_classes = tuple(record[1] for record in records)
    primary = tuple(generators[0] for generators in equivalence_classes)
    return ExpansionMap(source, target, primary, equivalence_classes)


def _representatives(
    zone: BrillouinZone,
    source_region: SourceRegion,
    constraint: SamplingConstraint,
) -> KSampling:
    sampler = BrillouinZoneSampler()
    converted = legacy_constraint(constraint)
    if source_region is SourceRegion.IBZ:
        sampled = sampler.sample_irreducible(zone, converted)
    elif source_region is SourceRegion.BZ:
        sampled = sampler.sample_full(zone, converted)
    else:
        raise ValueError("periodic plans require BZ or IBZ representatives")
    # IBZ weights are orbit-weighted and integrate over the full BZ. The BZ is
    # consequently the numerical integration domain even though representative
    # coordinates occupy its irreducible chamber.
    domain = brillouin_zone_domain(zone)
    result = KSampling(
        sampled.points[:, :2],
        sampled.weights,
        sampled.integration_element,
        domain,
        metadata={"source_region": source_region.value},
    )
    result = _canonicalize_sampling(result, zone.cell.basis[:, :2])
    operations = (
        zone.point_group
        if source_region is SourceRegion.IBZ
        else (_identity(zone.point_group),)
    )
    orbit_points = []
    orbit_sources = []
    for source_index, point in enumerate(result.points):
        for operation in operations:
            orbit_points.append(operation.apply(point)[:2])
            orbit_sources.append(source_index)
    candidates = np.vstack(orbit_points)
    scale = max(
        float(np.max(np.linalg.norm(candidates, axis=1))),
        zone.max_extent,
        np.finfo(float).tiny,
    )
    groups = _group_candidates(candidates, DEFAULT_TOLERANCES.relative * scale)
    unique_points = np.vstack(
        [
            candidates[group][
                np.lexsort((candidates[group, 1], candidates[group, 0]))[0]
            ]
            for group in groups
        ]
    )
    target_weights = voronoi_physical_weights(unique_points, domain)
    source_weights = np.zeros(len(result.points))
    for group, weight in zip(groups, target_weights):
        sources = sorted({orbit_sources[index] for index in group})
        for source_index in sources:
            source_weights[source_index] += weight / len(sources)
    keep = source_weights > np.finfo(float).eps * zone.area
    source_weights = source_weights[keep]
    source_weights /= np.sum(source_weights)
    return KSampling(
        result.points[keep],
        source_weights,
        result.integration_element,
        result.domain,
        metadata=result.metadata,
    )


def make_periodic_plan(
    direct_lattice: Lattice,
    physical_domain: KDomain,
    *,
    source: SourceRegion | str = SourceRegion.IBZ,
    target: TargetRegion | str = TargetRegion.PROPAGATING_SPECTRUM,
    constraint: SamplingConstraint = None,
    tolerances: Tolerances = DEFAULT_TOLERANCES,
) -> PeriodicSamplingPlan:
    if not isinstance(direct_lattice, Lattice):
        raise TypeError("direct_lattice must be a Lattice")
    if direct_lattice.lattice_type != "real_space":
        raise ValueError("periodic planning requires a direct-space lattice")
    if not isinstance(physical_domain, KDomain):
        raise TypeError("physical_domain must implement KDomain")
    source_region = SourceRegion(source)
    target_region = TargetRegion(target)
    if target_region is TargetRegion.EXTENDED_SPECTRUM and not isinstance(
        physical_domain, EvanescentDisk
    ):
        raise ValueError("extended-spectrum targets require an EvanescentDisk")
    if target_region is TargetRegion.PROPAGATING_SPECTRUM and isinstance(
        physical_domain, EvanescentDisk
    ):
        physical_domain = PropagationDisk(
            physical_domain.propagating_radius,
            physical_domain.center,
        )
    reciprocal = direct_lattice.make_reciprocal()
    zone = reciprocal.brillouin_zone
    representatives = _representatives(zone, source_region, constraint)
    target_domain: KDomain = (
        brillouin_zone_domain(zone) if target_region is TargetRegion.BZ else physical_domain
    )
    expansion = build_expansion_map(
        representatives,
        reciprocal,
        zone,
        target_domain,
        source_region,
        target_region,
        tolerances=tolerances,
    )
    return PeriodicSamplingPlan(
        direct_lattice,
        reciprocal,
        zone,
        source_region,
        target_region,
        representatives,
        expansion,
    )


class PeriodicKSpace:
    """A physical spectrum paired with a direct-space periodic structure."""

    def __init__(self, parent: object, direct_lattice: Lattice) -> None:
        if not isinstance(direct_lattice, Lattice):
            raise TypeError("direct_lattice must be a Lattice")
        if direct_lattice.lattice_type != "real_space":
            raise ValueError("with_periodic_structure expects a direct-space lattice")
        self.parent = parent
        self.direct_lattice = direct_lattice

    def plan(
        self,
        *,
        source: SourceRegion | str = SourceRegion.IBZ,
        target: TargetRegion | str = TargetRegion.PROPAGATING_SPECTRUM,
        constraint: SamplingConstraint = None,
        tolerances: Tolerances = DEFAULT_TOLERANCES,
    ) -> PeriodicSamplingPlan:
        domain = getattr(self.parent, "spectrum_domain", None)
        if not isinstance(domain, KDomain):
            raise ValueError("KSpace has no configured physical spectrum domain")
        plan = make_periodic_plan(
            self.direct_lattice,
            domain,
            source=source,
            target=target,
            constraint=constraint,
            tolerances=tolerances,
        )
        magnitude = getattr(self.parent, "wavevector_magnitude", None)
        if magnitude is None:
            return plan
        representatives = plan.representatives.with_kz(magnitude)
        target_sampling = plan.target.with_kz(magnitude)
        expansion = ExpansionMap(
            representatives,
            target_sampling,
            plan.expansion.primary_generators,
            plan.expansion.equivalent_generators,
        )
        return PeriodicSamplingPlan(
            plan.direct_lattice,
            plan.reciprocal_lattice,
            plan.zone,
            plan.source_region,
            plan.target_region,
            representatives,
            expansion,
        )


__all__ = [
    "ExpansionGenerator",
    "ExpansionMap",
    "PeriodicKSpace",
    "PeriodicSamplingPlan",
    "build_expansion_map",
    "make_periodic_plan",
]
