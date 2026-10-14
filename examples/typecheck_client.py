"""Representative core client checked by mypy in CI."""

import numpy as np

from reciprocal import (
    KSpace,
    KVectorGroup,
    Lattice,
    LatticeVectors,
    MaxSpacing,
    PointCounts,
    SamplingResult,
)
from reciprocal.spectrum import (
    CircularGrid,
    ContentSymmetry,
    FieldSamples,
    PolarGrid,
    ScalarRepresentation,
)

vectors: LatticeVectors = LatticeVectors.from_lengths_angle(1.0, 1.0, 90.0)
lattice: Lattice = Lattice(vectors)
space: KSpace = KSpace(np.pi, symmetry="D4", fermi_radius=2.0)
sampling: KVectorGroup
sampling, weights = space.regular_sampler.sample(constraint={"type": "n_points", "value": 25})
assert sampling.n_rows == weights.shape[0]
assert lattice.primitive_cell.area > 0.0

pupil: SamplingResult = space.sample_pupil(PolarGrid(3, 8))
assert pupil.normalized_weights.shape == (24,)

circular: SamplingResult = space.sample_domain(CircularGrid(radial=5), include_cells=True)
assert circular.integration_cells is not None

constrained: SamplingResult = space.sample_domain("cartesian", constraint=MaxSpacing((0.1, 0.2)))
reduced: SamplingResult = space.sample_domain(
    "circular",
    constraint=PointCounts(5),
    content_symmetry=ContentSymmetry("D4"),
    symmetry_cone="irreducible",
    include_cells=True,
)
assert reduced.symmetry_reduction is not None
data: FieldSamples[float] = FieldSamples(
    reduced, np.ones(len(reduced.points)), ScalarRepresentation()
)
expanded = reduced.symmetry_reduction.expand(data)
anomalies = space.with_periodic_structure(lattice).wood_anomalies(shells=(1, 2))
edges = anomalies.sample_edges(MaxSpacing(0.1))
assert edges.physical_weights.shape == (len(edges.points),)
