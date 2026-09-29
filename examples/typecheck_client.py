"""Representative core client checked by mypy in CI."""

import numpy as np

from reciprocal import KSpace, KVectorGroup, Lattice, LatticeVectors, SamplingResult
from reciprocal.spectrum import PolarGrid

vectors: LatticeVectors = LatticeVectors.from_lengths_angle(1.0, 1.0, 90.0)
lattice: Lattice = Lattice(vectors)
space: KSpace = KSpace(np.pi, symmetry="D4", fermi_radius=2.0)
sampling: KVectorGroup
sampling, weights = space.regular_sampler.sample(constraint={"type": "n_points", "value": 25})
assert sampling.n_rows == weights.shape[0]
assert lattice.primitive_cell.area > 0.0

pupil: SamplingResult = space.sample_pupil(PolarGrid(3, 8))
assert pupil.normalized_weights.shape == (24,)
