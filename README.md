# Reciprocal

`reciprocal` constructs two-dimensional direct and reciprocal lattices and
samples unit cells, Brillouin zones, and bounded reciprocal space. The core
package uses NumPy, SciPy, and Shapely; plotting is optional.

## Installation

Install the core library from a wheel or source checkout:

```console
python -m pip install reciprocal
```

Add plotting support with `python -m pip install "reciprocal[plot]"`, or use
`python -m pip install -e ".[dev]"` when developing the project.

Only the `reciprocal` top-level package is supported and included in built
distributions. The historical, incomplete `kspace_sample`/`kspacesampling`
tree is not part of the distribution. Modules beneath `reciprocal` remain
available for compatibility, but the names exported directly from
`reciprocal` form the stable public API for the 1.x series.

## Conventions and data contracts

- Coordinates are right-handed Cartesian coordinates. Two-dimensional inputs
  may have shape `(2,)`; `(3,)` inputs use a zero z component for lattice
  geometry.
- Lattice angles and the `theta`/`phi` arguments of `KVector` are in degrees.
  `theta` is measured from +z and `phi` in the x-y plane from +x.
- Direct-lattice vectors use any consistent length unit. Reciprocal vectors
  and wave vectors use its inverse and include the `2π` convention, so
  `a_i · b_j = 2π δ_ij`. `wavelength` must use the direct-space length unit.
- Public array inputs accept NumPy-compatible real array-like values. Returned
  arrays have floating dtype. A `KVectorGroup.k` array has shape `(N, 3)`;
  samplers return a `KVectorGroup` plus a one-dimensional weight array where
  applicable.
- Geometry and sampling objects are mutable for compatibility. Treat arrays
  obtained from their properties as views: copy them before modifying data
  that the object should retain.

Invalid dimensions, non-finite values, non-positive lengths, and degenerate
lattices raise `ValueError`.

## Lattices and sampling

The package root contains the supported imports used by common workflows:

```python
import numpy as np
from reciprocal import KSpace, Lattice, LatticeVectors

direct_vectors = LatticeVectors.from_lengths_angle(
    length1=500e-9,
    length2=500e-9,
    angle=90.0,
)
direct_lattice = Lattice(direct_vectors)
reciprocal_lattice = direct_lattice.make_reciprocal()

# Regular sampling inside |k| <= 2π / wavelength.
wavelength = 1.0e-6
space = KSpace(wavelength, symmetry="D4", fermi_radius=2 * np.pi / wavelength)
points, weights = space.regular_sampler.sample(
    grid_type="cartesian",
    constraint={"type": "n_points", "value": 81},
    restrict_to_sym_cone=True,
)

# Lattice-aware (periodic) sampling.
space.apply_lattice(direct_lattice)
periodic_points = space.periodic_sampler.sample(
    constraint={"type": "n_points", "value": 81},
    use_symmetry=True,
)
```

Symmetry names include cyclic groups such as `C4` and dihedral groups such as
`D4`. When symmetry-cone restriction is enabled, only the irreducible angular
wedge is sampled. Its boundary is included within the sampler's numerical
tolerance.

## Plotting

Plotting is explicit: `Canvas` accepts a Matplotlib `Axes` and never calls
`show()` or writes a file. The caller owns the figure and output path.

```python
import matplotlib.pyplot as plt
from reciprocal.canvas import Canvas

fig, ax = plt.subplots()
canvas = Canvas(ax)
canvas.plot_point_sampling(points)
canvas.plot_fermi_circle(space)
fig.savefig("sampling.png", dpi=150)
```

Run `python examples/sampling_plot.py sampling.png` for a deterministic,
complete version of this workflow.

## Citation

No archival publication is assigned to this software. In publications, cite
“Reciprocal, version 1.0.0, JCMwave,
https://github.com/jcmwave/reciprocal” and include the access date. The
installed version is available as `reciprocal.__version__`.
