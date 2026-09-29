# Reciprocal

`reciprocal` constructs two-dimensional direct and reciprocal lattices and
samples unit cells, Brillouin zones, and bounded reciprocal space. The core
package uses NumPy and SciPy; plotting is optional.

## Installation

Install the core library from a wheel or source checkout:

```console
python -m pip install reciprocal
```

Add plotting support with `python -m pip install "reciprocal[plot]"`, or use
`python -m pip install -e ".[dev]"` when developing the project.

The Sphinx documentation includes a quickstart, API reference, and executable
plotting examples. Build it locally with:

```console
python -m pip install -e ".[docs]"
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

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
  `theta` is the unsigned angle to the surface normal in `[0, 90]`, `phi` is
  measured in the x-y plane from +x, and `normal=+1/-1` independently selects
  +z/-z propagation or evanescent-decay direction.
- Direct-lattice vectors use any consistent length unit. Reciprocal vectors
  and wave vectors use its inverse and include the `2π` convention, so
  `a_i · b_j = 2π δ_ij`. `wavelength` must use the direct-space length unit.
- Public array inputs accept NumPy-compatible real array-like values. A
  `KVectorGroup.k` array has shape `(N, 3)` and uses complex dtype when the
  batch contains evanescent modes; absorbing (complex-index) media are not
  supported. Evanescent modes have imaginary `kz` and `theta = NaN`.
  samplers return a `KVectorGroup` plus a one-dimensional weight array where
  applicable.
- Legacy geometry and sampling objects remain mutable for compatibility.
  `LatticeVectors`, `Lattice`, and the new cell-domain objects are immutable;
  construct a replacement value instead of modifying their arrays or identity.

The additive `reciprocal.cells` API provides the replacement immutable cell
model. Its `PolygonDomain`, `UnitCell`, `SamplingResult`, and all contained
arrays are immutable; builders copy their inputs. A cell basis has shape
`(2, 3)` with translation vectors in rows. Sampling weights are normalized to
sum to one, while `integration_element` is the physical domain area.

Numerical comparisons use the immutable `reciprocal.DEFAULT_TOLERANCES`
policy. Length and coordinate comparisons are relative by default, angular
comparisons have an absolute tolerance in degrees, and lattice degeneracy is
tested through the dimensionless normalized cross product. Construct a
`Tolerances` value and pass it to methods such as
`Lattice.determine_bravais_lattice()` when measurement resolution requires a
different policy. Point duplicate detection also accepts explicit relative and
absolute tolerances.

Invalid dimensions, non-finite values, non-positive lengths, and degenerate
lattices raise `ValueError`.

Construct optical wave vectors explicitly from angles, Cartesian components,
or transverse components. The latter form also represents evanescent modes:

```python
from reciprocal import KVector

incident = KVector.from_angles(
    wavelength=1.0e-6, n=1.5, theta=30.0, phi=0.0, normal=-1
)
evanescent = KVector.from_transverse(
    wavelength=1.0e-6, n=1.0, kx=8.0e6, ky=0.0, normal=1
)
assert evanescent.is_evanescent
```

`PeriodicSampler.sample_bloch_families()` returns `BlochFamily` objects whose
integer `orders`, `representative`, and `reciprocal_basis` retain and validate
`k_parallel = representative + orders @ reciprocal_basis`. Slicing a family
returns a `BlochVector` with the same provenance. For point symmetry,
`reciprocal.symmetry.point_orbit_with_operations()` returns every unique orbit
member together with all operations that generate it.

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

`make_reciprocal()` is defined only for a real-space lattice. Calling it on a
reciprocal lattice raises `ValueError`. Use `Lattice.from_vectors()` or
`Lattice.from_lengths_angle()` when constructing either space explicitly.

New code can construct explicit cell and reciprocal-space objects without
using the behavior-heavy compatibility `UnitCell`:

```python
from reciprocal.brillouin_zone import BrillouinZoneSampler
from reciprocal.cells import CellSampler

primitive = direct_lattice.primitive_cell
conventional = direct_lattice.conventional_cell
zone = reciprocal_lattice.brillouin_zone

cell_sample = CellSampler().sample(primitive, {"type": "n_points", "value": 9})
zone_sample = BrillouinZoneSampler().sample_irreducible(
    zone, {"type": "n_points", "value": 9}
)
```

Special-point labels use the conventional two-dimensional paths Γ/X/M for
square, Γ/X/Y/S for rectangular families, Γ/M/K for hexagonal, and Γ/X/Y/C
for oblique lattices. Low-symmetry polygon-vertex orbits additionally use
deterministic H1, H2, … labels. Coordinates are fractional coordinates of the
reduced reciprocal primitive basis; boundary-equivalent representatives are
selected inside the Wigner–Seitz cell. This is the standard crystallographic
convention used by the Bilbao Crystallographic Server's layer-group
Brillouin-zone tables.

Uniform electronic-structure meshes and band paths use the same zone data:

```python
from reciprocal import (
    MonkhorstPackGrid,
    make_high_symmetry_path,
    reduce_monkhorst_pack,
)

mesh = reduce_monkhorst_pack(zone, MonkhorstPackGrid((8, 8)))
path = make_high_symmetry_path(zone, points_per_segment=51)

assert mesh.degeneracies.sum() == 64
assert path.labels == ("Γ", "X", "M", "Γ")
```

Cell, Brillouin-zone, pupil, and periodic samplers share the immutable
`SamplingResult` contract. Normalized weights sum to one, physical weights sum
to the declared integration-domain area, and stable point IDs associate solver
values with their sampling.

Symmetry names include cyclic groups such as `C4` and dihedral groups such as
`D4`. When symmetry-cone restriction is enabled, only the irreducible angular
wedge is sampled. Its boundary is included within the sampler's numerical
tolerance.

## Spectrum workflows

The additive `reciprocal.spectrum` API uses one immutable result contract for
periodic and non-periodic sampling. `KSampling.normalized_weights` sum to one,
`physical_weights` sum to the domain area, and stable `point_ids` associate
solver output with the points supplied to the solver.

Sample a microscopy pupil without implying periodicity:

```python
from reciprocal import KSpace
from reciprocal.spectrum import PolarGrid

space = KSpace.propagating(wavelength=1.0e-6, refractive_index=1.0)
pupil = space.sample_pupil(
    PolarGrid(radial=32, azimuthal=128),
    numerical_aperture=0.8,
)
```

For a periodic structure, build representatives in the IBZ and a reusable
mapping onto the complete propagating spectrum:

```python
import numpy as np
from reciprocal import Lattice, LatticeVectors
from reciprocal.spectrum import (
    FieldSamples,
    IdentityTranslation,
    MaxSpacing,
    PolarVectorRepresentation,
)

direct = Lattice(
    LatticeVectors.from_lengths_angle(500e-9, 500e-9, 90.0)
)
plan = space.with_periodic_structure(direct).plan(
    source="ibz",
    target="propagating_spectrum",
    constraint=MaxSpacing(2.0e5),
)

# Replace this fixture with fields returned for plan.representatives.point_ids.
electric_field = np.ones((len(plan.representatives.points), 3), dtype=complex)
representatives = FieldSamples(
    plan.representatives,
    electric_field,
    PolarVectorRepresentation(IdentityTranslation()),
)
expanded = plan.expansion.expand(representatives, verify=False)
integrated_field = expanded.integrate()
```

`IdentityTranslation` is an explicit declaration that the supplied value is
unchanged when a Bloch representative is relabelled by a reciprocal vector.
Generic electric-field expansion rejects nonzero reciprocal translations by
default because spatial Bloch fields can instead require a phase or Fourier
index shift. Supply a solver-specific translation action when that is the
correct transformation. `FourierOrderTranslation` handles the common case in
which each representative result contains an array indexed by reciprocal
orders; it rotates a target order back into the source solution before
selecting its coefficient.

`ExpansionMap` retains the source point, concrete point operation, reciprocal
order, and every equivalent generator for each unique target. Consequently,
points generated by both a reciprocal translation and a point operation occur
only once in the target sampling. Scalar, polar-vector, axial-vector, tensor,
and local s/p representations are available. Integration supports transverse
area, normalized area, solid angle, and user Jacobians; `interpolate()`
supports nearest, piecewise-linear, and Clough–Tocher scattered interpolation.

A finite evanescent extension is constructed with `KSpace.extended()`. The
extension always requires `max_parallel_wavevector`; unbounded evanescent
integration is intentionally rejected. Run `python examples/spectrum_workflow.py`
for an executable periodic and pupil example.

## Plotting

Plotting is explicit: `Canvas` accepts a Matplotlib `Axes` and never calls
`show()` or writes a file. The caller owns the figure and output path.

```python
import matplotlib.pyplot as plt
from reciprocal.canvas import Canvas

fig, ax = plt.subplots()
canvas = Canvas(ax)
canvas.plot_spectrum(space)
canvas.plot_point_sampling_weighted(pupil)
fig.savefig("sampling.png", dpi=150)
```

The same adapters accept the refactored periodic and field objects directly:

```python
canvas.plot_bzone(plan)
canvas.plot_ibzone(plan)
canvas.plot_bloch_families(plan)
canvas.plot_point_sampling_weighted(plan.target)
canvas.plot_field(expanded, magnitude=True)
```

`plot_tessellation()` uses `Lattice.primitive_cell` for direct lattices and
the immutable Brillouin-zone cell for reciprocal lattices. It no longer
materializes the deprecated `Lattice.unit_cell`. `plot_interpolation()` accepts
`FieldSamples` or `InterpolationResult` and uses SciPy's supported
`Delaunay.simplices` interface.

Run `python examples/sampling_plot.py sampling.png` for a deterministic,
complete version of this workflow.

## Citation

No archival publication is assigned to this software. In publications, cite
“Reciprocal, version 1.0.0, JCMwave,
https://github.com/jcmwave/reciprocal” and include the access date. The
installed version is available as `reciprocal.__version__`.

## Development and reproducibility

Run the same checks used by CI with:

```console
python -m ruff check .
python -m ruff format --check src/reciprocal/__init__.py src/reciprocal/numerics.py examples tests/conftest.py tests/test_numerics.py tests/test_properties.py tests/test_public_api.py
python -m mypy
python -m pytest
```

The pre-commit configuration runs linting, import sorting, formatting, and the
public API type check; its pre-push stage runs unit tests. Property tests use
fixed Hypothesis generation and cover reciprocal-basis, symmetry, domain, and
constant-integration invariants.

Figure tests force Matplotlib's non-interactive backend, write only below
pytest temporary directories, and close all figures. Golden documentation
images, if added later, must be generated and committed through an explicit
documentation update rather than a unit-test side effect.

New code uses snake-case method names. Compatibility aliases such as
`KVector.getNFromK()`, `Canvas.plot_tesselation()`, and
`PeriodicSampler.plotSymmetryFamilies()` remain available for the 1.x series
but emit `DeprecationWarning`; use `get_n_from_k()`, `plot_tessellation()`, and
`plot_symmetry_families()` respectively.
