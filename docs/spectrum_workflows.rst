Spectrum constraints, Wood anomalies, and content symmetry
==========================================================

Shared separation constraints
-----------------------------

``PointCounts(n)`` describes one dimension or broadcasts the same count to
two dimensions. ``PointCounts(nx, ny)`` supplies independent counts.
``MaxSpacing(h)`` supplies one physical spacing bound;
``MaxSpacing((hx, hy))`` supplies separate bounds. All spacings use the
inverse length unit of the wavevector coordinates.

``resolve_point_counts()`` is shared by periodic and non-periodic samplers.
It distinguishes grids of cell centers, endpoint-inclusive grids, and closed
curves. Counts refer to the grid before domain masking. A spacing constraint
bounds the underlying grid separation; masking can leave missing positions.

.. code-block:: python

   from reciprocal import KSpace, MaxSpacing, PointCounts

   space = KSpace.propagating(wavelength=500e-9)
   cartesian = space.sample_domain(
       "cartesian", constraint=MaxSpacing((1e6, 2e6)),
   )
   circular = space.sample_domain("circular", constraint=PointCounts(5))
   polar = space.sample_domain("polar", constraint=PointCounts(5, 24))

Existing explicit grid objects remain supported. Supplying both an explicit
grid object and a constraint raises ``ValueError``. Scheme names select a
grid from the shared resolver; they are ``"cartesian"``, ``"circular"``, and
``"polar"``.

For Cartesian grids the two dimensions are x and y. For polar grids they are
radial bands and angular sectors. Circular grids count the origin plus rings;
a second point count sets the outermost ring's angular count, with counts on
inner rings scaled by circumference. A scalar circular count leaves angular
counts adaptive. Circular maximum-spacing inputs bound radial separation
and arc spacing at each sampling ring. Polar inputs also bound angular arc
spacing at the domain's outer radius and widths of equal-area radial bands.

Periodic cell/BZ grids use endpoint-inclusive spacing along their cell-basis
directions. Monkhorst--Pack grids have periodic, rather than duplicated,
endpoints:

.. code-block:: python

   from reciprocal import BoundaryGrid, Lattice, MonkhorstPackGrid
   from reciprocal import sample_brillouin_zone

   lattice = Lattice.from_lengths_angle(500e-9, 500e-9, 90.0)
   zone = lattice.make_reciprocal().brillouin_zone
   constraint = MaxSpacing((1e6, 2e6))
   boundary = sample_brillouin_zone(zone, BoundaryGrid(constraint))
   mesh = MonkhorstPackGrid.from_constraint(zone, constraint)
   periodic = sample_brillouin_zone(zone, mesh)

Legacy ``n_points`` and ``max_length`` dictionaries are normalized through
the same resolver. Scalar constraints retain their dimensionality when
converted with ``as_legacy()``. Density constraints are not part of this API.

Wood-anomaly circles and edge sampling
--------------------------------------

``PeriodicKSpace.wood_anomalies()`` returns immutable geometry before any
point discretization. With the convention
``k_diffracted = k_incident + G``, a grazing order satisfies
``|k_incident + G| = n * k0``. Its circle is centered at ``-G`` and has radius
``n * k0``. Shell indices group reciprocal orders by increasing translation
length; each circle also carries its integer diffraction order.

Only circle edges that enter or touch the complete propagating disk are
returned. A filled circle enclosing the disk, whose edge never enters it,
is excluded. The propagating magnitude is taken from the configured
refractive index, including when ``KSpace`` is configured with a smaller
pupil or a finite evanescent extension.

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np
   from reciprocal import KSpace, Lattice, MaxSpacing
   from reciprocal.canvas import Canvas

   direct = Lattice.from_lengths_angle(500e-9, 500e-9, 90.0)
   scale = min(direct.make_reciprocal().vectors.lengths)
   space = KSpace.propagating(wavelength=2 * np.pi / (2.2 * scale))
   anomalies = space.with_periodic_structure(direct).wood_anomalies(
       shells=(1, 2, 3, 4),
   )
   edge_samples = anomalies.sample_edges(MaxSpacing(0.15 * scale))
   fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
   canvas = Canvas(ax)
   canvas.plot_spectrum(space, edgecolor="black")
   artists = canvas.plot_wood_anomalies(anomalies, linewidth=0.8)
   canvas.plot_point_sampling(edge_samples, s=6, color="black")
   ax.set(xlabel=r"$k_x$", ylabel=r"$k_y$", title="Wood-anomaly edge samples")

``CurveSamplingResult`` carries arc-length weights, circle indices,
diffraction orders, stable point identities, and segment boundaries. Each
connected visible arc receives its own count or maximum-spacing constraint.
Open arcs use endpoint-inclusive trapezoidal quadrature; closed circles omit
the repeated endpoint. ``PointCounts(1)`` uses one midpoint with the complete
arc's length as its weight. Integrating a constant returns the summed visible
arc length, not an area. Tangent contacts have zero length and are recorded
separately in ``tangent_points`` and ``tangent_circle_indices``. Empty or
tangent-only results have empty normalized weights and integrate to zero;
their average is undefined.

Symmetry of non-periodic contents
---------------------------------

``ContentSymmetry`` describes the complete structure/material response and
its excitation covariance; it does not attach a reciprocal lattice to the
spectrum. ``Cn`` and ``Dn`` specify finite rotations and reflections.
``orientation_degrees`` sets the reflection axes and the start of the
canonical cone. D4 at zero orientation uses incidence angles from 0 to 45
degrees. A square footprint is sufficient only when the interior and
material response also preserve D4.

.. code-block:: python

   from reciprocal.spectrum import CircularGrid, ContentSymmetry
   from reciprocal.spectrum import FieldSamples, ScalarRepresentation

   reduced = space.sample_domain(
       CircularGrid(radial=5),
       content_symmetry=ContentSymmetry("D4"),
       symmetry_cone="irreducible",
       include_cells=True,
   )
   # Replace this invariant scalar example with the simulated response.
   values = np.sum(reduced.points**2, axis=1)
   field = FieldSamples(reduced, values, ScalarRepresentation())
   mapping = reduced.symmetry_reduction
   expanded = mapping.expand(field)
   integral = field.integrate()

The returned ``SamplingResult`` has representative coordinates, weights
summed over full physical point orbits, and the full integration domain.
The origin and symmetry-axis points have smaller orbit sizes and are not
blindly assigned the full group multiplicity. The companion
``NonPeriodicSymmetryReduction`` retains the full target, generating
Cartesian operations, a full-to-reduced mapping, and a speedup ratio.
No reciprocal translations are used.

An explicit ``SymmetryCone(start_degrees, end_degrees)`` or an angular pair
can replace ``"irreducible"``. Every full orbit must have a representative
inside it. A too-small or misplaced cone raises an error instead of silently
dropping incidence directions. The cone limits representative coordinates;
it does not change the declared full-domain integration measure. Its actual
geometric domain is recorded as ``metadata["coordinate_domain"]``.

Grid and domain compatibility
-----------------------------

The declared symmetry must preserve the domain and the full quadrature.
Cartesian counts and radial angular counts can be incompatible with a
group. Explicit incompatible grids raise ``ValueError`` by default;
``adapt_symmetry=True`` permits compatible count increases. Rotated or
non-square Cartesian layouts that cannot preserve a group still raise an
error; a circular grid is often suitable instead.

When a scheme is resolved from a maximum-spacing constraint, compatible
count increases are automatic: they preserve the requested upper bounds.
Exact point counts and explicit grid objects require explicit adaptation.

Adaptive circular counts are chosen from multiples of the rotational group
order, and their angular origin follows the content orientation. Circular
maximum-spacing bounds remain upper bounds after these adjustments. Fixed
polar or outer-ring counts require compatibility or explicit adaptation.

``OrbitIntegrationCell`` retains a representative's original cell footprint
and its full-orbit quadrature allocation separately. The canvas plots those
footprints; ``area`` is the full-orbit integration weight and
``geometric_area`` is the footprint area. ``with_kz()`` preserves the mapping
and retained cells.

Response transformations
------------------------

Use ``ScalarRepresentation`` for an invariant scalar, or declare the
appropriate polar-vector, axial-vector, tensor, or s/p polarization
representation. ``mapping.expand(field)`` transforms the response into all
target directions. At symmetry boundaries, equivalent operations must
produce consistent responses; expansion checks this by default.

``FieldSamples.integrate()`` and ``average()`` apply the map before
integration when the sampling is reduced. This also handles vectors and
polarization correctly. Applying ``SamplingResult.integrate(values)``
directly to reduced values is appropriate only for invariant scalar
integrands. For fixed-polarization illumination, supply all necessary
response channels or declare the subgroup that preserves the excitation.

The contents need to share the declared symmetry: it is supplied by the
caller, not inferred from numerical solver results. Existing legacy
symmetry algebra and arbitrary-point weighting are unchanged.
