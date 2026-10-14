Non-periodic sampling
=====================

``KSpace.sample_domain()`` returns a ``SamplingResult`` containing points,
normalized quadrature weights, and the integration domain. Pass
``include_cells=True`` to retain each point's integration-cell geometry.
``Canvas.plot_integration_cells()`` adds and returns one Matplotlib patch per
point, in the same order as ``sampling.points`` and
``sampling.integration_cells``.

Colors below identify normalized integration weights, which sum to one.
Each point's physical integration weight is available through ``sampling.physical_weights``
or ``sampling.integration_cells[index].area``. The same cell-retention and
plotting API also works with ``KSpace.sample_pupil()``.

Cartesian sampling
------------------

The Cartesian grid places points at the centers of equally spaced rectangles
in the domain's bounding box. Points outside the propagating disk are removed;
retained points keep their original x and y coordinates, including near the
boundary. All retained points have the same normalized weight ``1 / N`` and
physical weight ``domain.area / N``.

The artists show full rectangles of width ``dx`` and height ``dy``, including
where they extend beyond the disk. Each rectangle's ``geometric_area`` is
``dx * dy``; its ``area`` records the effective quadrature area
``domain.area / N``. These quantities can differ because the masked grid
does not exactly tile the circular domain.

.. plot::
   :context: reset
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np

   from reciprocal import KSpace
   from reciprocal.canvas import Canvas, choose_color
   from reciprocal.spectrum import CartesianGrid, CircularGrid, PolarGrid


   def draw_sampling(grid):
       kspace = KSpace.propagating(wavelength=2 * np.pi)
       sampling = kspace.sample_domain(grid, include_cells=True)

       fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
       canvas = Canvas(ax)
       artists = canvas.plot_integration_cells(
           sampling,
           facecolor="#f5f7fa",
           edgecolor="#8b98a5",
           linewidth=0.7,
           zorder=1,
       )
       canvas.plot_domain(sampling.domain, linewidth=1.5, zorder=2)

       # The result carries weights and geometry in sampling-point order.
       # Each returned artist can be styled independently.
       unique_weights, groups = np.unique(
           np.round(sampling.normalized_weights, decimals=14), return_inverse=True,
       )
       for index, weight in enumerate(unique_weights):
           canvas.plot_point_sampling(
               sampling.points[groups == index],
               color=choose_color(index, len(unique_weights)).ravel(),
               label=f"{weight:.5f}",
               edgecolor="white",
               linewidth=0.6,
               s=40,
               zorder=3,
           )
       ax.legend(
           title="Integration weight",
           loc="center left",
           bbox_to_anchor=(1.02, 0.5),
       )
       ax.set(
           xlabel=r"$k_x$",
           ylabel=r"$k_y$",
           xlim=(-1.15, 1.15),
           ylim=(-1.15, 1.15),
           aspect="equal",
       )
       return fig


   draw_sampling(CartesianGrid((9, 9)))

Polar sampling
--------------

The polar grid uses five radial bands and 24 angular sectors per band.
Radial boundaries are equally spaced in radius squared, giving every annular
sector the same area and weight. Points lie inside their sectors, so this
grid does not place a sample at the origin. The returned artists are
Matplotlib ``Wedge`` patches.

.. plot::
   :context: close-figs
   :include-source:

   draw_sampling(PolarGrid(radial=5, azimuthal=24))

Circular sampling with adaptive ring counts
-------------------------------------------

``CircularGrid`` follows the center-and-rings approach of the legacy
``_sample_circular`` method. ``radial=5`` produces one point at
``kx = ky = 0`` and four nonzero-radius rings. For disk radius ``R``, the
fixed ring spacing is ``dr = R / (radial - 0.5)``. The center point owns a
circular cell of radius ``dr / 2``.

Each ring at radius ``r`` chooses the nearest integer to ``2 * pi * r / dr``
for its azimuthal point count. Counts increase outwards, keeping arc spacing
close to ``dr``. Ring points lie at the centers of annular wedges with radial
edges halfway between neighboring rings; the last wedges reach ``R``.
Weights are the circle and wedge areas divided by the disk area, so the
integration cells partition the disk and the weights sum to one.

The returned artists are one ``Circle`` followed by ``Wedge`` patches.
This grid supports origin-centered propagating disks, extended-spectrum
disks, and circular pupils without a central obstruction. An annular or
shifted domain cannot provide the required circular cell at the origin.

.. plot::
   :context: reset
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np

   from reciprocal import KSpace
   from reciprocal.canvas import Canvas, choose_color
   from reciprocal.spectrum import CircularGrid

   space = KSpace.propagating(wavelength=2 * np.pi)
   sampling = space.sample_domain(
       CircularGrid(radial=5),
       include_cells=True,
   )

   fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
   canvas = Canvas(ax)
   # One Circle for the origin, then one Wedge per point on each ring.
   artists = canvas.plot_integration_cells(
       sampling,
       facecolor="#f5f7fa",
       edgecolor="#8b98a5",
       linewidth=0.7,
       zorder=1,
   )
   canvas.plot_domain(sampling.domain, linewidth=1.5, zorder=2)

   weights, groups = np.unique(
       np.round(sampling.normalized_weights, decimals=14), return_inverse=True,
   )
   for index, weight in enumerate(weights):
       canvas.plot_point_sampling(
           sampling.points[groups == index],
           color=choose_color(index, len(weights)).ravel(),
           label=f"{weight:.5f}",
           edgecolor="white",
           linewidth=0.6,
           s=40,
           zorder=3,
       )
   ax.legend(
       title="Integration weight",
       loc="center left",
       bbox_to_anchor=(1.02, 0.5),
   )
   ax.set(
       xlabel=r"$k_x$",
       ylabel=r"$k_y$",
       xlim=(-1.15, 1.15),
       ylim=(-1.15, 1.15),
       aspect="equal",
       title="CircularGrid: center and four rings",
   )

Non-periodic contents with D4 symmetry
--------------------------------------

The content symmetry describes a structure whose interior and material
response share D4, such as a symmetric structure with a square footprint.
Only incidence directions in the 0--45 degree cone need to be simulated for
a response with the declared transformation law. The map reconstructs the
complete physical spectrum using point operations, with no lattice
translations. The origin and symmetry-boundary points retain their actual
orbit multiplicities.

.. plot::
   :context: reset
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np
   from matplotlib.patches import Wedge

   from reciprocal import KSpace
   from reciprocal.canvas import Canvas
   from reciprocal.spectrum import CircularGrid, ContentSymmetry
   from reciprocal.spectrum import FieldSamples, ScalarRepresentation

   space = KSpace.propagating(wavelength=2 * np.pi)
   representatives = space.sample_domain(
       CircularGrid(radial=5),
       content_symmetry=ContentSymmetry("D4"),
       symmetry_cone="irreducible",
       include_cells=True,
   )
   mapping = representatives.symmetry_reduction
   full = mapping.target
   # An invariant scalar response stands in for solver output here.
   values = np.sum(representatives.points**2, axis=1)
   data = FieldSamples(representatives, values, ScalarRepresentation())
   expanded = mapping.expand(data)
   np.testing.assert_allclose(
       expanded.values, np.sum(expanded.sampling.points**2, axis=1),
   )

   fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
   for ax, sampling, title in (
       (axes[0], full, f"Full incidence grid: {len(full.points)} points"),
       (axes[1], representatives,
        f"D4 representatives: {len(representatives.points)} solver points"),
   ):
       canvas = Canvas(ax)
       canvas.plot_integration_cells(
           sampling, facecolor="#f5f7fa", edgecolor="#8b98a5", linewidth=0.7,
       )
       canvas.plot_domain(space.spectrum_domain, linewidth=1.5)
       canvas.plot_point_sampling(sampling, s=28, color="black", zorder=3)
       ax.set(
           xlabel=r"$k_x$", ylabel=r"$k_y$", title=title,
           xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), aspect="equal",
       )
   axes[1].add_patch(Wedge(
       (0, 0), 1, 0, 45, fill=False, edgecolor="#2868a9", linewidth=2,
   ))

The displayed cells are representative footprints; their quadrature weights
include all members of the corresponding full-spectrum orbit. Use
``FieldSamples`` with vector, tensor, or polarization representations to
transform those responses appropriately. Fixed-polarization illumination
needs the required polarization channels or a compatible subgroup.
