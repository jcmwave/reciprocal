Symmetry-expanded azimuthal vector families
============================================

Oblique
-------

.. plot::
   :context: reset
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np
   from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator

   from reciprocal import KSpace, Lattice
   from reciprocal.canvas import Canvas, choose_color
   from reciprocal.spectrum import (
       AxialVectorRepresentation,
       IdentityTranslation,
       PointCounts,
   )


   def scale_reciprocal_axes(ax, lattice):
       reciprocal_scale = min(lattice.vectors.lengths)
       for axis, limits in (
           (ax.xaxis, ax.get_xlim()),
           (ax.yaxis, ax.get_ylim()),
       ):
           normalized_limits = np.asarray(limits) / reciprocal_scale
           normalized_ticks = MaxNLocator(nbins=5).tick_values(*normalized_limits)
           axis.set_major_locator(FixedLocator(normalized_ticks * reciprocal_scale))
           axis.set_major_formatter(
               FuncFormatter(lambda value, position: f"{value / reciprocal_scale:g}")
           )
       ax.set(
           xlabel=r"$k_x / |\mathbf{b}_{\min}|$",
           ylabel=r"$k_y / |\mathbf{b}_{\min}|$",
       )


   def symmetry_sector_families(plan):
       reciprocal_scale = min(plan.reciprocal_lattice.vectors.lengths)
       sector_scale = 0.1 * plan.zone.max_extent / np.max(
           np.linalg.norm(plan.target.points, axis=1)
       )

       parents = list(range(len(plan.representatives.points)))

       def find(source_index):
           while parents[source_index] != source_index:
               parents[source_index] = parents[parents[source_index]]
               source_index = parents[source_index]
           return source_index

       def union(first, second):
           first_root = find(first)
           second_root = find(second)
           parents[max(first_root, second_root)] = min(first_root, second_root)

       for generators in plan.expansion.equivalent_generators:
           source_indices = sorted({
               generator.source_index for generator in generators
           })
           for source_index in source_indices[1:]:
               union(source_indices[0], source_index)

       sector_candidates = {}
       for target_index, point in enumerate(plan.target.points):
           orbit_key = tuple(sorted(
               tuple(np.round(
                   operation.apply(point)[:2] / reciprocal_scale,
                   decimals=12,
               ))
               for operation in plan.zone.point_group
           ))
           sector_candidates.setdefault(orbit_key, [])
           if plan.zone.irreducible_domain.contains(point * sector_scale)[0]:
               sector_candidates[orbit_key].append(target_index)

       families_by_source = {}
       for candidates in sector_candidates.values():
           if not candidates:
               raise RuntimeError("a target orbit does not enter the symmetry sector")
           target_index = min(
               candidates,
               key=lambda index: tuple(np.round(
                   plan.target.points[index] / reciprocal_scale,
                   decimals=12,
               )),
           )
           generator = plan.expansion.primary_generators[target_index]
           source_root = find(generator.source_index)
           families_by_source.setdefault(source_root, []).append(
               plan.target.points[target_index]
           )

       return {
           family_index: families_by_source[source_root]
           for family_index, source_root in enumerate(sorted(families_by_source))
       }


   def azimuthal_vector(point):
       radius = np.linalg.norm(point)
       if radius == 0.0:
           return np.zeros(3)
       return np.array([-point[1] / radius, point[0] / radius, 0.0])


   def expand_vector_families(plan, families, reciprocal_scale):
       # The prescribed counterclockwise azimuthal field transforms as
       # det(R) R v: the determinant compensates for reflection handedness.
       representation = AxialVectorRepresentation(IdentityTranslation())
       expanded = {}
       for family_index, members in sorted(families.items()):
           targets = {}
           for source in members:
               vector = azimuthal_vector(source)
               for operation in plan.zone.point_group:
                   target = operation.apply(source)[:2]
                   transformed = representation.transform(
                       vector,
                       operation,
                       source,
                       target,
                       (0, 0),
                   )
                   key = tuple(np.round(target / reciprocal_scale, decimals=12))
                   targets.setdefault(key, (target, transformed[:2]))
           expanded[family_index] = (
               np.vstack([item[0] for item in targets.values()]),
               np.vstack([item[1] for item in targets.values()]),
           )
       return expanded


   def draw_vector_families(a, b, angle):
       direct_lattice = Lattice.from_lengths_angle(a, b, angle)
       lattice = direct_lattice.make_reciprocal()
       zone = lattice.brillouin_zone
       reciprocal_scale = min(lattice.vectors.lengths)
       k0 = 2.2 * reciprocal_scale
       kspace = KSpace.propagating(wavelength=2.0 * np.pi / k0)
       plan = kspace.with_periodic_structure(direct_lattice).plan(
           source="ibz",
           target="propagating_spectrum",
           constraint=PointCounts(5),
       )
       solver_families = symmetry_sector_families(plan)
       vector_families = expand_vector_families(
           plan,
           solver_families,
           reciprocal_scale,
       )
       tessellation_orders = np.array([
           (i, j) for i in range(-5, 6) for j in range(-5, 6)
       ])

       fig, ax = plt.subplots(figsize=(6, 5))
       canvas = Canvas(ax)
       canvas.plot_tessellation(
           lattice,
           orders=tessellation_orders,
           facecolor="#f7f9fb",
           edgecolor="#c4ced8",
           linewidth=0.8,
           increase_bbox=False,
       )
       canvas.plot_bzone(
           zone,
           facecolor=(0.90, 0.25, 0.18, 0.10),
           edgecolor="#c7352b",
           linewidth=2.5,
       )
       canvas.plot_ibzone(
           zone,
           facecolor=(0.20, 0.45, 0.80, 0.22),
           edgecolor="#2868a9",
           linewidth=2.0,
       )
       canvas.plot_spectrum(
           plan,
           fill=False,
           edgecolor="black",
           linewidth=2.0,
       )

       arrow_length = 0.195 * reciprocal_scale
       for color_index, (_family_index, (points, vectors)) in enumerate(
           sorted(vector_families.items())
       ):
           ax.quiver(
               points[:, 0],
               points[:, 1],
               arrow_length * vectors[:, 0],
               arrow_length * vectors[:, 1],
               color=choose_color(color_index, len(vector_families)).ravel(),
               angles="xy",
               scale_units="xy",
               scale=1.0,
               width=0.004,
               zorder=5,
           )

       margin = 0.08 * k0
       ax.set_xlim(-k0 - margin, k0 + margin)
       ax.set_ylim(-k0 - margin, k0 + margin)
       ax.set_aspect("equal")
       scale_reciprocal_axes(ax, lattice)
       return fig


   draw_vector_families(1000.0, 500.0, 75.0)

Rectangular
-----------

.. plot::
   :context: close-figs
   :include-source:

   draw_vector_families(1000.0, 500.0, 90.0)

Centered rectangular
--------------------

.. plot::
   :context: close-figs
   :include-source:

   draw_vector_families(500.0, 500.0, 70.0)

Square
------

.. plot::
   :context: close-figs
   :include-source:

   draw_vector_families(500.0, 500.0, 90.0)

Hexagonal
---------

.. plot::
   :context: close-figs
   :include-source:

   draw_vector_families(500.0, 500.0, 60.0)
