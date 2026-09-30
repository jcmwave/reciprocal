Symmetry-expanded Bloch families
================================

Oblique
-------

.. plot::
   :context: reset
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np
   from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator

   from reciprocal import KSpace, Lattice
   from reciprocal.canvas import Canvas
   from reciprocal.spectrum import PointCounts


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


   def draw_families(a, b, angle):
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
       canvas.plot_bloch_families(plan)

       margin = 0.08 * k0
       ax.set_xlim(-k0 - margin, k0 + margin)
       ax.set_ylim(-k0 - margin, k0 + margin)
       ax.set_aspect("equal")
       scale_reciprocal_axes(ax, lattice)
       return fig


   draw_families(1000.0, 500.0, 75.0)

Rectangular
-----------

.. plot::
   :context: close-figs
   :include-source:

   draw_families(1000.0, 500.0, 90.0)

Centered rectangular
--------------------

.. plot::
   :context: close-figs
   :include-source:

   draw_families(500.0, 500.0, 70.0)

Square
------

.. plot::
   :context: close-figs
   :include-source:

   draw_families(500.0, 500.0, 90.0)

Hexagonal
---------

.. plot::
   :context: close-figs
   :include-source:

   draw_families(500.0, 500.0, 60.0)
