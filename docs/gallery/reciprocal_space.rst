Reciprocal-space lattices
=========================

Oblique
-------

.. plot::
   :context: reset
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np
   from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator

   from reciprocal import Lattice
   from reciprocal.canvas import Canvas


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


   def draw_lattice(a, b, angle):
       direct_lattice = Lattice.from_lengths_angle(a, b, angle)
       lattice = direct_lattice.make_reciprocal()
       visible_orders = np.array([
           (i, j) for i in range(-1, 2) for j in range(-1, 2)
       ])
       plot_orders = np.array([
           (i, j) for i in range(-4, 5) for j in range(-4, 5)
       ])

       fig, ax = plt.subplots(figsize=(6, 5))
       canvas = Canvas(ax)
       canvas.plot_tessellation(
           lattice,
           orders=plot_orders,
           facecolor="#edf3f8",
           edgecolor="#9aadc0",
           linewidth=1.0,
           increase_bbox=False,
       )
       canvas.plot_lattice(
           lattice,
           orders=plot_orders,
           facecolor="#172b4d",
           increase_bbox=False,
       )
       canvas.plot_vectors(lattice)
       canvas.plot_bzone(
           lattice,
           facecolor=(0.90, 0.25, 0.18, 0.18),
           edgecolor="#c7352b",
           linewidth=2.5,
       )
       canvas.plot_ibzone(
           lattice,
           facecolor=(0.20, 0.45, 0.80, 0.22),
           edgecolor="#2868a9",
           linewidth=2.0,
       )
       canvas.plot_special_points(lattice)

       for label, vector in zip((r"$b_1$", r"$b_2$"), lattice.vectors.basis):
           ax.annotate(
               label,
               xy=vector[:2],
               xytext=(5, 5),
               textcoords="offset points",
               color="darkgreen",
           )

       positions = visible_orders @ lattice.vectors.basis[:, :2]
       margin = 0.45 * max(lattice.vectors.lengths)
       ax.set_xlim(positions[:, 0].min() - margin, positions[:, 0].max() + margin)
       ax.set_ylim(positions[:, 1].min() - margin, positions[:, 1].max() + margin)
       scale_reciprocal_axes(ax, lattice)
       return fig


   draw_lattice(1000.0, 500.0, 75.0)

Rectangular
-----------

.. plot::
   :context: close-figs
   :include-source:

   draw_lattice(1000.0, 500.0, 90.0)

Centered rectangular
--------------------

.. plot::
   :context: close-figs
   :include-source:

   draw_lattice(500.0, 500.0, 70.0)

Square
------

.. plot::
   :context: close-figs
   :include-source:

   draw_lattice(500.0, 500.0, 90.0)

Hexagonal
---------

.. plot::
   :context: close-figs
   :include-source:

   draw_lattice(500.0, 500.0, 60.0)
