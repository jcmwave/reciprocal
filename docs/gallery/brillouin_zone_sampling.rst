Full Brillouin-zone sampling
============================

Oblique
-------

.. plot::
   :context: reset
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np
   from fractions import Fraction
   from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator

   from reciprocal import BoundaryGrid, Lattice, PointCounts, sample_brillouin_zone
   from reciprocal.canvas import Canvas, choose_color


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


   def draw_sampling(a, b, angle):
       lattice = Lattice.from_lengths_angle(a, b, angle).make_reciprocal()
       zone = lattice.brillouin_zone
       sampling = sample_brillouin_zone(
           zone,
           BoundaryGrid(PointCounts(5)),
       )
       tessellation_orders = np.array([
           (i, j) for i in range(-1, 2) for j in range(-1, 2)
       ])

       fig, ax = plt.subplots(figsize=(6, 5))
       canvas = Canvas(ax)
       canvas.plot_tessellation(
           lattice,
           orders=tessellation_orders,
           facecolor="#f5f7fa",
           edgecolor="#b4c0cc",
           linewidth=1.0,
           increase_bbox=False,
       )
       canvas.plot_bzone(
           zone,
           facecolor=(0.90, 0.25, 0.18, 0.10),
           edgecolor="#c7352b",
           linewidth=2.5,
       )
       unique_weights = np.unique(np.round(sampling.normalized_weights, decimals=14))
       for index, weight in enumerate(unique_weights):
           selected = np.isclose(sampling.normalized_weights, weight)
           fraction = Fraction(float(weight)).limit_denominator()
           canvas.plot_point_sampling(
               sampling.points[selected],
               color=choose_color(index, len(unique_weights)).ravel(),
               marker="o",
               label=rf"${fraction.numerator}/{fraction.denominator}$",
               edgecolor="white",
               linewidth=0.6,
               s=55,
               zorder=5,
           )
       ax.legend(
           title="Integration weight",
           loc="center left",
           bbox_to_anchor=(1.02, 0.5),
       )

       vertices = zone.vertices[:, :2]
       span = np.ptp(vertices, axis=0)
       margin = 0.08 * span
       ax.set_xlim(vertices[:, 0].min() - margin[0], vertices[:, 0].max() + margin[0])
       ax.set_ylim(vertices[:, 1].min() - margin[1], vertices[:, 1].max() + margin[1])
       ax.set_aspect("equal")
       scale_reciprocal_axes(ax, lattice)
       return fig


   draw_sampling(1000.0, 500.0, 75.0)

Rectangular
-----------

.. plot::
   :context: close-figs
   :include-source:

   draw_sampling(1000.0, 500.0, 90.0)

Centered rectangular
--------------------

.. plot::
   :context: close-figs
   :include-source:

   draw_sampling(500.0, 500.0, 70.0)

Square
------

.. plot::
   :context: close-figs
   :include-source:

   draw_sampling(500.0, 500.0, 90.0)

Hexagonal
---------

.. plot::
   :context: close-figs
   :include-source:

   draw_sampling(500.0, 500.0, 60.0)
