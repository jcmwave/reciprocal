Wood anomalies in the propagating spectrum
==========================================

Oblique
-------

.. plot::
   :context: reset
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np
   from matplotlib.lines import Line2D
   from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator

   from reciprocal import KSpace, Lattice
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


   def draw_anomalies(a, b, angle):
       # Direct-space lengths are in meters (500 nm or 1000 nm below).
       direct_lattice = Lattice.from_lengths_angle(a, b, angle)
       lattice = direct_lattice.make_reciprocal()
       zone = lattice.brillouin_zone
       reciprocal_scale = min(lattice.vectors.lengths)
       k0 = 2.2 * reciprocal_scale
       kspace = KSpace.propagating(wavelength=2 * np.pi / k0)
       plot_orders = np.array([
           (i, j) for i in range(-5, 6) for j in range(-5, 6)
       ])

       fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
       canvas = Canvas(ax)
       canvas.plot_tessellation(
           lattice,
           orders=plot_orders,
           facecolor="#f7f9fb",
           edgecolor="#c4ced8",
           linewidth=0.8,
           increase_bbox=False,
       )
       canvas.plot_lattice(
           lattice,
           orders=plot_orders,
           facecolor="#172b4d",
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
           kspace, fill=False, edgecolor="black", linewidth=2.0, zorder=5,
       )

       anomalies = kspace.with_periodic_structure(direct_lattice).wood_anomalies(
           shells=(1, 2, 3, 4),
       )
       canvas.plot_wood_anomalies(anomalies, linewidth=1.2, zorder=4)
       legend = []
       for index, order in enumerate(anomalies.circles_by_shell):
           color = choose_color(index + 1, len(anomalies.circles_by_shell) + 1).ravel()
           legend.append(Line2D(
               [], [], color=color, linewidth=1.5,
               label=f"Order {order}",
           ))
       ax.legend(
           handles=legend,
           title="Wood anomalies",
           loc="center left",
           bbox_to_anchor=(1.02, 0.5),
       )

       margin = 0.08 * k0
       ax.set_xlim(-k0 - margin, k0 + margin)
       ax.set_ylim(-k0 - margin, k0 + margin)
       ax.set_aspect("equal")
       scale_reciprocal_axes(ax, lattice)
       return fig


   draw_anomalies(1000e-9, 500e-9, 75.0)

Rectangular
-----------

.. plot::
   :context: close-figs
   :include-source:

   draw_anomalies(1000e-9, 500e-9, 90.0)

Centered rectangular
--------------------

.. plot::
   :context: close-figs
   :include-source:

   draw_anomalies(500e-9, 500e-9, 70.0)

Square
------

.. plot::
   :context: close-figs
   :include-source:

   draw_anomalies(500e-9, 500e-9, 90.0)

Hexagonal
---------

.. plot::
   :context: close-figs
   :include-source:

   draw_anomalies(500e-9, 500e-9, 60.0)
