Real-space lattices with primitive cells
========================================

Oblique
-------

.. plot::
   :context: reset
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np

   from reciprocal import Lattice
   from reciprocal.canvas import Canvas


   def draw_lattice(a, b, angle):
       lattice = Lattice.from_lengths_angle(a, b, angle)
       viewport_orders = np.array([
           (i, j) for i in range(-2, 3) for j in range(-2, 3)
       ])
       plot_orders = np.array([
           (i, j) for i in range(-6, 7) for j in range(-6, 7)
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
       canvas.plot_unit_cell(
           lattice.primitive_cell,
           facecolor=(0.20, 0.55, 0.28, 0.18),
           edgecolor="#27823d",
           linewidth=2.5,
       )

       for label, vector in zip((r"$a_1$", r"$a_2$"), lattice.vectors.basis):
           ax.annotate(
               label,
               xy=vector[:2],
               xytext=(5, 5),
               textcoords="offset points",
               color="darkgreen",
           )

       positions = viewport_orders @ lattice.vectors.basis[:, :2]
       margin = 0.45 * max(lattice.vectors.lengths)
       ax.set_xlim(positions[:, 0].min() - margin, positions[:, 0].max() + margin)
       ax.set_ylim(positions[:, 1].min() - margin, positions[:, 1].max() + margin)
       ax.set(xlabel="x (nm)", ylabel="y (nm)")
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
