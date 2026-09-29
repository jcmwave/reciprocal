Plotting with Canvas
====================

:class:`reciprocal.canvas.Canvas` is a thin adapter around a caller-owned
Matplotlib :class:`~matplotlib.axes.Axes`. It adds artists to the axes, updates
useful bounds, and returns the created artists. It does not call ``show()`` or
save files; the caller controls figure layout, colorbars, display, and output.

Install ``reciprocal[plot]`` before running these examples.

Lattice and Brillouin-zone geometry
-----------------------------------

The same canvas can combine reciprocal-lattice points, translated cells, the
Brillouin zone, its irreducible domain, and named special points.

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np

   from reciprocal import Lattice
   from reciprocal.canvas import Canvas

   direct = Lattice.from_lengths_angle(1.0, 1.0, 60.0)
   reciprocal_lattice = direct.make_reciprocal()
   orders = np.array([
       [-1, -1], [-1, 0], [-1, 1],
       [0, -1], [0, 0], [0, 1],
       [1, -1], [1, 0], [1, 1],
   ])

   fig, ax = plt.subplots(figsize=(6, 5))
   canvas = Canvas(ax)
   canvas.plot_tessellation(reciprocal_lattice, orders=orders)
   canvas.plot_lattice(reciprocal_lattice, orders=orders)
   canvas.plot_bzone(reciprocal_lattice)
   canvas.plot_ibzone(reciprocal_lattice)
   canvas.plot_special_points(reciprocal_lattice)
   ax.set(xlabel=r"$k_x$", ylabel=r"$k_y$", title="Hexagonal reciprocal lattice")

Weighted pupil sampling
-----------------------

Sampling containers carry their quadrature weights, so no separate weight
array is needed. Pass ``physical=True`` to color by physical rather than
normalized weights.

.. plot::
   :include-source:

   import matplotlib.pyplot as plt

   from reciprocal import KSpace
   from reciprocal.canvas import Canvas
   from reciprocal.spectrum import PolarGrid

   space = KSpace.propagating(wavelength=1.0, refractive_index=1.0)
   pupil = space.sample_pupil(
       PolarGrid(radial=12, azimuthal=48),
       numerical_aperture=0.8,
   )

   fig, ax = plt.subplots(figsize=(6, 5))
   canvas = Canvas(ax)
   canvas.plot_spectrum(space, edgecolor="0.25")
   points = canvas.plot_point_sampling_weighted(
       pupil,
       cmap="viridis",
       s=18,
   )
   fig.colorbar(points, ax=ax, label="normalized quadrature weight")
   ax.set(xlabel=r"$k_x$", ylabel=r"$k_y$", title="Polar pupil sampling")

Periodic expansion and Bloch families
-------------------------------------

A periodic sampling plan can be passed directly to the geometry and family
plotters. Each color below identifies target points expanded from one IBZ
representative.

.. plot::
   :include-source:

   import matplotlib.pyplot as plt

   from reciprocal import KSpace, Lattice
   from reciprocal.canvas import Canvas
   from reciprocal.spectrum import PointCounts

   space = KSpace.propagating(wavelength=1.0, refractive_index=1.0)
   direct = Lattice.from_lengths_angle(1.0, 1.0, 90.0)
   plan = space.with_periodic_structure(direct).plan(
       source="ibz",
       target="propagating_spectrum",
       constraint=PointCounts(5),
   )

   fig, ax = plt.subplots(figsize=(6, 5))
   canvas = Canvas(ax)
   canvas.plot_spectrum(plan, edgecolor="0.4")
   canvas.plot_bzone(plan, fill=False, edgecolor="tab:red")
   canvas.plot_ibzone(plan, fill=False, edgecolor="tab:green")
   canvas.plot_bloch_families(plan)
   ax.set(xlabel=r"$k_x$", ylabel=r"$k_y$", title="Expanded Bloch families")

Scalar fields and interpolation
-------------------------------

``plot_field`` reads coordinates and validity flags from a
:class:`~reciprocal.spectrum.FieldSamples` object. ``plot_interpolation``
renders scalar values on the Delaunay triangulation of scattered points.

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np

   from reciprocal import KSpace
   from reciprocal.canvas import Canvas
   from reciprocal.spectrum import FieldSamples, PolarGrid, ScalarRepresentation

   space = KSpace.propagating(wavelength=1.0)
   sampling = space.sample_pupil(
       PolarGrid(radial=10, azimuthal=36),
       numerical_aperture=0.9,
   )
   radius = np.linalg.norm(sampling.points, axis=1)
   values = np.cos(2.0 * radius) * np.exp(-0.04 * radius**2)
   field = FieldSamples(sampling, values, ScalarRepresentation())

   fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
   field_artist = Canvas(left).plot_field(field, cmap="magma")
   interpolation_artist = Canvas(right).plot_interpolation(
       sampling,
       values,
       cmap="magma",
       edgecolors="none",
   )
   fig.colorbar(field_artist, ax=left, label="field value")
   fig.colorbar(interpolation_artist, ax=right, label="field value")
   left.set(title="Sampled field", xlabel=r"$k_x$", ylabel=r"$k_y$")
   right.set(title="Triangulated field", xlabel=r"$k_x$", ylabel=r"$k_y$")

High-symmetry paths and band axes
---------------------------------

``Canvas`` draws a path over its Brillouin zone. The same path provides the
cumulative coordinate and labelled ticks for a band plot.

.. plot::
   :include-source:

   import matplotlib.pyplot as plt
   import numpy as np

   from reciprocal import Lattice, make_high_symmetry_path, set_band_path_axis
   from reciprocal.canvas import Canvas

   direct = Lattice.from_lengths_angle(1.0, 1.0, 60.0)
   zone = direct.make_reciprocal().brillouin_zone
   path = make_high_symmetry_path(zone, points_per_segment=31)

   fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
   geometry = Canvas(left)
   geometry.plot_bzone(zone, fill=False)
   geometry.plot_high_symmetry_path(path)
   left.set(title="Γ–M–K–Γ path", xlabel=r"$k_x$", ylabel=r"$k_y$")

   band = 0.15 * path.distance**2 + np.sin(1.5 * path.distance)
   right.plot(path.distance, band)
   set_band_path_axis(right, path)
   right.set(xlabel="wave-vector path", ylabel="example band", title="Band-axis metadata")

Saving a figure
---------------

Use the normal Matplotlib lifecycle in scripts and batch jobs:

.. code-block:: python

   fig.savefig("sampling.png", dpi=150)
   plt.close(fig)

For headless environments, set ``MPLBACKEND=Agg``. The documentation build
does this automatically.
