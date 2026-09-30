Quickstart
==========

Lattice geometry
----------------

Create a direct lattice from two vector lengths and their angle, then derive
its reciprocal lattice and first Brillouin zone:

.. code-block:: python

   from reciprocal import Lattice

   direct = Lattice.from_lengths_angle(
       length1=500e-9,
       length2=500e-9,
       angle=90.0,
   )
   reciprocal_lattice = direct.make_reciprocal()
   zone = reciprocal_lattice.brillouin_zone

   print(zone.area)
   print(zone.special_points)

``make_reciprocal()`` is defined for direct-space lattices. Reciprocal vectors
satisfy :math:`a_i \cdot b_j = 2\pi\delta_{ij}`.

Non-periodic sampling
---------------------

A :class:`~reciprocal.KSpace` describes the physical transverse-wavevector
domain. This example samples a microscopy pupil on a polar grid:

.. code-block:: python

   from reciprocal import KSpace
   from reciprocal.spectrum import PolarGrid

   space = KSpace.propagating(wavelength=1.0e-6, refractive_index=1.0)
   pupil = space.sample_pupil(
       PolarGrid(radial=32, azimuthal=128),
       numerical_aperture=0.8,
   )

   assert pupil.points.shape[1] == 2
   assert abs(pupil.normalized_weights.sum() - 1.0) < 1e-12

All numerical samplers return the common
:class:`~reciprocal.SamplingResult` contract. ``normalized_weights`` sum to
one; ``physical_weights`` sum to the area of the sampled domain. When a
wavevector magnitude is configured, the sampling also retains longitudinal
components through ``pupil.kz``.

Periodic sampling
-----------------

Pair a physical spectrum with a direct lattice to build representatives in
the irreducible Brillouin zone (IBZ) and expand them onto a target domain:

.. code-block:: python

   from reciprocal.spectrum import MaxSpacing

   plan = space.with_periodic_structure(direct).plan(
       source="ibz",
       target="propagating_spectrum",
       constraint=MaxSpacing(2.0e5),
   )

   representatives = plan.representatives
   complete_sampling = plan.target

The plan records the symmetry operation and reciprocal-lattice translation
that relate each target point to its representative. See the
:mod:`reciprocal.spectrum` API for field representations, expansion,
interpolation, and integration.

Monkhorst--Pack meshes
----------------------

Create a conventional uniform mesh over the Brillouin zone, or reduce it by
the subgroup of point operations that preserves the configured shift:

.. code-block:: python

   from reciprocal import MonkhorstPackGrid, reduce_brillouin_zone

   mesh = MonkhorstPackGrid((8, 8), shift=(0.5, 0.5))
   reduction = reduce_brillouin_zone(zone, mesh)

   full_mesh = reduction.full
   irreducible_mesh = reduction.irreducible
   assert reduction.degeneracies.sum() == 64
   assert abs(irreducible_mesh.normalized_weights.sum() - 1.0) < 1e-12

Use ``centering="gamma"`` for a Gamma-centered grid. Shifts are measured in
mesh steps, so ``0.5`` means a half-step shift rather than half a reciprocal
basis vector.

High-symmetry paths
-------------------

Conventional paths are selected from the detected Bravais family:

.. code-block:: python

   from reciprocal import make_high_symmetry_path

   path = make_high_symmetry_path(zone, max_spacing=2.0e5)
   print(path.labels)
   solver_wavevectors = path.to_kvectors(
       wavelength=1.0e-6,
       refractive_index=1.0,
   )

The result contains Cartesian and fractional coordinates, cumulative distance,
node indices, labels, and segment metadata. Pass an explicit label sequence,
such as ``("Γ", "X", "M", "Γ")``, to construct a custom path.
