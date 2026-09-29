Sampling migration
==================

The common contract
-------------------

Cell, Brillouin-zone, non-periodic spectrum, periodic spectrum, and
Monkhorst--Pack samplers now use :class:`reciprocal.SamplingResult`.
``reciprocal.cells.SamplingResult`` and ``reciprocal.spectrum.KSampling`` are
aliases to that same class during the 1.x compatibility period.

The canonical attributes are:

``points``
   Transverse Cartesian coordinates with shape ``(N, 2)``.

``normalized_weights``
   Non-negative quadrature weights that sum to one. ``weights`` remains an
   alias.

``physical_weights``
   Weights multiplied by ``integration_element``.

``domain`` and ``integration_element``
   The numerical integration domain and its area.

``point_ids``
   Stable integer identities used to associate solver data with samples.

``metadata``
   Immutable source-specific information. Periodic expansion and mesh
   reduction mappings remain typed companion objects rather than being packed
   into this mapping.

Constructing results
--------------------

Code that constructed the former cell result must now supply its domain:

.. code-block:: python

   from reciprocal import SamplingResult

   result = SamplingResult(points, normalized_weights, domain.area, domain)

Use ``result.integrate(values)`` rather than multiplying and summing weights
manually. It preserves all trailing value dimensions.

Legacy tuple samplers
---------------------

``KSpace.regular_sampler`` and ``KSpace.periodic_sampler`` retain their legacy
tuple contracts in the 1.x series. New code should use ``sample_pupil()``,
``sample_domain()``, ``with_periodic_structure().plan()``, cell/BZ samplers,
or Monkhorst--Pack functions, all of which return the common result directly.

Mesh and path additions
-----------------------

Use :class:`reciprocal.MonkhorstPackGrid` with
:func:`reciprocal.sample_monkhorst_pack` for a full grid or
:func:`reciprocal.reduce_monkhorst_pack` when the full-to-irreducible mapping
and degeneracies are needed.

Band paths are not area quadratures and intentionally use
:class:`reciprocal.HighSymmetryPath`. It carries ordered coordinates,
cumulative distance, node indices, labels, and segment boundaries without
inventing meaningless two-dimensional integration weights.
