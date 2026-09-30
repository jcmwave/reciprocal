Choosing a Brillouin-zone sampler
=================================

The common zone interface separates the grid construction scheme, symmetry
reduction, and representative placement. Every path returns
:class:`~reciprocal.SamplingResult`, whose weights integrate over the full
Brillouin zone.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Grid
     - Intended use
   * - :class:`~reciprocal.BoundaryGrid`
     - Endpoint-inclusive polygon quadrature, explicit BZ boundaries, and
       canonical IBZ coordinates.
   * - :class:`~reciprocal.MonkhorstPackGrid`
     - Periodic fractional meshes, configurable centering and shifts, and
       exact discrete orbit reduction.

Full and irreducible sampling
-----------------------------

Pass a typed grid to :func:`~reciprocal.sample_brillouin_zone`. ``region``
controls whether the result contains the full grid or one representative per
symmetry orbit:

.. code-block:: python

   from reciprocal import (
       BoundaryGrid,
       Lattice,
       PointCounts,
       sample_brillouin_zone,
   )

   direct = Lattice.from_lengths_angle(500.0, 500.0, 90.0)
   zone = direct.make_reciprocal().brillouin_zone
   grid = BoundaryGrid(PointCounts(15))

   full = sample_brillouin_zone(zone, grid, region="bz")
   irreducible = sample_brillouin_zone(zone, grid, region="irreducible")

   assert abs(full.normalized_weights.sum() - 1.0) < 1e-12
   assert abs(irreducible.normalized_weights.sum() - 1.0) < 1e-12

``region="irreducible"`` means that symmetry representatives and orbit
weights are used to integrate over the full BZ. It does not mean that the
integration domain has changed to the geometric area of the IBZ.

Boundary grids include periodically shared BZ edges and vertices. Their full
weights include the appropriate fractional ownership. Reduction sums those
full-grid weights for every symmetry orbit.

Computational and canonical representatives
--------------------------------------------

Monkhorst--Pack reduction defaults to computational representatives. Each is
an actual full-mesh row, so its point ID and representative index can address
solver arrays directly:

.. code-block:: python

   from reciprocal import MonkhorstPackGrid, reduce_brillouin_zone

   mesh = MonkhorstPackGrid((15, 15))
   reduction = reduce_brillouin_zone(zone, mesh)

   computational = reduction.reduced
   assert (
       computational.points
       == reduction.full.points[reduction.representative_indices]
   ).all()

The smallest-index representatives need not occupy a contiguous polygon or
lie inside the displayed canonical IBZ. Request canonical placement when
coordinates in that chamber are required:

.. code-block:: python

   canonical = reduce_brillouin_zone(
       zone,
       mesh,
       placement="canonical_ibz",
       require_full_symmetry=True,
   )

   assert zone.irreducible_domain.contains(canonical.reduced.points).all()

``canonical.placement`` records the original full-mesh indices, source and
placed coordinates, point operations, and reciprocal-lattice translations.
Weights, degeneracies, point IDs, and the full-to-reduced mapping are
unchanged. Field values are not transformed implicitly.

Grid symmetry
-------------

Reduction uses only point operations that preserve the configured grid.
:func:`~reciprocal.grid_symmetry` reports that subgroup before sampling:

.. code-block:: python

   from reciprocal import GridCentering, grid_symmetry

   hexagonal = Lattice.from_lengths_angle(500.0, 500.0, 60.0)
   hexagonal_zone = hexagonal.make_reciprocal().brillouin_zone

   even_mp = MonkhorstPackGrid((12, 12))
   odd_mp = MonkhorstPackGrid((15, 15))
   even_gamma = MonkhorstPackGrid((12, 12), GridCentering.GAMMA)

   assert not grid_symmetry(hexagonal_zone, even_mp).preserves_full_group
   assert grid_symmetry(hexagonal_zone, odd_mp).preserves_full_group
   assert grid_symmetry(hexagonal_zone, even_gamma).preserves_full_group

An even conventional hexagonal MP grid has a half-step offset that is not
preserved by every 60-degree rotation. It remains a valid integration grid,
but its irreducible mesh is reduced only by the preserving subgroup. Use
``require_full_symmetry=True`` to reject such a grid instead of accepting the
subgroup reduction:

.. code-block:: python

   sample_brillouin_zone(
       hexagonal_zone,
       even_mp,
       region="irreducible",
       require_full_symmetry=True,
   )

For full hexagonal symmetry, use equal odd conventional counts or equal
Gamma-centered counts. Unequal dimensions and custom shifts can also reduce
the preserving subgroup; the library never changes those settings silently.

Reduction mappings
------------------

Use :func:`~reciprocal.reduce_brillouin_zone` when the complete mapping is
needed. Its :class:`~reciprocal.ZoneReduction` result contains:

``full`` and ``reduced``
   Full and symmetry-reduced :class:`~reciprocal.SamplingResult` values.

``full_to_reduced``
   The reduced-row index associated with every full-grid row.

``representative_indices``
   Full-grid indices of the computational representatives.

``degeneracies``
   Number of full-grid rows in each orbit.

``operations``
   The subgroup actually used for reduction.

The compatibility names ``irreducible`` and ``full_to_irreducible`` remain
available during the 1.x series.
