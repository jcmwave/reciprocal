import numpy as np
import pytest

from reciprocal import KSpace, Lattice
from reciprocal.spectrum import (
    AxialVectorRepresentation,
    FieldSamples,
    FourierOrderTranslation,
    IdentityTranslation,
    PointCounts,
    PolarVectorRepresentation,
    PropagationDisk,
    ScalarRepresentation,
    make_periodic_plan,
)


def _space_and_lattice():
    space = KSpace.propagating(2.0 * np.pi, 1.0)
    direct = Lattice.from_lengths_angle(2.0 * np.pi, 2.0 * np.pi, 90.0)
    return space, direct


def test_combined_expansion_has_unique_targets_and_provenance():
    space, direct = _space_and_lattice()
    plan = space.with_periodic_structure(direct).plan(
        source="ibz",
        target="propagating_spectrum",
        constraint=PointCounts(5),
    )

    rounded = np.round(plan.target.points, decimals=10)
    assert len(np.unique(rounded, axis=0)) == len(rounded)
    assert len(plan.expansion.primary_generators) == len(plan.target.points)
    assert all(plan.expansion.equivalent_generators)
    assert plan.target.integrate(np.ones(len(plan.target.points))) == pytest.approx(np.pi)


def test_scalar_expansion_reuses_map_and_vector_translation_is_explicit():
    space, direct = _space_and_lattice()
    plan = space.with_periodic_structure(direct).plan(
        source="ibz", constraint=PointCounts(4)
    )
    scalar = FieldSamples(
        plan.representatives,
        np.ones(len(plan.representatives.points)),
        ScalarRepresentation(),
    )
    expanded = plan.expansion.expand(scalar)
    np.testing.assert_allclose(expanded.values, 1.0)

    vectors = FieldSamples(
        plan.representatives,
        np.ones((len(plan.representatives.points), 3)),
        PolarVectorRepresentation(),
    )
    with pytest.raises(ValueError, match="TranslationAction"):
        plan.expansion.expand(vectors)

    declared = FieldSamples(
        plan.representatives,
        np.ones((len(plan.representatives.points), 3)),
        PolarVectorRepresentation(IdentityTranslation()),
    )
    assert plan.expansion.expand(declared, verify=False).values.shape == (
        len(plan.target.points),
        3,
    )


def test_ibz_and_expanded_bz_integrals_use_the_same_orbit_quadrature():
    space, direct = _space_and_lattice()
    plan = space.with_periodic_structure(direct).plan(
        source="ibz",
        target="bz",
        constraint=PointCounts(7),
    )
    values = np.sum(plan.representatives.points**2, axis=1)
    representatives = FieldSamples(
        plan.representatives,
        values,
        ScalarRepresentation(),
    )
    expanded = plan.expand(representatives)

    assert expanded.integrate() == pytest.approx(representatives.integrate(), rel=1e-12)


def test_polar_and_axial_vectors_differ_under_reflection():
    _, direct = _space_and_lattice()
    reflection = next(
        operation
        for operation in direct.make_reciprocal().brillouin_zone.point_group
        if operation.determinant == -1
    )
    value = np.array([1.0, 2.0, 3.0])
    source = np.array([0.1, 0.2])
    target = reflection.apply(source)[:2]
    polar = PolarVectorRepresentation(IdentityTranslation()).transform(
        value, reflection, source, target, (0, 0)
    )
    axial = AxialVectorRepresentation(IdentityTranslation()).transform(
        value, reflection, source, target, (0, 0)
    )
    np.testing.assert_allclose(axial, -polar)


def test_fourier_order_translation_rotates_target_order_back_to_solver_order():
    _, direct = _space_and_lattice()
    reciprocal = direct.make_reciprocal()
    rotation = next(
        operation
        for operation in reciprocal.brillouin_zone.point_group
        if operation.determinant == 1
        and not np.array_equal(operation.fractional, np.eye(2, dtype=int))
    )
    basis = reciprocal.vectors.basis[:, :2]
    orders = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    coefficients = np.array([10.0, 20.0, 30.0, 40.0])
    target_order = np.rint(
        (orders[0] @ basis) @ rotation.cartesian[:2, :2].T @ np.linalg.inv(basis)
    ).astype(int)
    representation = ScalarRepresentation(FourierOrderTranslation(orders, basis))

    transformed = representation.transform(
        coefficients,
        rotation,
        np.zeros(2),
        target_order @ basis,
        tuple(target_order),
    )

    assert transformed == 10.0


def test_periodic_api_rejects_a_reciprocal_input_lattice():
    space, direct = _space_and_lattice()
    with pytest.raises(ValueError, match="direct-space"):
        space.with_periodic_structure(direct.make_reciprocal())


def test_translation_bounds_are_complete_for_a_highly_skewed_basis():
    direct = Lattice.from_lengths_angle(2.0 * np.pi, 2.0 * np.pi, 20.0)
    domain = PropagationDisk(3.0)
    plan = make_periodic_plan(
        direct,
        domain,
        source="bz",
        constraint=PointCounts(3),
    )
    basis = plan.reciprocal_lattice.vectors.basis[:, :2]
    brute_force = []
    for point in plan.representatives.points:
        for first in range(-12, 13):
            for second in range(-12, 13):
                candidate = point + np.array([first, second]) @ basis
                if domain.contains(candidate)[0]:
                    brute_force.append(candidate)

    expected = {tuple(row) for row in np.round(brute_force, decimals=9)}
    actual = {tuple(row) for row in np.round(plan.target.points, decimals=9)}
    assert actual == expected


def test_extended_space_distinguishes_propagating_and_evanescent_targets():
    space = KSpace.extended(2.0 * np.pi, 1.0, 2.0)
    direct = Lattice.from_lengths_angle(2.0 * np.pi, 2.0 * np.pi, 90.0)
    periodic = space.with_periodic_structure(direct)

    propagating = periodic.plan(
        source="ibz", target="propagating_spectrum", constraint=PointCounts(3)
    )
    extended = periodic.plan(
        source="ibz", target="extended_spectrum", constraint=PointCounts(3)
    )

    assert np.max(np.linalg.norm(propagating.target.points, axis=1)) <= 1.0 + 1e-12
    assert np.any(np.linalg.norm(extended.target.points, axis=1) > 1.0)
    assert np.any(np.abs(np.imag(extended.target.kz)) > 0.0)
