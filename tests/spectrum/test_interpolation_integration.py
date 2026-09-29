import numpy as np

from reciprocal import KSpace, Lattice
from reciprocal.spectrum import (
    CartesianGrid,
    FieldSamples,
    NonPeriodicSampler,
    PointCounts,
    PropagationDisk,
    ScalarRepresentation,
    interpolate,
)


def test_linear_interpolation_reproduces_affine_complex_values():
    sampling = NonPeriodicSampler().sample(PropagationDisk(1.0), CartesianGrid((12, 12)))
    values = (2.0 * sampling.points[:, 0] - sampling.points[:, 1] + 1.0) * (1.0 + 2.0j)
    field = FieldSamples(sampling, values, ScalarRepresentation())
    targets = np.array([[0.0, 0.0], [0.2, -0.1], [-0.3, 0.2]])

    result = interpolate(field, targets, method="linear")

    expected = (2.0 * targets[:, 0] - targets[:, 1] + 1.0) * (1.0 + 2.0j)
    assert np.all(result.valid)
    np.testing.assert_allclose(result.values, expected, rtol=1e-12, atol=1e-12)


def test_interpolation_masks_queries_outside_convex_hull():
    sampling = NonPeriodicSampler().sample(PropagationDisk(1.0), CartesianGrid((8, 8)))
    field = FieldSamples(
        sampling,
        np.ones(len(sampling.points)),
        ScalarRepresentation(),
    )
    result = interpolate(field, np.array([[0.0, 0.0], [2.0, 0.0]]))
    assert result.valid.tolist() == [True, False]


def test_periodic_interpolation_is_continuous_across_bz_edges():
    space = KSpace.propagating(2.0 * np.pi, 1.0)
    direct = Lattice.from_lengths_angle(2.0 * np.pi, 2.0 * np.pi, 90.0)
    plan = space.with_periodic_structure(direct).plan(
        source="bz", target="bz", constraint=PointCounts(11)
    )
    basis = plan.reciprocal_lattice.vectors.basis[:, :2]
    fractional = plan.representatives.points @ np.linalg.inv(basis)
    values = np.cos(2.0 * np.pi * fractional[:, 0])
    field = FieldSamples(plan.representatives, values, ScalarRepresentation())
    query = np.array([[-0.499, 0.1], [0.501, 0.1]])

    result = field.interpolate(query, reciprocal_basis=basis)

    assert np.all(result.valid)
    np.testing.assert_allclose(result.values[0], result.values[1], atol=1e-12)
