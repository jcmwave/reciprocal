"""Exercise non-periodic and periodic reciprocal-spectrum workflows."""

from __future__ import annotations

import numpy as np

from reciprocal import KSpace, Lattice, LatticeVectors
from reciprocal.spectrum import (
    FieldSamples,
    IdentityTranslation,
    PointCounts,
    PolarGrid,
    PolarVectorRepresentation,
    ScalarRepresentation,
    interpolate,
)


def main() -> None:
    wavelength = 1.0e-6
    space = KSpace.propagating(wavelength, refractive_index=1.0)

    pupil = space.sample_pupil(
        PolarGrid(radial=8, azimuthal=32),
        numerical_aperture=0.8,
    )
    pupil_values = np.exp(-np.sum(pupil.points**2, axis=1) / space.k0**2)
    pupil_field = FieldSamples(pupil, pupil_values, ScalarRepresentation())
    print("pupil integral:", pupil_field.integrate())

    direct = Lattice(
        LatticeVectors.from_lengths_angle(500.0e-9, 500.0e-9, 90.0)
    )
    plan = space.with_periodic_structure(direct).plan(
        source="ibz",
        target="propagating_spectrum",
        constraint=PointCounts(9),
    )
    electric_field = np.ones((len(plan.representatives.points), 3), dtype=complex)
    representatives = FieldSamples(
        plan.representatives,
        electric_field,
        PolarVectorRepresentation(IdentityTranslation()),
    )
    expanded = plan.expansion.expand(representatives, verify=False)
    print("representatives / targets:", len(plan.representatives.points), len(expanded.values))

    query = np.array([[0.0, 0.0], [0.25 * space.k0, 0.0]])
    interpolated = interpolate(expanded, query, method="nearest", extrapolation="nearest")
    print("interpolated electric field:", interpolated.values)


if __name__ == "__main__":
    main()
