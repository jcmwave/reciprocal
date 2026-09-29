"""Immutable two-dimensional lattice bases and classified lattices."""

from __future__ import annotations

import warnings
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reciprocal.bravais import BravaisLattice
from reciprocal.cells.construction import (
    make_conventional_cell,
    make_primitive_cell,
    reduce_basis,
)
from reciprocal.numerics import DEFAULT_TOLERANCES, Tolerances

if TYPE_CHECKING:
    from reciprocal.unit_cell import UnitCell as LegacyUnitCell

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class LatticeType(str, Enum):
    """Whether a lattice basis is expressed in direct or reciprocal space."""

    REAL_SPACE = "real_space"
    RECIPROCAL = "reciprocal"


def _validate_tolerances(tolerances: Tolerances | None) -> Tolerances:
    if tolerances is None:
        return DEFAULT_TOLERANCES
    if not isinstance(tolerances, Tolerances):
        raise TypeError("tolerances must be a Tolerances instance")
    return tolerances


def unit_vector(vector: ArrayLike) -> FloatArray:
    """Return a finite, nonzero vector normalized to unit length."""

    array = np.asarray(vector, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("vector must be a finite one-dimensional array")
    scale = float(np.max(np.abs(array), initial=0.0))
    if scale == 0.0:
        raise ValueError("vector must be non-zero")
    scaled = array / scale
    return scaled / np.linalg.norm(scaled)


def angle_between(first: ArrayLike, second: ArrayLike) -> float:
    """Return the angle between two vectors in radians.

    ``atan2`` remains accurate for angles close to zero and pi, where an
    ``arccos`` of normalized vectors loses precision.
    """

    first_array = unit_vector(first)
    second_array = unit_vector(second)
    if first_array.shape != second_array.shape:
        raise ValueError("vectors must have matching dimensions")
    dot = float(np.clip(np.dot(first_array, second_array), -1.0, 1.0))
    if first_array.size == 2:
        perpendicular = abs(
            float(first_array[0] * second_array[1] - first_array[1] * second_array[0])
        )
    elif first_array.size == 3:
        perpendicular = float(np.linalg.norm(np.cross(first_array, second_array)))
    else:
        perpendicular = float(np.sqrt(max(0.0, 1.0 - dot * dot)))
    return float(np.arctan2(perpendicular, dot))


def make_vectors(length1: float, length2: float, angle: float) -> tuple[FloatArray, FloatArray]:
    """Return planar vectors with the first vector oriented along positive x."""

    values = (length1, length2, angle)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("lattice lengths and angle must be finite")
    if length1 <= 0.0 or length2 <= 0.0:
        raise ValueError("lattice lengths must be positive")
    if not 0.0 < angle < 180.0:
        raise ValueError("lattice angle must be strictly between 0 and 180 degrees")
    angle_radians = np.radians(angle)
    direction = np.array([np.cos(angle_radians), np.sin(angle_radians)])
    direction[np.abs(direction) <= np.finfo(float).eps * 8] = 0.0
    first = length1 * np.array([1.0, 0.0, 0.0])
    second = length2 * np.array([direction[0], direction[1], 0.0])
    return first, second


class LatticeVectors:
    """An immutable pair of independent planar lattice translations.

    The canonical stored representation is a read-only ``(2, 3)`` row-basis
    array. Lengths, angle, area, and the Gram matrix are derived properties, so
    they cannot become inconsistent with the vectors.
    """

    __slots__ = ("_basis",)

    def __init__(self, vector1: ArrayLike, vector2: ArrayLike) -> None:
        first = np.asarray(vector1, dtype=float)
        second = np.asarray(vector2, dtype=float)
        if first.shape != second.shape or first.ndim != 1 or first.size not in (2, 3):
            raise ValueError("lattice vectors must have matching shape (2,) or (3,)")
        if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
            raise ValueError("lattice vectors must contain only finite values")
        if first.size == 2:
            first = np.append(first, 0.0)
            second = np.append(second, 0.0)
        elif first[2] != 0.0 or second[2] != 0.0:
            raise ValueError("lattice vectors must lie in the x-y plane")

        first_length = float(np.linalg.norm(first))
        second_length = float(np.linalg.norm(second))
        if not np.isfinite(first_length) or not np.isfinite(second_length):
            raise ValueError("lattice-vector lengths must be representable as finite floats")
        if first_length == 0.0 or second_length == 0.0:
            raise ValueError("lattice vectors must be non-zero at floating-point precision")
        signed_area = float(first[0] * second[1] - first[1] * second[0])
        if not np.isfinite(signed_area):
            raise ValueError("lattice area must be representable as a finite float")
        if abs(signed_area) / (first_length * second_length) <= DEFAULT_TOLERANCES.degeneracy:
            raise ValueError("lattice vectors must be linearly independent")

        basis_values = np.vstack((first, second)).astype(float, copy=False)
        # Back the canonical array with immutable bytes. A simple write-protected
        # owning ndarray can otherwise be made writable again with ``setflags``.
        basis = np.frombuffer(basis_values.tobytes(), dtype=float).reshape(2, 3)
        object.__setattr__(self, "_basis", basis)

    def __setattr__(self, name: str, value: object) -> None:
        if hasattr(self, "_basis"):
            raise AttributeError("LatticeVectors is immutable")
        object.__setattr__(self, name, value)

    @classmethod
    def from_vectors(cls, vector1: ArrayLike, vector2: ArrayLike) -> LatticeVectors:
        """Construct a basis from two explicit translations."""

        return cls(vector1, vector2)

    @classmethod
    def from_lengths_angle(cls, length1: float, length2: float, angle: float) -> LatticeVectors:
        """Construct a basis from two lengths and their angle in degrees."""

        first, second = make_vectors(length1, length2, angle)
        return cls(first, second)

    @property
    def basis(self) -> FloatArray:
        view = self._basis.view()
        view.setflags(write=False)
        return view

    @property
    def vec1(self) -> FloatArray:
        view = self._basis[0].view()
        view.setflags(write=False)
        return view

    @property
    def vec2(self) -> FloatArray:
        view = self._basis[1].view()
        view.setflags(write=False)
        return view

    @property
    def length1(self) -> float:
        return float(np.linalg.norm(self.vec1))

    @property
    def length2(self) -> float:
        return float(np.linalg.norm(self.vec2))

    @property
    def lengths(self) -> FloatArray:
        values = np.linalg.norm(self._basis[:, :2], axis=1)
        values.setflags(write=False)
        return values

    @property
    def angle(self) -> float:
        return float(np.degrees(angle_between(self.vec1, self.vec2)))

    @property
    def signed_area(self) -> float:
        return float(self.vec1[0] * self.vec2[1] - self.vec1[1] * self.vec2[0])

    @property
    def area(self) -> float:
        return abs(self.signed_area)

    @property
    def gram_matrix(self) -> FloatArray:
        gram = self._basis[:, :2] @ self._basis[:, :2].T
        gram.setflags(write=False)
        return gram

    def reciprocal_vectors(self) -> LatticeVectors:
        """Return the reciprocal basis satisfying ``a_i dot b_j = 2*pi delta_ij``."""

        direct_columns = self._basis[:, :2].T
        reciprocal_columns = np.linalg.solve(direct_columns.T, 2.0 * np.pi * np.eye(2))
        return type(self)(
            np.append(reciprocal_columns[:, 0], 0.0),
            np.append(reciprocal_columns[:, 1], 0.0),
        )

    def get_shortest_vectors(self) -> tuple[FloatArray, FloatArray]:
        """Return a deterministic Gauss-reduced basis of successive minima."""

        reduced = reduce_basis(self)
        return reduced[0].copy(), reduced[1].copy()

    def __repr__(self) -> str:
        return f"LatticeVectors({self.vec1!r}, {self.vec2!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LatticeVectors) and np.array_equal(self._basis, other._basis)

    def __hash__(self) -> int:
        return hash(self._basis.tobytes())


def classify_bravais(
    vectors: LatticeVectors, tolerances: Tolerances | None = None
) -> BravaisLattice:
    """Classify a two-dimensional lattice from a reduced Gram matrix."""

    if not isinstance(vectors, LatticeVectors):
        raise TypeError("vectors must be a LatticeVectors instance")
    policy = _validate_tolerances(tolerances)
    reduced = reduce_basis(vectors)[:, :2]
    gram = reduced @ reduced.T
    first_length, second_length = np.sqrt(np.diag(gram))
    equal_lengths = np.isclose(
        first_length,
        second_length,
        rtol=policy.relative,
        atol=policy.absolute,
    )
    cosine = float(gram[0, 1] / (first_length * second_length))
    angular_tolerance = float(np.radians(policy.angle_degrees))
    orthogonal = np.isclose(cosine, 0.0, rtol=0.0, atol=angular_tolerance)
    hexagonal_angle = np.isclose(abs(cosine), 0.5, rtol=0.0, atol=angular_tolerance)
    # A rhombic primitive basis can Gauss-reduce to unequal successive minima
    # on the boundary 2|a.b| = min(|a|^2, |b|^2). It still describes a
    # centered-rectangular lattice and commonly appears in reciprocal bases.
    metric_scale = max(float(np.max(np.abs(gram))), np.finfo(float).tiny)
    reduction_boundary = np.isclose(
        2.0 * abs(float(gram[0, 1])),
        min(float(gram[0, 0]), float(gram[1, 1])),
        rtol=policy.relative,
        atol=policy.absolute + angular_tolerance * metric_scale,
    )

    if equal_lengths and hexagonal_angle:
        return BravaisLattice.HEXAGONAL
    if equal_lengths and orthogonal:
        return BravaisLattice.SQUARE
    if equal_lengths or reduction_boundary:
        return BravaisLattice.CENTERED_RECTANGULAR
    if orthogonal:
        return BravaisLattice.RECTANGULAR
    return BravaisLattice.OBLIQUE


def _validate_nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _coefficient_orders(max_coefficient: int) -> IntArray:
    coefficients = np.arange(-max_coefficient, max_coefficient + 1, dtype=np.int64)
    first, second = np.meshgrid(coefficients, coefficients, indexing="ij")
    orders = np.column_stack((first.ravel(), second.ravel()))
    return orders[np.any(orders != 0, axis=1)]


def _group_orders(
    orders: IntArray,
    basis: FloatArray,
    tolerances: Tolerances,
) -> tuple[list[IntArray], FloatArray]:
    if len(orders) == 0:
        return [], np.empty(0, dtype=float)
    gram = basis[:, :2] @ basis[:, :2].T
    squared_distances = np.einsum("ni,ij,nj->n", orders, gram, orders)
    distances = np.sqrt(np.maximum(squared_distances, 0.0))
    ordering = np.lexsort((orders[:, 1], orders[:, 0], distances))
    orders = orders[ordering]
    distances = distances[ordering]

    groups: list[IntArray] = []
    representatives: list[float] = []
    start = 0
    for index in range(1, len(distances) + 1):
        at_end = index == len(distances)
        same_group = not at_end and np.isclose(
            distances[index],
            distances[start],
            rtol=tolerances.relative,
            atol=tolerances.absolute,
        )
        if same_group:
            continue
        group = np.array(orders[start:index], dtype=np.int64, copy=True)
        group.setflags(write=False)
        groups.append(group)
        representatives.append(float(distances[start]))
        start = index
    result_distances = np.asarray(representatives, dtype=float)
    result_distances.setflags(write=False)
    return groups, result_distances


class Lattice:
    """An immutable classified 2D lattice with lazily derived cells."""

    __slots__ = ("_bravais", "_initialized", "_lattice_type", "_vectors", "__dict__")

    def __init__(
        self,
        lattice_vectors: LatticeVectors,
        lattice_type: str | LatticeType = LatticeType.REAL_SPACE,
    ) -> None:
        if not isinstance(lattice_vectors, LatticeVectors):
            raise TypeError("lattice_vectors must be a LatticeVectors instance")
        try:
            parsed_type = LatticeType(lattice_type)
        except (TypeError, ValueError) as error:
            choices = tuple(item.value for item in LatticeType)
            raise ValueError(f"lattice type must be one of {choices}") from error

        object.__setattr__(self, "_initialized", False)
        object.__setattr__(
            self,
            "_vectors",
            LatticeVectors(lattice_vectors.vec1, lattice_vectors.vec2),
        )
        object.__setattr__(self, "_lattice_type", parsed_type)
        object.__setattr__(self, "_bravais", classify_bravais(self._vectors))
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_initialized", False):
            raise AttributeError("Lattice is immutable; construct a new lattice instead")
        object.__setattr__(self, name, value)

    @property
    def vectors(self) -> LatticeVectors:
        return self._vectors

    @property
    def lattice_type(self) -> str:
        return self._lattice_type.value

    @property
    def bravais(self) -> BravaisLattice:
        return self._bravais

    @cached_property
    def primitive_cell(self):
        return make_primitive_cell(self.vectors)

    @cached_property
    def conventional_cell(self):
        return make_conventional_cell(self.vectors, self.bravais)

    @cached_property
    def brillouin_zone(self):
        from reciprocal.brillouin_zone import make_brillouin_zone

        return make_brillouin_zone(self)

    @cached_property
    def unit_cell(self) -> LegacyUnitCell:
        """Return the lazily constructed legacy compatibility cell."""

        from reciprocal.unit_cell import UnitCell as LegacyUnitCell

        warnings.warn(
            "Lattice.unit_cell is deprecated; use primitive_cell, conventional_cell, "
            "or brillouin_zone",
            DeprecationWarning,
            stacklevel=2,
        )
        return LegacyUnitCell(self, WignerSeitz=self._lattice_type is LatticeType.RECIPROCAL)

    @classmethod
    def from_vectors(
        cls,
        vector1: ArrayLike,
        vector2: ArrayLike,
        *,
        lattice_type: str | LatticeType = LatticeType.REAL_SPACE,
    ) -> Lattice:
        return cls(LatticeVectors(vector1, vector2), lattice_type=lattice_type)

    @classmethod
    def from_lengths_angle(
        cls,
        length1: float,
        length2: float,
        angle: float,
        *,
        lattice_type: str | LatticeType = LatticeType.REAL_SPACE,
    ) -> Lattice:
        vectors = LatticeVectors.from_lengths_angle(length1, length2, angle)
        return cls(vectors, lattice_type=lattice_type)

    @classmethod
    def from_lat_vec_args(cls, **kwargs: object) -> Lattice:
        """Compatibility constructor accepting one exact vector argument form."""

        values = dict(kwargs)
        lattice_type = values.pop("lattice_type", LatticeType.REAL_SPACE)
        keys = set(values)
        if keys == {"vector1", "vector2"}:
            return cls.from_vectors(
                values["vector1"],
                values["vector2"],
                lattice_type=lattice_type,  # type: ignore[arg-type]
            )
        if keys == {"length1", "length2", "angle"}:
            return cls.from_lengths_angle(
                float(values["length1"]),
                float(values["length2"]),
                float(values["angle"]),
                lattice_type=lattice_type,  # type: ignore[arg-type]
            )
        raise ValueError(
            "expected exactly vector1/vector2 or length1/length2/angle; " f"got {sorted(keys)}"
        )

    def make_reciprocal(self) -> Lattice:
        """Return the reciprocal of a real-space lattice.

        Reciprocal-of-reciprocal construction is intentionally rejected so
        that the returned object's space semantics are never ambiguous.
        """

        if self._lattice_type is LatticeType.RECIPROCAL:
            raise ValueError("make_reciprocal() requires a real-space lattice")
        return type(self)(self.vectors.reciprocal_vectors(), LatticeType.RECIPROCAL)

    def determine_bravais_lattice(self, tolerances: Tolerances | None = None) -> BravaisLattice:
        """Compatibility wrapper for :func:`classify_bravais`."""

        return classify_bravais(self.vectors, tolerances)

    def enumerate_orders(self, max_coefficient: int) -> tuple[IntArray, FloatArray]:
        """Return translations in a finite coefficient square.

        This is coefficient enumeration, not a guarantee of complete radial
        shells. Use :meth:`translation_shells` when shell completeness matters.
        """

        maximum = _validate_nonnegative_integer(max_coefficient, "max_coefficient")
        orders = _coefficient_orders(maximum)
        if len(orders) == 0:
            distances = np.empty(0, dtype=float)
        else:
            gram = self.vectors.gram_matrix
            squared = np.einsum("ni,ij,nj->n", orders, gram, orders)
            distances = np.sqrt(np.maximum(squared, 0.0))
            ordering = np.lexsort((orders[:, 1], orders[:, 0], distances))
            orders = orders[ordering]
            distances = distances[ordering]
        orders = np.array(orders, dtype=np.int64, copy=True)
        distances = np.array(distances, dtype=float, copy=True)
        orders.setflags(write=False)
        distances.setflags(write=False)
        return orders, distances

    def orders_by_distance(
        self,
        max_order: int,
        tolerances: Tolerances | None = None,
    ) -> tuple[list[IntArray], FloatArray]:
        """Group a finite coefficient square by translation distance.

        This compatibility method preserves the historical coefficient-bound
        meaning of ``max_order`` without decimal rounding.
        """

        maximum = _validate_nonnegative_integer(max_order, "max_order")
        policy = _validate_tolerances(tolerances)
        return _group_orders(_coefficient_orders(maximum), self.vectors.basis, policy)

    def translation_shells(
        self,
        number_of_shells: int,
        tolerances: Tolerances | None = None,
    ) -> tuple[list[IntArray], FloatArray]:
        """Return complete radial shells, expressed in the original basis."""

        count = _validate_nonnegative_integer(number_of_shells, "number_of_shells")
        policy = _validate_tolerances(tolerances)
        if count == 0:
            return [], np.empty(0, dtype=float)

        original = self.vectors.basis[:, :2]
        reduced = reduce_basis(self.vectors)[:, :2]
        transformation = np.linalg.solve(original.T, reduced.T).T
        integer_transformation = np.rint(transformation).astype(np.int64)
        if not np.allclose(transformation, integer_transformation, rtol=0.0, atol=1e-9):
            raise RuntimeError("basis reduction did not produce an integer transformation")

        smallest_singular_value = float(np.min(np.linalg.svd(reduced, compute_uv=False)))
        maximum = max(1, count)
        for _ in range(32):
            reduced_orders = _coefficient_orders(maximum)
            groups, distances = _group_orders(reduced_orders, reduced, policy)
            if len(groups) >= count:
                target_distance = float(distances[count - 1])
                outside_lower_bound = smallest_singular_value * (maximum + 1)
                allowance = policy.absolute + policy.relative * target_distance
                if outside_lower_bound > target_distance + allowance:
                    original_groups = []
                    for group in groups[:count]:
                        converted = group @ integer_transformation
                        converted = np.asarray(converted, dtype=np.int64)
                        converted.setflags(write=False)
                        original_groups.append(converted)
                    result_distances = np.array(distances[:count], copy=True)
                    result_distances.setflags(write=False)
                    return original_groups, result_distances
            maximum *= 2
        raise RuntimeError("could not establish complete translation shells")

    def __repr__(self) -> str:
        return f"Lattice({self.vectors!r}, lattice_type={self.lattice_type!r})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Lattice)
            and self._lattice_type is other._lattice_type
            and self.vectors == other.vectors
        )

    def __hash__(self) -> int:
        return hash((self._lattice_type, self.vectors))
