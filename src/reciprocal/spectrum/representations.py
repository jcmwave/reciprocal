"""Transformation laws for values attached to reciprocal-space points."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

import numpy as np

from reciprocal.symmetry import PointOperation

T = TypeVar("T")


class TranslationAction(Protocol[T]):
    def apply(
        self,
        value: T,
        reciprocal_order: tuple[int, int],
        source_k: np.ndarray,
        target_k: np.ndarray,
        operation: PointOperation,
    ) -> T: ...


@dataclass(frozen=True, slots=True)
class IdentityTranslation:
    """Declare a value invariant under reciprocal-lattice relabelling."""

    def apply(self, value: T, reciprocal_order, source_k, target_k, operation) -> T:
        return value


@dataclass(frozen=True, slots=True)
class RejectTranslations:
    """Reject nonzero reciprocal orders when no Bloch-gauge rule is known."""

    def apply(self, value: T, reciprocal_order, source_k, target_k, operation) -> T:
        if tuple(reciprocal_order) != (0, 0):
            raise ValueError(
                "nonzero reciprocal translation requires an explicit TranslationAction"
            )
        return value


@dataclass(frozen=True, slots=True, eq=False)
class FourierOrderTranslation:
    """Select a solver Fourier coefficient for a transformed target order.

    ``orders`` and ``reciprocal_basis`` use the solver's original row-basis
    convention. The target order is rotated back to the untransformed source
    solution before its coefficient is selected.
    """

    orders: np.ndarray
    reciprocal_basis: np.ndarray
    order_axis: int = 0

    def __post_init__(self) -> None:
        orders = np.asarray(self.orders)
        if orders.ndim != 2 or orders.shape[1] != 2 or not np.all(np.isfinite(orders)):
            raise ValueError("orders must have shape (N, 2) and be finite")
        if not np.all(orders == np.rint(orders)):
            raise ValueError("orders must contain integers")
        basis = np.asarray(self.reciprocal_basis, dtype=float)
        if basis.shape == (2, 3):
            basis = basis[:, :2]
        if basis.shape != (2, 2) or not np.all(np.isfinite(basis)):
            raise ValueError("reciprocal_basis must have shape (2, 2) or (2, 3)")
        if abs(np.linalg.det(basis)) == 0.0:
            raise ValueError("reciprocal_basis vectors must be independent")
        if isinstance(self.order_axis, bool) or not isinstance(self.order_axis, int):
            raise TypeError("order_axis must be an integer")
        orders = np.asarray(np.rint(orders), dtype=np.int64)
        orders.setflags(write=False)
        basis = np.array(basis, copy=True)
        basis.setflags(write=False)
        object.__setattr__(self, "orders", orders)
        object.__setattr__(self, "reciprocal_basis", basis)

    def apply(self, value, reciprocal_order, source_k, target_k, operation):
        point_operation = _operation(operation)
        target_vector = np.asarray(reciprocal_order, dtype=float) @ self.reciprocal_basis
        source_vector = target_vector @ point_operation.cartesian[:2, :2]
        source_fractional = source_vector @ np.linalg.inv(self.reciprocal_basis)
        source_order = np.rint(source_fractional).astype(np.int64)
        if not np.allclose(source_fractional, source_order, rtol=0.0, atol=1e-8):
            raise ValueError("point operation does not preserve the supplied reciprocal basis")
        matches = np.flatnonzero(np.all(self.orders == source_order, axis=1))
        if len(matches) != 1:
            raise KeyError(f"solver data does not contain reciprocal order {tuple(source_order)}")
        array = np.asarray(value)
        axis = self.order_axis if self.order_axis >= 0 else array.ndim + self.order_axis
        if axis < 0 or axis >= array.ndim or array.shape[axis] != len(self.orders):
            raise ValueError("order_axis does not identify the Fourier-order dimension")
        return np.take(array, int(matches[0]), axis=axis)


def _operation(operation: object) -> PointOperation:
    if not isinstance(operation, PointOperation):
        raise TypeError("operation must be a PointOperation")
    return operation


@dataclass(frozen=True, slots=True)
class ScalarRepresentation:
    """An invariant scalar observable."""

    translation: TranslationAction[Any] = field(default_factory=IdentityTranslation)

    def transform(self, value, operation, source_k, target_k, reciprocal_order):
        point_operation = _operation(operation)
        return self.translation.apply(
            value, reciprocal_order, source_k, target_k, point_operation
        )


@dataclass(frozen=True, slots=True)
class ComplexScalarRepresentation(ScalarRepresentation):
    """An invariant complex scalar with an explicit translation policy."""


@dataclass(frozen=True, slots=True)
class PolarVectorRepresentation:
    """A Cartesian polar vector, such as an electric field."""

    translation: TranslationAction[Any] = field(default_factory=RejectTranslations)

    def transform(self, value, operation, source_k, target_k, reciprocal_order):
        matrix = _operation(operation).cartesian
        array = np.asarray(value)
        if array.shape[-1:] != (3,):
            raise ValueError("polar-vector values must end in an axis of length three")
        transformed = array @ matrix.T
        return self.translation.apply(
            transformed, reciprocal_order, source_k, target_k, _operation(operation)
        )


@dataclass(frozen=True, slots=True)
class AxialVectorRepresentation:
    """A Cartesian axial vector, such as a magnetic field."""

    translation: TranslationAction[Any] = field(default_factory=RejectTranslations)

    def transform(self, value, operation, source_k, target_k, reciprocal_order):
        point_operation = _operation(operation)
        array = np.asarray(value)
        if array.shape[-1:] != (3,):
            raise ValueError("axial-vector values must end in an axis of length three")
        transformed = point_operation.determinant * (array @ point_operation.cartesian.T)
        return self.translation.apply(
            transformed, reciprocal_order, source_k, target_k, point_operation
        )


@dataclass(frozen=True, slots=True)
class TensorRepresentation:
    """A Cartesian rank-two tensor transformed as ``R T R.T``."""

    translation: TranslationAction[Any] = field(default_factory=RejectTranslations)

    def transform(self, value, operation, source_k, target_k, reciprocal_order):
        matrix = _operation(operation).cartesian
        array = np.asarray(value)
        if array.shape[-2:] != (3, 3):
            raise ValueError("tensor values must end in axes of shape (3, 3)")
        transformed = np.einsum("ij,...jk,lk->...il", matrix, array, matrix)
        return self.translation.apply(
            transformed, reciprocal_order, source_k, target_k, _operation(operation)
        )


def _sp_frame(k_parallel: np.ndarray, magnitude: float, direction: int) -> np.ndarray:
    transverse = np.asarray(k_parallel, dtype=float)[:2]
    norm = float(np.linalg.norm(transverse))
    if norm == 0.0:
        s = np.array([0.0, 1.0, 0.0], dtype=complex)
    else:
        s = np.array([-transverse[1] / norm, transverse[0] / norm, 0.0], dtype=complex)
    kz = direction * np.sqrt(complex(magnitude * magnitude - norm * norm))
    khat = np.array([transverse[0], transverse[1], kz], dtype=complex) / magnitude
    p = np.cross(s, khat)
    p /= np.sqrt(np.sum(np.abs(p) ** 2))
    return np.stack((s, p), axis=0)


@dataclass(frozen=True, slots=True)
class SPPolarizationRepresentation:
    """Electric-field components in local s/p frames."""

    wavevector_magnitude: float
    direction: int = 1
    translation: TranslationAction[Any] = field(default_factory=RejectTranslations)

    def __post_init__(self) -> None:
        magnitude = float(self.wavevector_magnitude)
        if not np.isfinite(magnitude) or magnitude <= 0.0:
            raise ValueError("wavevector_magnitude must be finite and positive")
        if self.direction not in (-1, 1):
            raise ValueError("direction must be +1 or -1")
        object.__setattr__(self, "wavevector_magnitude", magnitude)

    def transform(self, value, operation, source_k, target_k, reciprocal_order):
        array = np.asarray(value)
        if array.shape[-1:] != (2,):
            raise ValueError("s/p values must end in an axis of length two")
        source_frame = _sp_frame(source_k, self.wavevector_magnitude, self.direction)
        target_frame = _sp_frame(target_k, self.wavevector_magnitude, self.direction)
        cartesian = np.einsum("...i,ij->...j", array, source_frame)
        transformed = cartesian @ _operation(operation).cartesian.T
        components = np.einsum("...j,ij->...i", transformed, np.conjugate(target_frame))
        return self.translation.apply(
            components, reciprocal_order, source_k, target_k, _operation(operation)
        )


__all__ = [
    "AxialVectorRepresentation",
    "ComplexScalarRepresentation",
    "FourierOrderTranslation",
    "IdentityTranslation",
    "PolarVectorRepresentation",
    "RejectTranslations",
    "SPPolarizationRepresentation",
    "ScalarRepresentation",
    "TensorRepresentation",
    "TranslationAction",
]
