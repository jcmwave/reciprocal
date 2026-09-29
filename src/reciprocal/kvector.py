"""Optical wave vectors and provenance-aware vector collections."""

from __future__ import annotations

from enum import Enum
import warnings

import numpy as np


def _validate_wavelength(wavelength):
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("wavelength must be a finite positive number")
    return float(wavelength)


def _validate_direction(direction):
    array = np.asarray(direction)
    if not np.all(np.isfinite(array)) or not np.all(np.isin(array, (-1, 1))):
        raise ValueError("normal must contain only +1 or -1")
    return array


def _validate_refractive_index(index):
    array = np.asarray(index)
    if np.iscomplexobj(array) and np.any(np.imag(array) != 0):
        raise ValueError("complex refractive indices are not supported")
    real = np.asarray(np.real(array), dtype=float)
    if not np.all(np.isfinite(real)) or np.any(real <= 0):
        raise ValueError("n must contain finite positive values")
    return real


def _direction_from_kz(kz):
    values = np.asarray(kz, dtype=complex)
    scale = np.maximum(1.0, np.abs(values))
    real_nonzero = np.abs(values.real) > np.finfo(float).eps * scale * 16
    direction = np.where(real_nonzero, np.sign(values.real), np.sign(values.imag))
    return np.where(direction == 0, 1.0, direction)


def _index_from_cartesian(k, k0):
    """Infer a real isotropic index, including a pure-imaginary longitudinal part."""
    dispersion = np.sum(np.asarray(k, dtype=complex) ** 2, axis=-1)
    scale = np.maximum(1.0, np.abs(dispersion.real))
    if np.any(np.abs(dispersion.imag) > np.finfo(float).eps * scale * 64):
        raise ValueError("wave vector is incompatible with a real refractive index")
    real = dispersion.real
    tolerance = np.finfo(float).eps * np.maximum(1.0, np.abs(real)) * 64
    if np.any(real <= tolerance):
        raise ValueError("wave vector must imply a finite positive refractive index")
    return np.sqrt(real) / k0


def _angles_from_cartesian(kx, ky, kz):
    kx_array = np.asarray(kx)
    ky_array = np.asarray(ky)
    kz_array = np.asarray(kz, dtype=complex)
    phi = np.degrees(np.arctan2(np.real(ky_array), np.real(kx_array)))
    transverse = np.sqrt(np.real(kx_array) ** 2 + np.real(ky_array) ** 2)
    propagating = (
        np.abs(kz_array.imag) <= np.finfo(float).eps * np.maximum(1.0, np.abs(kz_array)) * 16
    )
    theta = np.where(
        propagating,
        np.degrees(np.arctan2(transverse, np.abs(kz_array.real))),
        np.nan,
    )
    if np.ndim(theta) == 0:
        return float(theta), float(phi)
    return theta, phi


class KVector:
    """A plane-wave vector in an isotropic, non-absorbing medium.

    ``theta`` is the unsigned angle to the surface normal in ``[0, 90]``.
    ``normal`` independently selects +z/-z propagation or evanescent decay.
    Evanescent vectors have complex ``kz`` and ``theta = NaN``.
    """

    def __init__(
        self,
        wavelength,
        n=None,
        theta=None,
        phi=None,
        normal=None,
        kx=None,
        ky=None,
        kz=None,
        weighting=None,
        validate=True,
    ):
        self.wavelength = _validate_wavelength(wavelength)
        self.k0 = 2 * np.pi / self.wavelength
        self.n, self.theta, self.phi = n, theta, phi
        self.normal_ = normal
        self.kx, self.ky, self.kz = kx, ky, kz
        self.weighting = weighting
        if validate:
            self.complete_data(self.validate_data())

    @classmethod
    def from_cartesian(cls, wavelength, kx, ky, kz, *, weighting=None):
        return cls(wavelength, kx=kx, ky=ky, kz=kz, weighting=weighting)

    @classmethod
    def from_angles(cls, wavelength, n, theta, phi, normal, *, weighting=None):
        return cls(wavelength, n=n, theta=theta, phi=phi, normal=normal, weighting=weighting)

    @classmethod
    def from_transverse(cls, wavelength, n, kx, ky, normal, *, weighting=None):
        return cls(wavelength, n=n, kx=kx, ky=ky, normal=normal, weighting=weighting)

    def __repr__(self):
        return (
            f"KVector(k={self.k!r}, theta={self.theta!r}, phi={self.phi!r}, "
            f"normal={self.normal!r}, n={self.n!r})"
        )

    def validate_data(self):
        combinations = (
            (self.kx, self.ky, self.kz),
            (self.kx, self.ky, self.n, self.normal_),
            (self.theta, self.phi, self.n, self.normal_),
        )
        for combination, values in enumerate(combinations):
            if all(value is not None for value in values):
                if combination in (1, 2):
                    self.n = float(_validate_refractive_index(self.n))
                    self.normal_ = float(_validate_direction(self.normal_))
                if combination == 2:
                    if not np.isfinite(self.theta) or not 0 <= self.theta <= 90:
                        raise ValueError("theta must be finite and in [0, 90] degrees")
                    if not np.isfinite(self.phi):
                        raise ValueError("phi must be finite")
                return combination
        raise ValueError("not enough information to uniquely determine plane wave")

    def complete_data(self, combination):
        if combination == 0:
            k = np.asarray([self.kx, self.ky, self.kz], dtype=complex)
            if not np.all(np.isfinite(k)):
                raise ValueError("wave-vector components must be finite")
            if np.any(np.abs(k[:2].imag) > 0):
                raise ValueError("kx and ky must be real")
            self.kx, self.ky = float(k[0].real), float(k[1].real)
            self.kz = complex(k[2]) if k[2].imag != 0 else float(k[2].real)
            self.n = float(self.get_n_from_k())
            self.normal_ = float(self.get_normal_from_k())
            self.theta, self.phi = self.get_theta_phi_from_k()
        elif combination == 1:
            if not np.all(np.isfinite([self.kx, self.ky])):
                raise ValueError("kx and ky must be finite")
            self.kx, self.ky = float(self.kx), float(self.ky)
            self.kz = self.get_kz_from_kxy_n_normal()
            self.theta, self.phi = self.get_theta_phi_from_k()
        elif combination == 2:
            self.kx, self.ky, self.kz = self.get_k_from_theta_phi_n_normal()

    def get_n_from_k(self):
        return float(_index_from_cartesian(self.k, self.k0))

    def get_normal_from_k(self):
        return float(_direction_from_kz(self.kz))

    def get_kz_from_kxy_n_normal(self):
        radicand = self.knorm**2 - self.kx**2 - self.ky**2
        tolerance = np.finfo(float).eps * max(1.0, self.knorm**2) * 64
        if -tolerance <= radicand < 0:
            radicand = 0.0
        root = np.sqrt(complex(radicand))
        result = self.normal_ * root
        return complex(result) if result.imag != 0 else float(result.real)

    def get_theta_phi_from_k(self):
        return list(_angles_from_cartesian(self.kx, self.ky, self.kz))

    def get_k_from_theta_phi_n_normal(self):
        theta, phi = np.radians(self.theta), np.radians(self.phi)
        return [
            float(self.knorm * np.cos(phi) * np.sin(theta)),
            float(self.knorm * np.sin(phi) * np.sin(theta)),
            float(self.normal_ * self.knorm * np.cos(theta)),
        ]

    def _deprecated_method(self, old_name, new_name, *args):
        warnings.warn(f"{old_name} is deprecated; use {new_name}", DeprecationWarning, stacklevel=2)
        return getattr(self, new_name)(*args)

    def validateData(self):
        return self._deprecated_method("validateData", "validate_data")

    def completeData(self, combination):
        return self._deprecated_method("completeData", "complete_data", combination)

    def getNFromK(self):
        return self._deprecated_method("getNFromK", "get_n_from_k")

    def getNormalFromK(self):
        return self._deprecated_method("getNormalFromK", "get_normal_from_k")

    def getKZFromKXYNNormal(self):
        return self._deprecated_method("getKZFromKXYNNormal", "get_kz_from_kxy_n_normal")

    def getThetaPhiFromK(self):
        return self._deprecated_method("getThetaPhiFromK", "get_theta_phi_from_k")

    def getKFromThetaPhiNNormal(self):
        return self._deprecated_method("getKFromThetaPhiNNormal", "get_k_from_theta_phi_n_normal")

    @property
    def knorm(self):
        return self.n * self.k0

    @property
    def k(self):
        return np.asarray([self.kx, self.ky, self.kz])

    @property
    def is_evanescent(self):
        return bool(np.imag(self.kz) != 0)

    @property
    def normal(self):
        return "+z" if self.normal_ == 1 else "-z"

    def str(self):
        return str(self)


class KVectorGroupColumns(Enum):
    kx = 0
    ky = 1
    kz = 2
    theta = 3
    phi = 4
    normal = 5
    n = 6
    weighting = 7


class KVectorGroup:
    """A vectorized batch of optical wave vectors."""

    def __init__(
        self,
        wavelength,
        n_rows,
        n=None,
        theta=None,
        phi=None,
        normal=None,
        kx=None,
        ky=None,
        kz=None,
        validate=True,
        data=None,
        weighting=None,
    ):
        self.wavelength = _validate_wavelength(wavelength)
        self.k0 = 2 * np.pi / self.wavelength
        if not isinstance(n_rows, (int, np.integer)) or n_rows < 0:
            raise ValueError("n_rows must be a non-negative integer")
        self.n_rows, self.cols = int(n_rows), KVectorGroupColumns
        if data is None:
            is_complex = any(np.iscomplexobj(value) for value in (kx, ky, kz) if value is not None)
            self.data_ = np.full((self.n_rows, 8), np.nan, dtype=complex if is_complex else float)
        else:
            source = np.asarray(data)
            if source.ndim != 2 or source.shape[0] != self.n_rows or source.shape[1] < 8:
                raise ValueError("data must have shape (n_rows, 8) or wider")
            self.data_ = np.array(source, copy=True)
        for column, item in enumerate((kx, ky, kz, theta, phi, normal, n, weighting)):
            if item is not None:
                self.data_[:, column] = item
        if validate:
            self.complete_data(self.validate_data())

    @classmethod
    def from_cartesian(cls, wavelength, k, *, weighting=None):
        array = np.asarray(k)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError("k must have shape (N, 3)")
        return cls(
            wavelength,
            len(array),
            kx=array[:, 0],
            ky=array[:, 1],
            kz=array[:, 2],
            weighting=weighting,
        )

    @classmethod
    def from_angles(cls, wavelength, n, theta, phi, normal, *, weighting=None):
        arrays = np.broadcast_arrays(n, theta, phi, normal)
        return cls(
            wavelength,
            arrays[0].size,
            n=np.ravel(arrays[0]),
            theta=np.ravel(arrays[1]),
            phi=np.ravel(arrays[2]),
            normal=np.ravel(arrays[3]),
            weighting=weighting,
        )

    @classmethod
    def from_transverse(cls, wavelength, n, kx, ky, normal, *, weighting=None):
        arrays = np.broadcast_arrays(n, kx, ky, normal)
        return cls(
            wavelength,
            arrays[0].size,
            n=np.ravel(arrays[0]),
            kx=np.ravel(arrays[1]),
            ky=np.ravel(arrays[2]),
            normal=np.ravel(arrays[3]),
            weighting=weighting,
        )

    def __repr__(self):
        return f"KVectorGroup(k={self.k!r}, theta={self.theta!r}, phi={self.phi!r}, normal={self.normal!r}, n={self.n!r}, weighting={self.weighting!r})"

    def validate_data(self):
        combinations = (
            (self.cols.kx, self.cols.ky, self.cols.kz),
            (self.cols.kx, self.cols.ky, self.cols.n, self.cols.normal),
            (self.cols.theta, self.cols.phi, self.cols.n, self.cols.normal),
        )
        for combination, columns in enumerate(combinations):
            if all(not np.any(np.isnan(self.data_[:, col.value])) for col in columns):
                if combination in (1, 2):
                    _validate_refractive_index(self.n)
                    _validate_direction(self.normal_)
                if combination == 2:
                    if np.any(~np.isfinite(self.theta)) or np.any(
                        (self.theta < 0) | (self.theta > 90)
                    ):
                        raise ValueError("theta must be finite and in [0, 90] degrees")
                    if np.any(~np.isfinite(self.phi)):
                        raise ValueError("phi must be finite")
                return combination
        raise ValueError("not enough information to uniquely determine plane wave")

    def complete_data(self, combination):
        if combination == 0:
            if np.any(np.abs(np.imag(self.k[:, :2])) > 0):
                raise ValueError("kx and ky must be real")
            self.data_[:, self.cols.n.value] = self.get_n_from_k()
            self.data_[:, self.cols.normal.value] = self.get_normal_from_k()
            theta, phi = self.get_theta_phi_from_k()
            self.data_[:, self.cols.theta.value] = theta
            self.data_[:, self.cols.phi.value] = phi
        elif combination == 1:
            radicand = self.knorm**2 - self.kx**2 - self.ky**2
            tolerance = np.finfo(float).eps * np.maximum(1.0, self.knorm**2) * 64
            if np.any(radicand < -tolerance) and not np.iscomplexobj(self.data_):
                self.data_ = self.data_.astype(complex)
            self.data_[:, self.cols.kz.value] = self.get_kz_from_kxy_n_normal()
            theta, phi = self.get_theta_phi_from_k()
            self.data_[:, self.cols.theta.value] = theta
            self.data_[:, self.cols.phi.value] = phi
        elif combination == 2:
            kx, ky, kz = self.get_k_from_theta_phi_n_normal()
            self.data_[:, self.cols.kx.value] = kx
            self.data_[:, self.cols.ky.value] = ky
            self.data_[:, self.cols.kz.value] = kz

    def get_n_from_k(self):
        return _index_from_cartesian(self.k, self.k0)

    def get_normal_from_k(self):
        return _direction_from_kz(self.kz)

    def get_kz_from_kxy_n_normal(self):
        radicand = self.knorm**2 - self.kx**2 - self.ky**2
        tolerance = np.finfo(float).eps * np.maximum(1.0, self.knorm**2) * 64
        radicand = np.where((radicand < 0) & (radicand >= -tolerance), 0.0, radicand)
        result = self.normal_ * np.sqrt(radicand.astype(complex))
        return result if np.any(np.imag(result) != 0) else np.real(result)

    def get_theta_phi_from_k(self):
        return list(_angles_from_cartesian(self.kx, self.ky, self.kz))

    def get_k_from_theta_phi_n_normal(self):
        theta, phi = np.radians(self.theta), np.radians(self.phi)
        return [
            self.knorm * np.cos(phi) * np.sin(theta),
            self.knorm * np.sin(phi) * np.sin(theta),
            self.normal_ * self.knorm * np.cos(theta),
        ]

    @property
    def knorm(self):
        return self.n * self.k0

    @property
    def k(self):
        return self.data_[:, :3]

    def _real_column(self, column):
        return np.asarray(np.real(self.data_[:, column.value]), dtype=float)

    @property
    def kx(self):
        return self._real_column(self.cols.kx)

    @property
    def ky(self):
        return self._real_column(self.cols.ky)

    @property
    def kz(self):
        values = self.data_[:, self.cols.kz.value]
        return (
            values if np.iscomplexobj(values) and np.any(np.imag(values) != 0) else np.real(values)
        )

    @property
    def theta(self):
        return self._real_column(self.cols.theta)

    @property
    def phi(self):
        return self._real_column(self.cols.phi)

    @property
    def normal_(self):
        return self._real_column(self.cols.normal)

    @property
    def n(self):
        return self._real_column(self.cols.n)

    @property
    def weighting(self):
        return self._real_column(self.cols.weighting)

    @property
    def is_evanescent(self):
        return np.abs(np.imag(np.asarray(self.kz, dtype=complex))) > 0

    @property
    def normal(self):
        return np.where(self.normal_ == 1, "+z", "-z")

    def _deprecated_method(self, old_name, new_name, *args):
        warnings.warn(f"{old_name} is deprecated; use {new_name}", DeprecationWarning, stacklevel=2)
        return getattr(self, new_name)(*args)

    def validateData(self):
        return self._deprecated_method("validateData", "validate_data")

    def completeData(self, combination):
        return self._deprecated_method("completeData", "complete_data", combination)

    def getNFromK(self):
        return self._deprecated_method("getNFromK", "get_n_from_k")

    def getNormalFromK(self):
        return self._deprecated_method("getNormalFromK", "get_normal_from_k")

    def getKZFromKXYNNormal(self):
        return self._deprecated_method("getKZFromKXYNNormal", "get_kz_from_kxy_n_normal")

    def getThetaPhiFromK(self):
        return self._deprecated_method("getThetaPhiFromK", "get_theta_phi_from_k")

    def getKFromThetaPhiNNormal(self):
        return self._deprecated_method("getKFromThetaPhiNNormal", "get_k_from_theta_phi_n_normal")

    def sort(self, column, order="ascending"):
        try:
            values = self.data_[:, self.cols[column].value]
        except KeyError as exc:
            raise ValueError(f"unknown k-vector column: {column}") from exc
        key = np.abs(values) if order.startswith("absolute_") else np.real(values)
        if order in ("ascending", "absolute_ascending"):
            indices = np.argsort(key)
        elif order in ("descending", "absolute_descending"):
            indices = np.argsort(-key)
        else:
            raise ValueError(f"unknown sort order: {order}")
        self.data_ = self.data_[indices, :]

    def slice(self, row):
        data = self.data_[row, :]
        return KVector(
            self.wavelength,
            n=float(np.real(data[6])),
            theta=float(np.real(data[3])),
            phi=float(np.real(data[4])),
            normal=float(np.real(data[5])),
            kx=float(np.real(data[0])),
            ky=float(np.real(data[1])),
            kz=data[2],
            weighting=float(np.real(data[7])),
            validate=False,
        )

    def __add__(self, other):
        if not isinstance(other, KVectorGroup):
            return NotImplemented
        if not np.isclose(other.wavelength, self.wavelength):
            raise ValueError("cannot combine k-vector groups with different wavelengths")
        data = np.concatenate((self.data_[:, :8], other.data_[:, :8]), axis=0)
        return KVectorGroup(self.wavelength, len(data), data=data, validate=False)


class BlochFamilyColumns(Enum):
    order1 = 8
    order2 = 9


class BlochVector(KVector):
    """One vector together with reciprocal-translation provenance."""

    def __init__(self, *args, order, representative=None, reciprocal_basis=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.order = tuple(int(value) for value in order)
        self.representative = (
            None if representative is None else np.array(representative, copy=True)
        )
        self.reciprocal_basis = (
            None if reciprocal_basis is None else np.array(reciprocal_basis, copy=True)
        )

    @property
    def order1(self):
        return self.order[0]

    @property
    def order2(self):
        return self.order[1]


class BlochFamily(KVectorGroup):
    """Wave vectors generated as ``q + m*G1 + n*G2``."""

    def __init__(
        self, *args, order1=None, order2=None, representative=None, reciprocal_basis=None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        extended = np.full((self.n_rows, 10), np.nan, dtype=self.data_.dtype)
        extended[:, :8] = self.data_[:, :8]
        self.data_ = extended
        self.representative = self._planar_vector(representative, "representative")
        self.reciprocal_basis = self._basis(reciprocal_basis)
        if (order1 is None) != (order2 is None):
            raise ValueError("order1 and order2 must be provided together")
        if order1 is not None:
            self.set_orders(order1, order2)

    @staticmethod
    def _planar_vector(value, name):
        if value is None:
            return None
        array = np.asarray(value, dtype=float)
        if array.shape not in ((2,), (3,)) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be a finite planar vector")
        if array.shape == (3,) and array[2] != 0:
            raise ValueError(f"{name} must lie in the x-y plane")
        return np.array(array[:2], copy=True)

    @staticmethod
    def _basis(value):
        if value is None:
            return None
        array = np.asarray(value, dtype=float)
        if array.shape == (2, 3):
            if np.any(array[:, 2] != 0):
                raise ValueError("reciprocal_basis must lie in the x-y plane")
            array = array[:, :2]
        if array.shape != (2, 2) or not np.all(np.isfinite(array)):
            raise ValueError("reciprocal_basis must have shape (2, 2) or (2, 3)")
        if abs(np.linalg.det(array)) == 0:
            raise ValueError("reciprocal_basis vectors must be independent")
        return np.array(array, copy=True)

    @classmethod
    def from_kvector_group(
        cls, kvector_group, *, representative=None, reciprocal_basis=None, order1=None, order2=None
    ):
        return cls(
            kvector_group.wavelength,
            kvector_group.n_rows,
            data=kvector_group.data_[:, :8],
            validate=False,
            representative=representative,
            reciprocal_basis=reciprocal_basis,
            order1=order1,
            order2=order2,
        )

    def set_orders(self, order1, order2):
        first, second = np.asarray(order1), np.asarray(order2)
        if first.shape != (self.n_rows,) or second.shape != (self.n_rows,):
            raise ValueError("Bloch orders must have shape (n_rows,)")
        if (
            not np.all(np.isfinite(first))
            or not np.all(np.isfinite(second))
            or not np.all(first == np.rint(first))
            or not np.all(second == np.rint(second))
        ):
            raise ValueError("Bloch orders must be finite integers")
        self.data_[:, 8], self.data_[:, 9] = np.rint(first), np.rint(second)
        self._validate_generation()

    def _validate_generation(self):
        if self.representative is None or self.reciprocal_basis is None:
            return
        expected = self.representative + self.orders @ self.reciprocal_basis
        if not np.allclose(self.k[:, :2], expected, rtol=1e-9, atol=1e-12):
            raise ValueError("members do not equal representative + orders @ reciprocal_basis")

    @property
    def orders(self):
        values = np.real(self.data_[:, 8:10])
        return values if np.any(np.isnan(values)) else np.asarray(np.rint(values), dtype=np.int64)

    @property
    def order1(self):
        return self.orders[:, 0]

    @property
    def order2(self):
        return self.orders[:, 1]

    def slice(self, row):
        vector = super().slice(row)
        return BlochVector(
            vector.wavelength,
            n=vector.n,
            theta=vector.theta,
            phi=vector.phi,
            normal=vector.normal_,
            kx=vector.kx,
            ky=vector.ky,
            kz=vector.kz,
            weighting=vector.weighting,
            validate=False,
            order=self.orders[row],
            representative=self.representative,
            reciprocal_basis=self.reciprocal_basis,
        )

    def __add__(self, other):
        if not isinstance(other, BlochFamily):
            return super().__add__(other)
        if not np.isclose(other.wavelength, self.wavelength):
            raise ValueError("cannot combine Bloch families with different wavelengths")
        for name in ("representative", "reciprocal_basis"):
            first, second = getattr(self, name), getattr(other, name)
            if (first is None) != (second is None) or (
                first is not None and not np.allclose(first, second)
            ):
                raise ValueError(f"cannot combine Bloch families with different {name}")
        result = BlochFamily(
            self.wavelength,
            self.n_rows + other.n_rows,
            data=np.concatenate((self.data_[:, :8], other.data_[:, :8])),
            validate=False,
            representative=self.representative,
            reciprocal_basis=self.reciprocal_basis,
        )
        result.set_orders(
            np.concatenate((self.order1, other.order1)), np.concatenate((self.order2, other.order2))
        )
        return result

    def __repr__(self):
        return f"BlochFamily(k={self.k!r}, orders={self.orders!r}, representative={self.representative!r}, reciprocal_basis={self.reciprocal_basis!r})"
