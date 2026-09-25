from typing import Any, Literal, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .kvector import KVectorGroup
from .lattice import Lattice
from .symmetry import Symmetry, SymmetryCombination

FloatArray = NDArray[np.float64]
SamplingConstraint = dict[str, object]

class RegularSampler:
    @overload
    def sample(self, grid_type: Literal["cartesian", "circular"] = ..., constraint: SamplingConstraint | None = ..., center: bool = ..., cutoff_tol: float = ..., restrict_to_sym_cone: bool = ..., return_artists: Literal[False] = ...) -> tuple[KVectorGroup, FloatArray]: ...
    @overload
    def sample(self, grid_type: Literal["cartesian", "circular"] = ..., constraint: SamplingConstraint | None = ..., center: bool = ..., cutoff_tol: float = ..., restrict_to_sym_cone: bool = ..., return_artists: Literal[True] = ...) -> tuple[KVectorGroup, FloatArray, list[object]]: ...

class PeriodicSampler:
    def sample(self, constraint: SamplingConstraint | None = ..., center: ArrayLike = ..., use_symmetry: bool = ..., cutoff_tol: float = ..., restrict_to_sym_cone: bool = ...) -> KVectorGroup: ...
    def plot_symmetry_families(self, ax: Any, n: int | Literal["all"] = ..., color: Any | None = ...) -> None: ...
    def plotSymmetryFamilies(self, ax: Any, n: int | Literal["all"] = ..., color: Any | None = ...) -> None: ...

class KSpace:
    wavelength: float
    k0: float
    fermi_radius: float | None
    symmetry: Symmetry | SymmetryCombination | None
    symmetry_cone: FloatArray | None
    regular_sampler: RegularSampler
    periodic_sampler: PeriodicSampler | None
    def __init__(self, wavelength: float, symmetry: str | None = ..., fermi_radius: float | None = ...) -> None: ...
    def set_symmetry(self, symmetry: str) -> None: ...
    def apply_lattice(self, lattice: Lattice) -> None: ...
    def restrict_to_fermi_radius(self, points: ArrayLike, tol: float = ..., return_indices: bool = ...) -> FloatArray | tuple[FloatArray, NDArray[bool]]: ...
