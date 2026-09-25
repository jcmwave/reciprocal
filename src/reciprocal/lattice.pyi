from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .bravais import BravaisLattice
from .numerics import Tolerances
from .unit_cell import UnitCell

FloatArray = NDArray[np.float64]

class LatticeVectors:
    vec1: FloatArray
    vec2: FloatArray
    angle: float
    length1: float
    length2: float
    def __init__(self, vector1: ArrayLike, vector2: ArrayLike) -> None: ...
    @classmethod
    def from_lengths_angle(cls, length1: float, length2: float, angle: float) -> LatticeVectors: ...
    def reciprocal_vectors(self) -> LatticeVectors: ...
    def get_shortest_vectors(self) -> tuple[FloatArray, FloatArray]: ...

class Lattice:
    lattice_type: Literal["real_space", "reciprocal"]
    vectors: LatticeVectors
    bravais: BravaisLattice
    unit_cell: UnitCell
    brillouin_zone: UnitCell
    def __init__(self, lattice_vectors: LatticeVectors, lattice_type: Literal["real_space", "reciprocal"] = ...) -> None: ...
    @classmethod
    def from_lat_vec_args(cls, **kwargs: object) -> Lattice: ...
    def make_reciprocal(self) -> Lattice: ...
    def determine_bravais_lattice(self, tolerances: Tolerances | None = ...) -> BravaisLattice: ...
    def orders_by_distance(self, max_order: int) -> tuple[list[FloatArray], FloatArray]: ...
