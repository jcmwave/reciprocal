"""Stable, user-facing API for :mod:`reciprocal`.

Plotting is intentionally not imported here, so the core package works without
the optional Matplotlib dependency.
"""

from importlib.metadata import PackageNotFoundError, version

from .kspace import KSpace
from .kvector import BlochFamily, KVector, KVectorGroup
from .lattice import Lattice, LatticeVectors
from .numerics import DEFAULT_TOLERANCES, Tolerances
from .symmetry import PointSymmetry, SpecialPoint, Symmetry, SymmetryCombination
from .unit_cell import UnitCell
from .utils import BravaisLattice

try:
    __version__ = version("reciprocal")
except PackageNotFoundError:  # Source checkout used without installation.
    __version__ = "0+unknown"

__all__ = [
    "BlochFamily",
    "BravaisLattice",
    "DEFAULT_TOLERANCES",
    "KSpace",
    "KVector",
    "KVectorGroup",
    "Lattice",
    "LatticeVectors",
    "PointSymmetry",
    "SpecialPoint",
    "Symmetry",
    "SymmetryCombination",
    "Tolerances",
    "UnitCell",
    "__version__",
]
