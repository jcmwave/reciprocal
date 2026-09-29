"""Stable, user-facing API for :mod:`reciprocal`.

Plotting is intentionally not imported here, so the core package works without
the optional Matplotlib dependency.
"""

from importlib.metadata import PackageNotFoundError, version

from .band_path import HighSymmetryPath, make_high_symmetry_path, set_band_path_axis
from .bravais import BravaisLattice
from .brillouin_zone import BrillouinZone
from .kspace import KSpace
from .kvector import BlochFamily, KVector, KVectorGroup
from .lattice import Lattice, LatticeVectors
from .numerics import DEFAULT_TOLERANCES, Tolerances
from .reciprocal_mesh import (
    GridCentering,
    MeshReduction,
    MonkhorstPackGrid,
    reduce_monkhorst_pack,
    sample_monkhorst_pack,
)
from .sampling import MaxSpacing, PointCounts, SamplingDomain, SamplingResult
from .symmetry import PointSymmetry, SpecialPoint, Symmetry, SymmetryCombination
from .unit_cell import UnitCell

try:
    __version__ = version("reciprocal")
except PackageNotFoundError:  # Source checkout used without installation.
    __version__ = "0+unknown"

__all__ = [
    "BlochFamily",
    "BravaisLattice",
    "BrillouinZone",
    "DEFAULT_TOLERANCES",
    "GridCentering",
    "HighSymmetryPath",
    "KSpace",
    "KVector",
    "KVectorGroup",
    "Lattice",
    "LatticeVectors",
    "MaxSpacing",
    "MeshReduction",
    "MonkhorstPackGrid",
    "PointCounts",
    "PointSymmetry",
    "SpecialPoint",
    "Symmetry",
    "SymmetryCombination",
    "SamplingDomain",
    "SamplingResult",
    "Tolerances",
    "UnitCell",
    "__version__",
    "make_high_symmetry_path",
    "reduce_monkhorst_pack",
    "sample_monkhorst_pack",
    "set_band_path_axis",
]
