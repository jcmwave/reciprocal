"""Bravais-lattice classifications."""

from enum import Enum


class BravaisLattice(Enum):
    """Supported two-dimensional Bravais-lattice types."""

    HEXAGON = 0
    SQUARE = 1
    RECTANGLE = 2
    OBLIQUE = 3
    RHOMBUS = 4
