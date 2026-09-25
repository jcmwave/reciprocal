"""Bravais-lattice classifications."""

from enum import Enum


class BravaisLattice(Enum):
    """Supported two-dimensional Bravais-lattice types."""

    OBLIQUE = 0
    RECTANGULAR = 1
    CENTERED_RECTANGULAR = 2
    SQUARE = 3
    HEXAGONAL = 4




