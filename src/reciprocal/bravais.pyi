from enum import Enum

class BravaisLattice(Enum):
    HEXAGONAL: int
    SQUARE: int
    RECTANGULAR: int
    OBLIQUE: int
    CENTERED_RECTANGULAR: int
