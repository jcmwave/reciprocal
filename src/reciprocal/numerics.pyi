from numpy.typing import ArrayLike

class Tolerances:
    relative: float
    absolute: float
    angle_degrees: float
    degeneracy: float
    boundary: float
    def __init__(self, relative: float = ..., absolute: float = ..., angle_degrees: float = ..., degeneracy: float = ..., boundary: float = ...) -> None: ...

DEFAULT_TOLERANCES: Tolerances

def contains_close_point(points: ArrayLike, point: ArrayLike, *, relative_tolerance: float = ..., absolute_tolerance: float = ...) -> bool: ...
