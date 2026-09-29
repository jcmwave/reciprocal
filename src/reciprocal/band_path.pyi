from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .brillouin_zone import BrillouinZone

class HighSymmetryPath:
    points: NDArray[np.float64]
    fractional_points: NDArray[np.float64]
    distance: NDArray[np.float64]
    node_indices: NDArray[np.int64]
    labels: tuple[str, ...]
    segments: tuple[tuple[str, str], ...]
    reciprocal_basis: NDArray[np.float64]
    break_indices: NDArray[np.int64]
    @property
    def tick_positions(self) -> NDArray[np.float64]: ...
    @property
    def tick_labels(self) -> tuple[str, ...]: ...
    def to_kvectors(self, wavelength: float, refractive_index: float, direction: int = ...) -> object: ...

def make_high_symmetry_path(zone: BrillouinZone, labels: Sequence[str] | Sequence[Sequence[str]] | None = ..., *, points_per_segment: int | Sequence[int] | None = ..., max_spacing: float | None = ...) -> HighSymmetryPath: ...
def set_band_path_axis(ax: object, path: HighSymmetryPath) -> None: ...
