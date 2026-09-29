from typing import Any

from reciprocal.brillouin_zone import BrillouinZone
from reciprocal.lattice import Lattice
from reciprocal.numerics import Tolerances
from reciprocal.symmetry import PointOperation

from .domains import KDomain
from .model import FieldSamples, KSampling, SamplingConstraint, SourceRegion, TargetRegion

class ExpansionGenerator:
    source_index: int
    operation: PointOperation
    reciprocal_order: tuple[int, int]

class ExpansionMap:
    source: KSampling
    target: KSampling
    primary_generators: tuple[ExpansionGenerator, ...]
    equivalent_generators: tuple[tuple[ExpansionGenerator, ...], ...]
    def expand(self, data: FieldSamples[Any], *, verify: bool = ...) -> FieldSamples[Any]: ...

class PeriodicSamplingPlan:
    direct_lattice: Lattice
    reciprocal_lattice: Lattice
    zone: BrillouinZone
    source_region: SourceRegion
    target_region: TargetRegion
    representatives: KSampling
    expansion: ExpansionMap
    @property
    def target(self) -> KSampling: ...
    @property
    def speedup(self) -> float: ...
    def expand(self, data: FieldSamples[Any], *, verify: bool = ...) -> FieldSamples[Any]: ...

class PeriodicKSpace:
    parent: object
    direct_lattice: Lattice
    def __init__(self, parent: object, direct_lattice: Lattice) -> None: ...
    def plan(self, *, source: SourceRegion | str = ..., target: TargetRegion | str = ..., constraint: SamplingConstraint = ..., tolerances: Tolerances = ...) -> PeriodicSamplingPlan: ...

def build_expansion_map(source: KSampling, reciprocal_lattice: Lattice, zone: BrillouinZone, target_domain: KDomain, source_region: SourceRegion, target_region: TargetRegion, *, tolerances: Tolerances = ...) -> ExpansionMap: ...
def make_periodic_plan(direct_lattice: Lattice, physical_domain: KDomain, *, source: SourceRegion | str = ..., target: TargetRegion | str = ..., constraint: SamplingConstraint = ..., tolerances: Tolerances = ...) -> PeriodicSamplingPlan: ...
