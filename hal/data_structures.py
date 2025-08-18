"""
Core data structures for HAL algorithm.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
import numpy as np


@dataclass
class QECCLayout:
    """Comprehensive layout representation containing all HAL algorithm results."""
    node_positions: Dict[int, Tuple[int, int]]  # node identifier -> grid coordinates
    edge_routes: Dict[Tuple[int, int], List[Tuple[int, int, int]]]  # edge -> routing path
    tiers: List['RoutingTier']  # multi-tier routing infrastructure
    metrics: Dict[str, float]  # performance metrics (tiers, length, bumps, tsvs)
    hardware_cost: float


@dataclass
class RoutingTier:
    """Hardware routing tier representation with occupancy tracking."""
    tier_id: int
    grid: np.ndarray  # three-dimensional occupancy grid (x, y, layer)
    edges: List[Tuple[int, int]]  # connectivity edges assigned to tier
    tsvs: Set[Tuple[int, int]]  # through-silicon via coordinates
    bump_transitions: Dict[Tuple[int, int], int]  # edge -> bump bond transitions
    
    def __post_init__(self):
        if self.grid is None:
            self.grid = np.zeros((100, 100, 2), dtype=bool)  # default grid dimensions
    
    def is_occupied(self, x: int, y: int, layer: int = 0) -> bool:
        """Determine grid cell occupancy status."""
        if 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1] and 0 <= layer < self.grid.shape[2]:
            return self.grid[x, y, layer]
        return True  # enforce boundary constraints
    
    def set_occupied(self, x: int, y: int, layer: int = 0, occupied: bool = True):
        """Configure grid cell occupancy state."""
        if 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1] and 0 <= layer < self.grid.shape[2]:
            self.grid[x, y, layer] = occupied


@dataclass
class PlacementResult:
    """Node placement algorithm results with spatial assignments."""
    node_positions: Dict[int, Tuple[int, int]]
    planar_subgraph_edges: Set[Tuple[int, int]]
    grid_bounds: Tuple[int, int, int, int]  # spatial boundary constraints
    communities: Dict[int, int]  # node -> community assignments


@dataclass
class RoutingResult:
    """Multi-tier routing algorithm results with path assignments."""
    edge_routes: Dict[Tuple[int, int], List[Tuple[int, int, int]]]  # edge -> routing path
    tiers: List[RoutingTier]
    unrouted_edges: Set[Tuple[int, int]]
    metrics: Dict[str, float]


@dataclass
class RouteSegment:
    """Individual routing path segment with transition metadata."""
    start: Tuple[int, int, int]  # segment origin coordinates
    end: Tuple[int, int, int]  # segment destination coordinates
    tier: int
    is_bump_transition: bool = False
    is_tsv: bool = False