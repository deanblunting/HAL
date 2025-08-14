"""
Core data structures for HAL algorithm.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
import numpy as np


@dataclass
class QECCLayout:
    """Complete layout result from HAL algorithm."""
    node_positions: Dict[int, Tuple[int, int]]  # node_id -> (x, y)
    edge_routes: Dict[Tuple[int, int], List[Tuple[int, int, int]]]  # edge -> path
    tiers: List[Set[Tuple[int, int]]]  # occupied cells per tier
    metrics: Dict[str, float]  # tiers, length, bumps, tsvs
    hardware_cost: float


@dataclass
class RoutingTier:
    """Represents a single routing tier in the hardware."""
    tier_id: int
    grid: np.ndarray  # 3D occupancy grid (x, y, layer)
    edges: List[Tuple[int, int]]  # edges routed on this tier
    tsvs: Set[Tuple[int, int]]  # TSV locations
    bump_transitions: Dict[Tuple[int, int], int]  # edge -> bump count
    
    def __post_init__(self):
        if self.grid is None:
            self.grid = np.zeros((100, 100, 2), dtype=bool)  # default size
    
    def is_occupied(self, x: int, y: int, layer: int = 0) -> bool:
        """Check if a grid cell is occupied."""
        if 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1] and 0 <= layer < self.grid.shape[2]:
            return self.grid[x, y, layer]
        return True  # treat out-of-bounds as occupied
    
    def set_occupied(self, x: int, y: int, layer: int = 0, occupied: bool = True):
        """Set occupancy of a grid cell."""
        if 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1] and 0 <= layer < self.grid.shape[2]:
            self.grid[x, y, layer] = occupied


@dataclass
class PlacementResult:
    """Result from the placement phase."""
    node_positions: Dict[int, Tuple[int, int]]
    planar_subgraph_edges: Set[Tuple[int, int]]
    grid_bounds: Tuple[int, int, int, int]  # min_x, max_x, min_y, max_y
    communities: Dict[int, int]  # node_id -> community_id


@dataclass
class RoutingResult:
    """Result from the routing phase."""
    edge_routes: Dict[Tuple[int, int], List[Tuple[int, int, int]]]  # edge -> path
    tiers: List[RoutingTier]
    unrouted_edges: Set[Tuple[int, int]]
    metrics: Dict[str, float]


@dataclass
class RouteSegment:
    """A segment of a routed path."""
    start: Tuple[int, int, int]  # (x, y, layer)
    end: Tuple[int, int, int]
    tier: int
    is_bump_transition: bool = False
    is_tsv: bool = False