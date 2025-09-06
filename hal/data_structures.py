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
    existing_paths: Dict[int, List[List[Tuple[int, int]]]] = None  # layer -> list of 2D paths for crossing detection
    
    def __post_init__(self):
        if self.grid is None:
            self.grid = np.zeros((100, 100, 2), dtype=bool)  # default grid dimensions
        if self.existing_paths is None:
            self.existing_paths = {}
    
    def is_occupied(self, x: int, y: int, layer: int = 0) -> bool:
        """Determine grid cell occupancy status."""
        if 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1] and 0 <= layer < self.grid.shape[2]:
            return self.grid[x, y, layer]
        return True  # enforce boundary constraints
    
    def set_occupied(self, x: int, y: int, layer: int = 0, occupied: bool = True):
        """Configure grid cell occupancy state."""
        if 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1] and 0 <= layer < self.grid.shape[2]:
            self.grid[x, y, layer] = occupied
    
    def would_path_cross(self, path_2d: List[Tuple[int, int]], layer: int) -> bool:
        """Check if a 2D path would cross any existing paths on the given layer."""
        if layer not in self.existing_paths:
            return False
        
        for existing_path in self.existing_paths[layer]:
            if self._paths_intersect(path_2d, existing_path):
                return True
        return False
    
    def _paths_intersect(self, path1: List[Tuple[int, int]], path2: List[Tuple[int, int]]) -> bool:
        """Check if two paths intersect by testing all segment pairs."""
        if len(path1) < 2 or len(path2) < 2:
            return False
        
        for i in range(len(path1) - 1):
            for j in range(len(path2) - 1):
                if self._line_segments_intersect(path1[i], path1[i + 1], path2[j], path2[j + 1]):
                    return True
        return False
    
    def _line_segments_intersect(self, p1: Tuple[int, int], q1: Tuple[int, int], 
                                p2: Tuple[int, int], q2: Tuple[int, int]) -> bool:
        """Check if two line segments intersect using orientation method."""
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0:
                return 0  # collinear
            return 1 if val > 0 else 2  # clockwise or counterclockwise
        
        def on_segment(p, q, r):
            return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                    q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
        
        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)
        
        # General case
        if o1 != o2 and o3 != o4:
            return True
        
        # Special cases for collinear points
        if (o1 == 0 and on_segment(p1, p2, q1)) or \
           (o2 == 0 and on_segment(p1, q2, q1)) or \
           (o3 == 0 and on_segment(p2, p1, q2)) or \
           (o4 == 0 and on_segment(p2, q1, q2)):
            return True
        
        return False
    
    def add_path(self, path_2d: List[Tuple[int, int]], layer: int):
        """Add a 2D path to the crossing detection tracking."""
        if layer not in self.existing_paths:
            self.existing_paths[layer] = []
        self.existing_paths[layer].append(path_2d)


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