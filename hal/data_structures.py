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
    """Hardware routing tier representation with 3D coordinate-based occupancy tracking."""
    tier_id: int
    grid: np.ndarray  # three-dimensional occupancy grid (x, y, layer) - kept for compatibility
    edges: List[Tuple[int, int]]  # connectivity edges assigned to tier
    tsvs: Set[Tuple[int, int]]  # through-silicon via coordinates
    bump_transitions: Dict[Tuple[int, int], int]  # edge -> bump bond transitions
    occupied_coords: Set[Tuple[int, int, int]] = field(default_factory=set)  # 3D coordinates that are occupied
    crossing_detector: 'CrossingDetector' = field(default_factory=lambda: None)  # unified crossing detection

    def __post_init__(self):
        if self.grid is None:
            self.grid = np.zeros((100, 100, 2), dtype=bool)  # default grid dimensions
        if self.crossing_detector is None:
            from .crossing_detector import CrossingDetector
            self.crossing_detector = CrossingDetector()
    
    def is_occupied(self, x: int, y: int, layer: int = 0) -> bool:
        """Determine if coordinate (x,y,z) is occupied using simple coordinate-based tracking."""
        # Check bounds
        if (x < 0 or x >= self.grid.shape[0] or 
            y < 0 or y >= self.grid.shape[1] or 
            layer < 0 or layer >= self.grid.shape[2]):
            return True  # out of bounds = occupied
        
        # Simple coordinate check
        return (x, y, layer) in self.occupied_coords
    
    def set_occupied(self, x: int, y: int, layer: int = 0, occupied: bool = True):
        """Configure coordinate (x,y,z) occupancy state using simple coordinate-based tracking."""
        # Check bounds
        if (0 <= x < self.grid.shape[0] and 
            0 <= y < self.grid.shape[1] and 
            0 <= layer < self.grid.shape[2]):
            
            coord = (x, y, layer)
            if occupied:
                self.occupied_coords.add(coord)
            else:
                self.occupied_coords.discard(coord)
            
            # Also update grid for compatibility
            self.grid[x, y, layer] = occupied
    
    def is_coordinate_available(self, x: int, y: int, layer: int) -> bool:
        """Check if coordinate (x,y,z) is available for routing."""
        return not self.is_occupied(x, y, layer)
    


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