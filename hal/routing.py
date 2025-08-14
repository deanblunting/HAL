"""
Routing engine for HAL algorithm with multi-tier pathfinding.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
import heapq
from collections import defaultdict, deque

from .data_structures import RoutingTier, RoutingResult, RouteSegment
from .config import HALConfig


class AStarPathfinder:
    """A* pathfinding algorithm for 3D grid routing."""
    
    def __init__(self, config: HALConfig):
        self.config = config
    
    def find_path(self, start: Tuple[int, int, int], end: Tuple[int, int, int], 
                  tier: RoutingTier, avoid_nodes: Set[Tuple[int, int]] = None) -> Optional[List[Tuple[int, int, int]]]:
        """
        Find path from start to end using A* algorithm.
        
        Args:
            start: (x, y, layer) starting position
            end: (x, y, layer) ending position  
            tier: RoutingTier to route on
            avoid_nodes: Set of (x, y) positions to avoid
            
        Returns:
            List of (x, y, layer) positions forming path, or None if no path found
        """
        if avoid_nodes is None:
            avoid_nodes = set()
        
        # Priority queue: (f_score, g_score, position)
        open_set = [(0, 0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, end)}
        visited = set()
        
        while open_set:
            current_f, current_g, current = heapq.heappop(open_set)
            
            if current in visited:
                continue
            visited.add(current)
            
            if current[:2] == end[:2]:  # Reached target (ignore layer for endpoint)
                return self._reconstruct_path(came_from, current)
            
            for neighbor in self._get_neighbors(current, tier):
                if neighbor in visited:
                    continue
                    
                # Check if neighbor is blocked
                nx, ny, nl = neighbor
                if (nx, ny) in avoid_nodes:
                    continue
                if tier.is_occupied(nx, ny, nl):
                    continue
                
                # Calculate movement cost
                tentative_g = g_score[current] + self._movement_cost(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], tentative_g, neighbor))
        
        return None  # No path found
    
    def _heuristic(self, pos1: Tuple[int, int, int], pos2: Tuple[int, int, int]) -> float:
        """Manhattan distance heuristic with layer change penalty."""
        x1, y1, l1 = pos1
        x2, y2, l2 = pos2
        manhattan = abs(x1 - x2) + abs(y1 - y2)
        layer_penalty = abs(l1 - l2) * self.config.layer_change_cost
        return manhattan + layer_penalty
    
    def _movement_cost(self, pos1: Tuple[int, int, int], pos2: Tuple[int, int, int]) -> float:
        """Cost of moving from pos1 to pos2."""
        x1, y1, l1 = pos1
        x2, y2, l2 = pos2
        
        # Base movement cost
        cost = 1.0
        
        # Layer change penalty
        if l1 != l2:
            cost += self.config.layer_change_cost
        
        # Diagonal movement penalty
        if abs(x1 - x2) + abs(y1 - y2) > 1:
            cost += 0.4  # Slightly prefer axis-aligned moves
        
        return cost
    
    def _get_neighbors(self, pos: Tuple[int, int, int], tier: RoutingTier) -> List[Tuple[int, int, int]]:
        """Get valid neighboring positions."""
        x, y, layer = pos
        neighbors = []
        
        # 8-connected movement in same layer
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < tier.grid.shape[0] and 
                    0 <= new_y < tier.grid.shape[1]):
                    neighbors.append((new_x, new_y, layer))
        
        # Layer changes (vias/bumps)
        for new_layer in range(tier.grid.shape[2]):
            if new_layer != layer:
                neighbors.append((x, y, new_layer))
        
        return neighbors
    
    def _reconstruct_path(self, came_from: Dict, current: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Reconstruct path from came_from chain."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]


class StraightLineRouter:
    """Route edges as straight lines when possible."""
    
    def __init__(self, config: HALConfig):
        self.config = config
    
    def route_straight_line(self, start: Tuple[int, int], end: Tuple[int, int], 
                           tier: RoutingTier, layer: int = 0) -> Optional[List[Tuple[int, int, int]]]:
        """
        Route edge as straight line if no collisions.
        
        Args:
            start: (x, y) start position
            end: (x, y) end position
            tier: RoutingTier to route on
            layer: Layer to route on
            
        Returns:
            List of (x, y, layer) positions, or None if blocked
        """
        x1, y1 = start
        x2, y2 = end
        
        # Generate line points using Bresenham's algorithm
        line_points = self._bresenham_line(x1, y1, x2, y2)
        
        # Check for collisions
        for x, y in line_points:
            if tier.is_occupied(x, y, layer):
                return None
        
        # Convert to 3D coordinates
        path = [(x, y, layer) for x, y in line_points]
        return path
    
    def _bresenham_line(self, x1: int, y1: int, x2: int, y2: int) -> List[Tuple[int, int]]:
        """Bresenham's line algorithm for integer coordinates."""
        points = []
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        x_step = 1 if x1 < x2 else -1
        y_step = 1 if y1 < y2 else -1
        
        x, y = x1, y1
        
        if dx > dy:
            error = dx / 2.0
            while x != x2:
                points.append((x, y))
                error -= dy
                if error < 0:
                    y += y_step
                    error += dx
                x += x_step
            points.append((x2, y2))
        else:
            error = dy / 2.0
            while y != y2:
                points.append((x, y))
                error -= dx
                if error < 0:
                    x += x_step
                    error += dy
                y += y_step
            points.append((x2, y2))
        
        return points


class BumpTransitionManager:
    """Manage bump bond transitions between layers."""
    
    def __init__(self, config: HALConfig):
        self.config = config
    
    def add_bump_transitions(self, path: List[Tuple[int, int, int]]) -> Tuple[List[RouteSegment], int]:
        """
        Convert path to segments with bump transition tracking.
        
        Returns:
            Tuple of (segments, bump_count)
        """
        if not path:
            return [], 0
        
        segments = []
        bump_count = 0
        
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            
            # Check if this is a layer change (bump transition)
            is_bump = start[2] != end[2]
            if is_bump:
                bump_count += 1
            
            segment = RouteSegment(
                start=start,
                end=end,
                tier=0,  # Will be set by caller
                is_bump_transition=is_bump,
                is_tsv=(start[:2] == end[:2] and start[2] != end[2])  # Vertical transition
            )
            segments.append(segment)
        
        return segments, bump_count
    
    def exceeds_bump_limit(self, bump_count: int) -> bool:
        """Check if bump count exceeds configured limit."""
        return bump_count > self.config.max_bump_transitions


class RoutingEngine:
    """Main routing engine coordinating all routing algorithms."""
    
    def __init__(self, config: HALConfig):
        self.config = config
        self.straight_router = StraightLineRouter(config)
        self.pathfinder = AStarPathfinder(config)
        self.bump_manager = BumpTransitionManager(config)
        
    def route_edges(self, graph: nx.Graph, node_positions: Dict[int, Tuple[int, int]], 
                   planar_subgraph_edges: Set[Tuple[int, int]]) -> RoutingResult:
        """
        Route all edges across multiple tiers.
        
        Args:
            graph: Connectivity graph
            node_positions: Node positions from placement
            planar_subgraph_edges: Edges in planar subgraph
            
        Returns:
            RoutingResult with all routing information
        """
        tiers = []
        edge_routes = {}
        unrouted_edges = set()
        tier_usage = defaultdict(int)
        
        # Initialize first tier (qubit tier)
        qubit_tier = self._create_tier(0, node_positions)
        tiers.append(qubit_tier)
        
        # Route planar edges first on qubit tier
        remaining_edges = set(graph.edges()) - planar_subgraph_edges
        
        for edge in planar_subgraph_edges:
            path = self._route_edge_on_tier(edge, node_positions, qubit_tier, 0)
            if path:
                edge_routes[edge] = path
                self._mark_path_occupied(path, qubit_tier)
                tier_usage[0] += 1
            else:
                remaining_edges.add(edge)
        
        # Route remaining edges across multiple tiers
        current_tier_id = 0
        
        while remaining_edges and current_tier_id < self.config.max_tiers:
            newly_routed = set()
            current_tier = tiers[current_tier_id] if current_tier_id < len(tiers) else None
            
            # Try to route remaining edges on current tier
            if current_tier is None:
                current_tier = self._create_tier(current_tier_id, node_positions)
                tiers.append(current_tier)
            
            for edge in list(remaining_edges):
                path = self._route_edge_on_tier(edge, node_positions, current_tier, current_tier_id)
                if path:
                    edge_routes[edge] = path
                    self._mark_path_occupied(path, current_tier)
                    newly_routed.add(edge)
                    tier_usage[current_tier_id] += 1
            
            remaining_edges -= newly_routed
            
            # Move to next tier if current tier is congested or no progress made
            if not newly_routed or self._is_tier_congested(current_tier):
                current_tier_id += 1
        
        # Mark unrouted edges
        unrouted_edges = remaining_edges
        
        # Calculate metrics
        metrics = self._calculate_routing_metrics(edge_routes, tiers, node_positions)
        
        return RoutingResult(
            edge_routes=edge_routes,
            tiers=tiers,
            unrouted_edges=unrouted_edges,
            metrics=metrics
        )
    
    def _create_tier(self, tier_id: int, node_positions: Dict[int, Tuple[int, int]]) -> RoutingTier:
        """Create a new routing tier."""
        # Determine grid size from node positions
        if node_positions:
            coords = list(node_positions.values())
            max_x = max(x for x, y in coords) + 10
            max_y = max(y for x, y in coords) + 10
            grid_size = (max(max_x, 50), max(max_y, 50), 2)  # 2 layers per tier
        else:
            grid_size = (50, 50, 2)
        
        tier = RoutingTier(
            tier_id=tier_id,
            grid=np.zeros(grid_size, dtype=bool),
            edges=[],
            tsvs=set(),
            bump_transitions={}
        )
        
        # Mark node positions as occupied on layer 0 for qubit tier
        if tier_id == 0:
            for node, (x, y) in node_positions.items():
                tier.set_occupied(x, y, 0, True)
        
        return tier
    
    def _route_edge_on_tier(self, edge: Tuple[int, int], node_positions: Dict[int, Tuple[int, int]], 
                           tier: RoutingTier, tier_id: int) -> Optional[List[Tuple[int, int, int]]]:
        """Route a single edge on a specific tier."""
        u, v = edge
        if u not in node_positions or v not in node_positions:
            return None
        
        start_pos = node_positions[u]
        end_pos = node_positions[v]
        
        # For tier 0, try straight line routing on layer 0 first
        if tier_id == 0:
            path = self.straight_router.route_straight_line(start_pos, end_pos, tier, 0)
            if path:
                return path
        
        # Try straight line routing on different layers
        for layer in range(tier.grid.shape[2]):
            path = self.straight_router.route_straight_line(start_pos, end_pos, tier, layer)
            if path:
                return path
        
        # Fall back to A* pathfinding
        start_3d = (start_pos[0], start_pos[1], 0)
        end_3d = (end_pos[0], end_pos[1], 0)
        
        path = self.pathfinder.find_path(start_3d, end_3d, tier)
        if path:
            # Check bump transition limits
            segments, bump_count = self.bump_manager.add_bump_transitions(path)
            if not self.bump_manager.exceeds_bump_limit(bump_count):
                tier.bump_transitions[edge] = bump_count
                return path
        
        return None
    
    def _mark_path_occupied(self, path: List[Tuple[int, int, int]], tier: RoutingTier):
        """Mark path cells as occupied in tier grid."""
        for x, y, layer in path:
            tier.set_occupied(x, y, layer, True)
    
    def _is_tier_congested(self, tier: RoutingTier) -> bool:
        """Check if tier is too congested for more routing."""
        occupancy_rate = np.sum(tier.grid) / tier.grid.size
        return occupancy_rate > 0.8  # 80% occupancy threshold
    
    def _calculate_routing_metrics(self, edge_routes: Dict, tiers: List[RoutingTier], 
                                  node_positions: Dict[int, Tuple[int, int]]) -> Dict[str, float]:
        """Calculate routing quality metrics."""
        if not edge_routes:
            return {'tiers': max(len(tiers), 1), 'length': 0.0, 'bumps': 0.0, 'tsvs': 0.0}
        
        total_length = 0.0
        total_bumps = 0
        total_tsvs = 0
        
        for edge, path in edge_routes.items():
            # Calculate path length
            path_length = len(path) if path else 0
            total_length += path_length
            
            # Count bump transitions and TSVs
            for i in range(len(path) - 1):
                current = path[i]
                next_pos = path[i + 1]
                
                # Layer change indicates bump or TSV
                if current[2] != next_pos[2]:
                    if current[:2] == next_pos[:2]:  # Same (x,y) = TSV
                        total_tsvs += 1
                    else:  # Different (x,y) = bump transition
                        total_bumps += 1
        
        num_edges = len(edge_routes)
        
        return {
            'tiers': len(tiers),
            'length': total_length / num_edges if num_edges > 0 else 0.0,
            'bumps': total_bumps / num_edges if num_edges > 0 else 0.0,
            'tsvs': total_tsvs / num_edges if num_edges > 0 else 0.0
        }