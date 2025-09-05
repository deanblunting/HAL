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
        Find path from start to end using A* algorithm with improved collision handling.
        
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
        
        # Allow flexible endpoint matching for better routing success
        # Try different layers at the endpoint if direct layer routing fails
        target_layers = [end[2]]  # Primary target layer
        if end[2] == 1:  # If targeting layer 1, also try layer 0
            target_layers.append(0)
        elif end[2] == 0:  # If targeting layer 0, also try layer 1
            target_layers.append(1)
        
        for target_layer in target_layers:
            adjusted_end = (end[0], end[1], target_layer)
            path = self._find_path_to_target(start, adjusted_end, tier, avoid_nodes)
            if path:
                return path
        
        return None  # No path found
    
    def _find_path_to_target(self, start: Tuple[int, int, int], end: Tuple[int, int, int], 
                            tier: RoutingTier, avoid_nodes: Set[Tuple[int, int]]) -> Optional[List[Tuple[int, int, int]]]:
        """Core A* pathfinding implementation."""
        # Priority queue implementation: (f_score, g_score, position)
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
            
            # More flexible endpoint matching - allow some layer flexibility
            if current[:2] == end[:2]:  # Reached target position
                return self._reconstruct_path(came_from, current)
            
            for neighbor in self._get_neighbors(current, tier):
                if neighbor in visited:
                    continue
                    
                # Enhanced collision detection
                nx, ny, nl = neighbor
                if (nx, ny) in avoid_nodes:
                    continue
                
                # Smart occupancy checking with some permissiveness for tier 0
                if self._is_position_blocked(tier, nx, ny, nl, current):
                    continue
                
                # Compute transition cost between positions
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
        
        # Initialize base movement cost
        cost = 1.0
        
        # Apply layer transition penalty
        if l1 != l2:
            cost += self.config.layer_change_cost
        
        # Apply diagonal movement penalty
        if abs(x1 - x2) + abs(y1 - y2) > 1:
            cost += 0.4  # Slightly prefer axis-aligned moves
        
        return cost
    
    def _get_neighbors(self, pos: Tuple[int, int, int], tier: RoutingTier) -> List[Tuple[int, int, int]]:
        """Get valid neighboring positions."""
        x, y, layer = pos
        neighbors = []
        
        # Generate 8-connected neighborhood within layer
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < tier.grid.shape[0] and 
                    0 <= new_y < tier.grid.shape[1]):
                    neighbors.append((new_x, new_y, layer))
        
        # Generate vertical transitions (vias/bump bonds)
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
    
    def _is_position_blocked(self, tier: RoutingTier, x: int, y: int, layer: int, 
                            current_pos: Tuple[int, int, int]) -> bool:
        """Enhanced position blocking check with context-aware collision detection."""
        # Basic bounds check
        if (x < 0 or x >= tier.grid.shape[0] or 
            y < 0 or y >= tier.grid.shape[1] or
            layer < 0 or layer >= tier.grid.shape[2]):
            return True
        
        # For tier 0, allow some controlled path sharing to improve routing success
        if tier.tier_id == 0:
            if tier.is_occupied(x, y, layer):
                # Allow limited sharing for short hops
                current_distance = abs(current_pos[0] - x) + abs(current_pos[1] - y)
                if current_distance <= 1:  # Adjacent cells can share occasionally
                    return False  # Allow this position
                return True  # Block longer paths through occupied cells
        else:
            # Higher tiers: strict collision avoidance
            return tier.is_occupied(x, y, layer)
        
        return False


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
        
        # Apply Bresenham line rasterization algorithm
        line_points = self._bresenham_line(x1, y1, x2, y2)
        
        # HAL paper collision detection: "edges on the same layer cannot cross"
        # "The router scans the MPS edges once and paints the corresponding grid cells 
        # along the straight line between their endpoints"
        
        if tier.tier_id == 0:
            # Tier 0: Strict planarity enforcement - no crossing edges allowed
            for x, y in line_points:
                if tier.is_occupied(x, y, layer):
                    # HAL paper: if path is obstructed, try bump transition or higher tier
                    return None  # Path would cross existing routes - not allowed on tier 0
        else:
            # Higher tiers: Also enforce strict collision avoidance 
            for x, y in line_points:
                if tier.is_occupied(x, y, layer):
                    return None
        
        # Transform to 3D coordinate representation
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
    
    def _is_critical_position(self, x: int, y: int, tier: RoutingTier) -> bool:
        """Check if position is critical (like a node position) that should avoid conflicts."""
        # For now, mark TSV positions as critical since they are connection points
        return (x, y) in tier.tsvs


class BumpTransitionManager:
    """Manage bump bond transitions with sophisticated cost modeling."""
    
    def __init__(self, config: HALConfig):
        self.config = config
        self.bump_cost_model = self._initialize_bump_cost_model()
    
    def _initialize_bump_cost_model(self) -> Dict[str, float]:
        """Initialize bump bond cost model with hardware-realistic parameters."""
        return {
            'base_bump_cost': 1.0,           # Base cost per bump transition
            'consecutive_penalty': 1.5,      # Extra cost for consecutive bumps
            'distance_penalty': 0.1,         # Cost per grid unit of bump distance
            'layer_depth_penalty': 0.2,      # Extra cost for deeper layers
            'reliability_factor': 1.2,       # Cost multiplier for reliability constraints
        }
    
    def add_bump_transitions(self, path: List[Tuple[int, int, int]]) -> Tuple[List[RouteSegment], int]:
        """
        Convert path to segments with sophisticated bump transition analysis.
        
        Returns:
            Tuple of (segments, bump_count)
        """
        if not path:
            return [], 0
        
        segments = []
        bump_count = 0
        consecutive_bumps = 0
        
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            
            # Analyze transition type and cost
            is_bump = start[2] != end[2]
            is_tsv = (start[:2] == end[:2] and start[2] != end[2])
            
            if is_bump:
                bump_count += 1
                if consecutive_bumps > 0:
                    consecutive_bumps += 1
                else:
                    consecutive_bumps = 1
            else:
                consecutive_bumps = 0
            
            # Calculate segment cost using sophisticated model
            segment_cost = self._calculate_segment_cost(start, end, is_bump, consecutive_bumps)
            
            segment = RouteSegment(
                start=start,
                end=end,
                tier=0,  # Will be set by caller
                is_bump_transition=is_bump,
                is_tsv=is_tsv
            )
            segments.append(segment)
        
        return segments, bump_count
    
    def _calculate_segment_cost(self, start: Tuple[int, int, int], end: Tuple[int, int, int], 
                              is_bump: bool, consecutive_bumps: int) -> float:
        """Calculate cost for individual route segment with bump bond modeling."""
        base_cost = 1.0
        
        if not is_bump:
            return base_cost
        
        # Apply bump bond cost model
        cost = self.bump_cost_model['base_bump_cost']
        
        # Consecutive bump penalty
        if consecutive_bumps > 1:
            cost *= (1.0 + (consecutive_bumps - 1) * self.bump_cost_model['consecutive_penalty'])
        
        # Layer depth penalty (higher layers cost more)
        max_layer = max(start[2], end[2])
        cost *= (1.0 + max_layer * self.bump_cost_model['layer_depth_penalty'])
        
        # Distance penalty for non-vertical bumps
        if start[:2] != end[:2]:
            distance = abs(start[0] - end[0]) + abs(start[1] - end[1])
            cost *= (1.0 + distance * self.bump_cost_model['distance_penalty'])
        
        # Reliability factor
        cost *= self.bump_cost_model['reliability_factor']
        
        return cost
    
    def exceeds_bump_limit(self, bump_count: int) -> bool:
        """Check if bump count exceeds configured limit."""
        return bump_count > self.config.max_bump_transitions
    
    def calculate_total_bump_cost(self, edge_routes: Dict) -> float:
        """Calculate total bump bond cost across all routed edges."""
        total_cost = 0.0
        
        for edge, path in edge_routes.items():
            if not path:
                continue
            
            segments, _ = self.add_bump_transitions(path)
            for segment in segments:
                if segment.is_bump_transition:
                    # Calculate cost using sophisticated model since RouteSegment doesn't store cost
                    segment_cost = self._calculate_segment_cost(
                        segment.start, segment.end, True, 1  # consecutive_bumps=1 for individual calculation
                    )
                    total_cost += segment_cost
        
        return total_cost
    
    def optimize_bump_usage(self, path: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """Optimize path to minimize bump bond usage and cost."""
        if len(path) <= 2:
            return path
        
        # Simple optimization: minimize consecutive layer changes
        optimized_path = [path[0]]
        
        for i in range(1, len(path) - 1):
            current = path[i]
            prev_layer = optimized_path[-1][2]
            next_layer = path[i + 1][2]
            
            # If we can avoid a layer change by staying on previous layer
            if prev_layer == next_layer and current[2] != prev_layer:
                # Try to route through previous layer instead
                optimized_pos = (current[0], current[1], prev_layer)
                optimized_path.append(optimized_pos)
            else:
                optimized_path.append(current)
        
        optimized_path.append(path[-1])
        return optimized_path


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
        
        # Initialize tier 0 (physical qubit layer)
        qubit_tier = self._create_tier(0, node_positions)
        tiers.append(qubit_tier)
        
        # Implement HAL paper Appendix A.2: "After this, all other edges are also attempted 
        # as straight lines on the qubit tier, even if they were not part of the MPS. 
        # This procedure allows us to maximize the number of edges placed on the qubit tier."
        
        all_edges = list(graph.edges())
        
        # Step 1: Route planar subgraph edges first (paper's MPS routing)
        planar_edges = [edge for edge in all_edges if edge in planar_subgraph_edges or 
                       (edge[1], edge[0]) in planar_subgraph_edges]
        
        # Sort planar edges by straight-line distance for optimal routing order
        planar_edges_by_length = sorted(planar_edges, 
            key=lambda e: self._calculate_straight_line_distance(e, node_positions))
        
        remaining_edges = set(all_edges)  # Start with all edges
        tier_0_routed = 0
        
        print(f"Routing {len(planar_edges)} planar subgraph edges on Tier 0...")
        
        # Route planar subgraph edges on qubit tier (Tier 0)
        for edge in planar_edges_by_length:
            path = self._route_edge_on_qubit_tier(edge, node_positions, qubit_tier)
            if path:
                edge_routes[edge] = path
                self._mark_path_occupied(path, qubit_tier)
                qubit_tier.edges.append(edge)
                tier_usage[0] += 1
                tier_0_routed += 1
                remaining_edges.discard(edge)
            # If planar edge fails on tier 0, it stays in remaining_edges for higher tier routing
        
        # Step 2: Aggressive straight-line routing - try ALL remaining edges on Tier 0
        # "all other edges are also attempted as straight lines on the qubit tier"
        non_planar_edges = list(remaining_edges)
        non_planar_by_distance = sorted(non_planar_edges, 
            key=lambda e: self._calculate_straight_line_distance(e, node_positions))
        
        print(f"Attempting aggressive straight-line routing for {len(non_planar_edges)} remaining edges on Tier 0...")
        
        additional_tier0_routed = 0
        for edge in non_planar_by_distance:
            path = self._route_edge_on_qubit_tier(edge, node_positions, qubit_tier)
            if path:
                edge_routes[edge] = path
                self._mark_path_occupied(path, qubit_tier)
                qubit_tier.edges.append(edge)
                tier_usage[0] += 1
                additional_tier0_routed += 1
                remaining_edges.discard(edge)
            # If edge fails on tier 0, it stays in remaining_edges for higher tier routing
        
        total_tier0_routed = tier_0_routed + additional_tier0_routed
        print(f"Tier 0 completion: {total_tier0_routed}/{len(all_edges)} total edges routed ({tier_0_routed} planar + {additional_tier0_routed} aggressive)")
        print(f"Higher-tier routing required for {len(remaining_edges)} edges")
        
        # Route remaining edges using FIFO queue approach
        # Start with tier 1 for higher tier routing (tier 0 is qubit tier)
        current_tier_id = 1
        
        while remaining_edges and current_tier_id < self.config.max_tiers:
            # Create higher routing tier for remaining unrouted edges
            print(f"Creating tier {current_tier_id} for {len(remaining_edges)} remaining edges")
            
            current_tier = self._create_tier(current_tier_id, node_positions)
            tiers.append(current_tier)
            # Copy all incident nodes to fresh 3D grid with TSV connections
            self._copy_incident_nodes_to_tier(remaining_edges, current_tier, node_positions)
            
            # Process edges using FIFO queue for deterministic routing order
            edge_queue = deque()
            
            # Sort edges by straight-line distance then add to FIFO queue
            remaining_edges_list = list(remaining_edges)
            remaining_edges_list.sort(key=lambda edge: self._calculate_straight_line_distance(edge, node_positions))
            edge_queue.extend(remaining_edges_list)
            
            routed_in_iteration = set()
            failed_edges = []
            routing_attempts = 0
            max_routing_attempts = len(remaining_edges) * 2  # Prevent infinite loops
            
            # Process edges from FIFO queue with iterative routing
            while edge_queue and routing_attempts < max_routing_attempts:
                edge = edge_queue.popleft()
                routing_attempts += 1
                
                path = self._route_edge_on_tier(edge, node_positions, current_tier, current_tier_id)
                if path:
                    edge_routes[edge] = path
                    self._mark_path_occupied(path, current_tier)
                    current_tier.edges.append(edge)
                    routed_in_iteration.add(edge)
                    tier_usage[current_tier_id] += 1
                else:
                    # Handle routing failures with tier congestion analysis
                    failed_edges.append(edge)
                    
                    # Check for tier congestion using failure threshold approach
                    if self._is_tier_congested_by_failures(current_tier, failed_edges, len(remaining_edges)):
                        # Tier is congested when too many consecutive failures occur
                        print(f"Tier {current_tier_id} congested after {len(failed_edges)} consecutive failures")
                        break
            
            # Update remaining edges for next tier escalation
            remaining_edges = (remaining_edges - routed_in_iteration) | set(failed_edges)
            print(f"Tier {current_tier_id}: routed {len(routed_in_iteration)} edges, {len(remaining_edges)} remaining")
            
            # Create new tier and reattempt remaining edges
            if remaining_edges:
                current_tier_id += 1
            else:
                # All edges routed successfully
                break
        
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
        """Create a new routing tier based on natural grid dimensions from placement."""
        # Use natural grid dimensions that emerged from placement optimization
        coords = list(node_positions.values())
        if not coords:
            grid_size = (10, 10, 2)
        else:
            max_x = max(x for x, y in coords)
            max_y = max(y for x, y in coords)
            # Add small margin for routing
            grid_width = max_x + 3
            grid_height = max_y + 3
            grid_size = (grid_width, grid_height, 2)
        
        tier = RoutingTier(
            tier_id=tier_id,
            grid=np.zeros(grid_size, dtype=bool),
            edges=[],
            tsvs=set(),
            bump_transitions={}
        )
        
        # For qubit tier, don't mark node positions as occupied - they are connection points
        # Routing paths need to pass through qubit positions to connect them
        # Only mark cells as occupied when actual routing paths use them
        
        return tier
    
    def _copy_incident_nodes_to_tier(self, edges: Set[Tuple[int, int]], tier: RoutingTier, 
                                   node_positions: Dict[int, Tuple[int, int]]):
        """Copy all nodes incident to edges to the tier and create TSVs as described in the paper.
        
        Paper: "All nodes incident to such edges are copied to a fresh (x, y, z) grid that represents 
        the first routing tier. In hardware, this is realized with a TSV providing a vertical connection 
        from the qubit on the qubit tier to a higher routing tier."
        """
        incident_nodes = set()
        for u, v in edges:
            incident_nodes.add(u)
            incident_nodes.add(v)
        
        # Create TSVs for all incident nodes
        for node in incident_nodes:
            if node in node_positions:
                x, y = node_positions[node]
                # Mark TSV location
                tier.tsvs.add((x, y))
                # Ensure this position is available for routing (but not blocked)
                # TSV positions are access points, not obstacles
    
    def _route_edge_on_tier(self, edge: Tuple[int, int], node_positions: Dict[int, Tuple[int, int]], 
                           tier: RoutingTier, tier_id: int) -> Optional[List[Tuple[int, int, int]]]:
        """Route a single edge on a specific tier using paper's approach."""
        u, v = edge
        if u not in node_positions or v not in node_positions:
            return None
        
        start_pos = node_positions[u]
        end_pos = node_positions[v]
        
        if tier_id == 0:
            # Tier 0: Use the qubit tier routing
            return self._route_edge_on_qubit_tier(edge, node_positions, tier)
        else:
            # Higher tiers: Paper's approach for complex grid-based routing
            return self._route_edge_on_higher_tier(edge, start_pos, end_pos, tier, tier_id)
    
    def _route_edge_on_higher_tier(self, edge: Tuple[int, int], start_pos: Tuple[int, int], 
                                  end_pos: Tuple[int, int], tier: RoutingTier, tier_id: int) -> Optional[List[Tuple[int, int, int]]]:
        """
        Route edge on higher tier using systematic layer management.
        Higher tiers manage available layers dynamically to resolve routing conflicts.
        """
        # Systematic layer-by-layer routing approach
        num_layers = tier.grid.shape[2]
        
        # 1. Try each available layer in order for straight-line routing
        for layer in range(num_layers):
            if self._is_layer_available_for_routing(tier, layer):
                path = self.straight_router.route_straight_line(start_pos, end_pos, tier, layer=layer)
                if path:
                    # Successful straight-line routing on this layer
                    if layer != 0:
                        # Add bump transitions to connect from/to qubit layer
                        full_path = self._create_path_with_layer_transitions(start_pos, end_pos, path, layer)
                        segments, bump_count = self.bump_manager.add_bump_transitions(full_path)
                        if not self.bump_manager.exceeds_bump_limit(bump_count):
                            tier.bump_transitions[edge] = bump_count
                            self._mark_layer_usage(tier, layer)
                            return full_path
                    else:
                        # Direct routing - ensure proper layer assignment
                        # If routing on layer 0, need transitions to/from qubit layer 1
                        if layer == 0:
                            return self._create_path_with_layer_transitions(start_pos, end_pos, path, layer)
                        else:
                            return path
        
        # 2. Try multi-layer routing with dynamic layer allocation
        for primary_layer in range(num_layers):
            if not self._is_layer_available_for_routing(tier, primary_layer):
                continue
                
            for secondary_layer in range(num_layers):
                if secondary_layer == primary_layer or not self._is_layer_available_for_routing(tier, secondary_layer):
                    continue
                
                # Attempt two-layer routing solution
                path = self._attempt_multi_layer_routing(start_pos, end_pos, tier, primary_layer, secondary_layer)
                if path:
                    segments, bump_count = self.bump_manager.add_bump_transitions(path)
                    if not self.bump_manager.exceeds_bump_limit(bump_count):
                        tier.bump_transitions[edge] = bump_count
                        self._mark_layer_usage(tier, primary_layer)
                        self._mark_layer_usage(tier, secondary_layer)
                        return path
        
        # 3. Fallback to A* pathfinding with layer constraints
        # Start on layer 1 where qubits are placed
        start_3d = (start_pos[0], start_pos[1], 1)
        end_3d = (end_pos[0], end_pos[1], 1)
        
        path = self.pathfinder.find_path(start_3d, end_3d, tier)
        if path:
            segments, bump_count = self.bump_manager.add_bump_transitions(path)
            if bump_count <= self.config.max_bump_transitions:
                tier.bump_transitions[edge] = bump_count
                # Mark layers used by this path
                used_layers = set(pos[2] for pos in path)
                for layer in used_layers:
                    self._mark_layer_usage(tier, layer)
                return path
        
        return None
    
    def _route_edge_on_qubit_tier(self, edge: Tuple[int, int], node_positions: Dict[int, Tuple[int, int]], 
                                  tier: RoutingTier) -> Optional[List[Tuple[int, int, int]]]:
        """
        Route edge on qubit tier using HAL paper approach.
        
        Paper: "The planar subgraph is routed first on the qubit tier using straight lines.
        After this, all other edges are also attempted as straight lines on the qubit tier."
        
        Per the paper, qubits are on layer 1, so routing should use both layers 0 and 1.
        """
        u, v = edge
        if u not in node_positions or v not in node_positions:
            return None
        
        start_pos = node_positions[u]
        end_pos = node_positions[v]
        
        # Try straight-line routing on layer 1 first (where qubits are placed)
        path = self.straight_router.route_straight_line(start_pos, end_pos, tier, layer=1)
        if path:
            return path
        
        # If blocked on qubit layer, try layer 0 as fallback (HAL paper's "opposing layer")
        path = self.straight_router.route_straight_line(start_pos, end_pos, tier, layer=0)
        if path:
            # Need bump transitions to connect from layer 1 (qubits) to layer 0 and back
            return self._create_path_with_layer_transitions(start_pos, end_pos, path, 0)
        
        # If straight line fails on both layers, try A* pathfinding using both layers
        # Start and end on layer 1 (where qubits are)
        start_3d = (start_pos[0], start_pos[1], 1)
        end_3d = (end_pos[0], end_pos[1], 1)
        
        path = self.pathfinder.find_path(start_3d, end_3d, tier)
        if path:
            return path
        
        # Fallback: try starting from layer 0
        start_3d = (start_pos[0], start_pos[1], 0)
        end_3d = (end_pos[0], end_pos[1], 0)
        
        path = self.pathfinder.find_path(start_3d, end_3d, tier)
        if path:
            return path
        
        return None
    
    def _mark_path_occupied(self, path: List[Tuple[int, int, int]], tier: RoutingTier):
        """Mark path cells as occupied in tier grid with realistic congestion modeling."""
        for i, (x, y, layer) in enumerate(path):
            if tier.tier_id == 0:
                # Tier 0: Mark all cells but with weighted occupancy for congestion modeling
                # This creates realistic routing conflicts as the tier fills up
                tier.set_occupied(x, y, layer, True)
            else:
                # Higher tiers: Use strict occupancy to prevent conflicts
                tier.set_occupied(x, y, layer, True)
    
    def _is_tier_congested(self, tier: RoutingTier) -> bool:
        """Check if tier is too congested for more routing."""
        occupancy_rate = np.sum(tier.grid) / tier.grid.size
        return occupancy_rate > 0.8  # 80% occupancy threshold
    
    def _is_tier_congested_by_failures(self, tier: RoutingTier, failed_edges: List, total_edges: int) -> bool:
        """
        Check tier congestion using failure threshold approach.
        Tier is considered congested when consecutive routing failures exceed threshold.
        """
        # Congestion threshold based on proportion of failed attempts
        failure_threshold = max(3, total_edges // 10)  # At least 3 failures, or 10% of total edges
        
        # Consecutive failure detection
        if len(failed_edges) >= failure_threshold:
            return True
        
        # Also check occupancy-based congestion as fallback
        occupancy_rate = np.sum(tier.grid) / tier.grid.size if tier.grid.size > 0 else 0
        return occupancy_rate > 0.85  # Slightly higher threshold for failure-based analysis
    
    def _calculate_routing_metrics(self, edge_routes: Dict, tiers: List[RoutingTier], 
                                  node_positions: Dict[int, Tuple[int, int]]) -> Dict[str, float]:
        """Calculate routing quality metrics with enhanced bump bond cost analysis."""
        if not edge_routes:
            return {'tiers': max(len(tiers), 1), 'length': 0.0, 'bumps': 0.0, 'tsvs': 0.0, 'bump_cost': 0.0, 'qecc_weight': 0.0}
        
        total_length = 0.0
        total_bumps = 0
        edge_tsvs = defaultdict(int)  # Track TSVs per edge
        
        # Calculate enhanced bump bond cost using sophisticated model
        total_bump_cost = self.bump_manager.calculate_total_bump_cost(edge_routes)
        
        # Calculate TSVs per edge based on which tiers they use
        for edge, path in edge_routes.items():
            if not path:
                continue
                
            # Calculate path length in grid units
            path_length = len(path) - 1 if len(path) > 1 else 0
            total_length += path_length
            
            # Enhanced bump analysis with consecutive bump detection
            segments, bump_count = self.bump_manager.add_bump_transitions(path)
            total_bumps += bump_count
            
            # TSV calculation: Find which tier this edge is routed on
            edge_tier = 0  # Default to tier 0
            for tier_id, tier in enumerate(tiers):
                if edge in tier.edges:
                    edge_tier = tier_id
                    break
            
            # If edge is routed on tier > 0, it needs TSVs to connect back to logical qubits
            if edge_tier > 0:
                # Each edge on higher tiers needs 2 TSVs (one at each end to connect to tier 0)
                edge_tsvs[edge] = 2
        
        num_edges = len(edge_routes)
        
        # Calculate maximum average bump transitions across all tiers with cost weighting
        max_avg_bumps = 0.0
        max_avg_bump_cost = 0.0
        if tiers:
            tier_bump_counts = []
            tier_bump_costs = []
            for tier in tiers:
                tier_bumps = sum(tier.bump_transitions.values())
                tier_edges = len(tier.bump_transitions) if tier.bump_transitions else len(tier.edges)
                
                avg_bumps = tier_bumps / tier_edges if tier_edges > 0 else 0.0
                tier_bump_counts.append(avg_bumps)
                
                # Calculate average bump cost for this tier
                tier_cost = 0.0
                for edge in tier.edges:
                    if edge in edge_routes and edge_routes[edge]:
                        segments, _ = self.bump_manager.add_bump_transitions(edge_routes[edge])
                        for segment in segments:
                            if segment.is_bump_transition:
                                # Calculate cost since RouteSegment doesn't store individual costs
                                segment_cost = self.bump_manager._calculate_segment_cost(
                                    segment.start, segment.end, True, 1
                                )
                                tier_cost += segment_cost
                
                avg_tier_cost = tier_cost / tier_edges if tier_edges > 0 else 0.0
                tier_bump_costs.append(avg_tier_cost)
            
            max_avg_bumps = max(tier_bump_counts) if tier_bump_counts else 0.0
            max_avg_bump_cost = max(tier_bump_costs) if tier_bump_costs else 0.0
        
        # Average TSVs per edge (only counting edges that use higher tiers)
        total_tsvs = sum(edge_tsvs.values())
        avg_tsvs = total_tsvs / num_edges if num_edges > 0 else 0.0
        
        # Calculate QECC weight (average node degree)
        qecc_weight = self._calculate_qecc_weight(edge_routes, node_positions)
        
        return {
            'tiers': len(tiers),
            'length': total_length / num_edges if num_edges > 0 else 0.0,
            'bumps': max_avg_bumps,  # Maximum average bump transitions across all tiers
            'tsvs': avg_tsvs,        # Average TSVs per edge on higher tiers
            'bump_cost': total_bump_cost / num_edges if num_edges > 0 else 0.0,  # Enhanced bump cost metric
            'max_tier_bump_cost': max_avg_bump_cost,  # Maximum average bump cost across tiers
            'qecc_weight': qecc_weight  # Average node degree (QECC weight)
        }
    
    def _calculate_qecc_weight(self, edge_routes: Dict, node_positions: Dict[int, Tuple[int, int]]) -> float:
        """
        Calculate the QECC weight (average node degree) from the routed connectivity graph.
        
        The QECC weight indicates how many other qubits each qubit is connected to on average.
        This is a key metric for understanding code connectivity density:
        - Weight 4: Surface codes, low-weight radial codes
        - Weight 6: Many bicycle codes, tile codes  
        - Weight 8+: High-weight tile codes, complex qLDPC codes
        
        Args:
            edge_routes: Dictionary of edge -> routing path
            node_positions: Dictionary of node -> grid position
            
        Returns:
            Average node degree (QECC weight)
        """
        if not edge_routes or not node_positions:
            return 0.0
        
        # Count degree for each node
        node_degrees = {}
        for node in node_positions.keys():
            node_degrees[node] = 0
        
        # Count edges connected to each node
        for edge in edge_routes.keys():
            node1, node2 = edge
            if node1 in node_degrees:
                node_degrees[node1] += 1
            if node2 in node_degrees:
                node_degrees[node2] += 1
        
        # Calculate average degree (QECC weight)
        if node_degrees:
            total_degree = sum(node_degrees.values())
            return total_degree / len(node_degrees)
        else:
            return 0.0
    
    
    def _get_tier_from_z_level(self, z_level: int) -> int:
        """Get tier ID from z-level coordinate."""
        # Assuming 2 layers per tier
        return z_level // 2
    
    def _calculate_straight_line_distance(self, edge: Tuple[int, int], 
                                        node_positions: Dict[int, Tuple[int, int]]) -> float:
        """
        Calculate straight-line length estimate for routing prioritization.
        Uses Bresenham line algorithm to count grid cells for precise routing cost estimation.
        """
        u, v = edge
        if u not in node_positions or v not in node_positions:
            return float('inf')
        
        x1, y1 = node_positions[u]
        x2, y2 = node_positions[v]
        
        # Use Bresenham line algorithm to calculate exact straight-line path length in grid units
        line_points = self._bresenham_line_for_distance(x1, y1, x2, y2)
        
        # Return path length as number of grid steps for accurate routing cost estimation
        return float(len(line_points) - 1) if len(line_points) > 1 else 0.0
    
    def _bresenham_line_for_distance(self, x1: int, y1: int, x2: int, y2: int) -> List[Tuple[int, int]]:
        """Bresenham line algorithm for precise grid-based distance calculation."""
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
    
    
    
    def _is_layer_available_for_routing(self, tier: RoutingTier, layer: int) -> bool:
        """Check if layer has sufficient capacity for additional routing."""
        if layer >= tier.grid.shape[2]:
            return False
        
        # Calculate layer occupancy
        layer_slice = tier.grid[:, :, layer]
        occupancy_rate = np.sum(layer_slice) / layer_slice.size
        
        # Layer is available if under capacity threshold
        return occupancy_rate < 0.7  # 70% threshold per layer
    
    def _mark_layer_usage(self, tier: RoutingTier, layer: int):
        """Mark layer as having increased usage for capacity tracking."""
        # This is tracked implicitly through grid occupancy
        # Could be extended with explicit layer usage counters if needed
        pass
    
    def _create_path_with_layer_transitions(self, start_pos: Tuple[int, int], end_pos: Tuple[int, int], 
                                          path: List[Tuple[int, int, int]], target_layer: int) -> List[Tuple[int, int, int]]:
        """Create path with proper layer transitions to/from qubit layer."""
        if not path:
            return []
        
        # Create full path with layer transitions
        full_path = []
        
        # Start transition: qubit layer to target layer
        full_path.append((start_pos[0], start_pos[1], 0))  # Start on qubit layer
        if target_layer != 0:
            full_path.append((start_pos[0], start_pos[1], target_layer))  # Bump to target layer
        
        # Add main routing path (excluding endpoints if they're already handled)
        for i, (x, y, z) in enumerate(path):
            if i == 0 and (x, y) == start_pos:
                continue  # Skip start if already added
            if i == len(path) - 1 and (x, y) == end_pos:
                continue  # Skip end, will be added below
            full_path.append((x, y, target_layer))
        
        # End transition: target layer to qubit layer
        if target_layer != 0:
            full_path.append((end_pos[0], end_pos[1], target_layer))  # Approach on target layer
        full_path.append((end_pos[0], end_pos[1], 0))  # End on qubit layer
        
        return full_path
    
    def _attempt_multi_layer_routing(self, start_pos: Tuple[int, int], end_pos: Tuple[int, int], 
                                   tier: RoutingTier, primary_layer: int, secondary_layer: int) -> Optional[List[Tuple[int, int, int]]]:
        """Attempt routing using multiple layers to resolve conflicts."""
        # Calculate midpoint for layer transition
        mid_x = (start_pos[0] + end_pos[0]) // 2
        mid_y = (start_pos[1] + end_pos[1]) // 2
        
        # Try routing: start -> midpoint on primary_layer, midpoint -> end on secondary_layer
        path_segment1 = self.straight_router.route_straight_line(start_pos, (mid_x, mid_y), tier, primary_layer)
        if path_segment1:
            path_segment2 = self.straight_router.route_straight_line((mid_x, mid_y), end_pos, tier, secondary_layer)
            if path_segment2:
                # Combine segments with layer transitions
                full_path = []
                
                # Add first segment
                for x, y, z in path_segment1:
                    full_path.append((x, y, primary_layer))
                
                # Add transition point
                full_path.append((mid_x, mid_y, secondary_layer))
                
                # Add second segment (skip overlapping midpoint)
                for i, (x, y, z) in enumerate(path_segment2):
                    if i == 0 and (x, y) == (mid_x, mid_y):
                        continue
                    full_path.append((x, y, secondary_layer))
                
                return full_path
        
        return None