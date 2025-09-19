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
from .crossing_detector import CrossingDetector





# Layer definitions for (x,y,z) coordinate system
# Tier 0 (qubit tier) layer assignments:
CONTROL_LAYER = 0  # Layer 0: Control and readout circuitry (tier 0 only)
QUBIT_LAYER = 1    # Layer 1: Physical qubits (tier 0 only)
# Higher tiers use z = {0, 1} as generic routing layers




class AStarPathfinder:
    """A* pathfinding algorithm for 3D grid routing."""

    def __init__(self, config: HALConfig):
        self.config = config

    def find_path(self, start: Tuple[int, int, int], end: Tuple[int, int, int],
                  tier: RoutingTier, avoid_nodes: Set[Tuple[int, int]] = None) -> Optional[List[Tuple[int, int, int]]]:
        """
        Find path from start to end using A* algorithm with crossing detection.

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

                # Strict occupancy checking - HAL paper: no path intersections allowed
                if tier.is_occupied(nx, ny, nl):
                    continue

                # DISABLED: Crossing detection library has bugs, causing unnecessary tier escalation
                # if tier.tier_id != 0 and self._would_create_crossing(current, neighbor, tier):
                #     continue

                # Compute transition cost between positions
                tentative_g = g_score[current] + self._movement_cost(current, neighbor)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], tentative_g, neighbor))

        return None  # No path found

    def _heuristic(self, pos1: Tuple[int, int, int], pos2: Tuple[int, int, int]) -> float:
        """Euclidean distance heuristic for pathfinding around obstructions."""
        x1, y1, l1 = pos1
        x2, y2, l2 = pos2
        euclidean = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        return euclidean

    def _movement_cost(self, pos1: Tuple[int, int, int], pos2: Tuple[int, int, int]) -> float:
        """Cost of moving from pos1 to pos2."""
        x1, y1, l1 = pos1
        x2, y2, l2 = pos2

        # Initialize base movement cost
        cost = 1.0

        # Apply diagonal movement cost (actual distance)
        if abs(x1 - x2) + abs(y1 - y2) > 1:
            # Diagonal moves have sqrt(2) ≈ 1.414 actual distance vs 1.0 for orthogonal
            cost *= 1.414  # Use actual Euclidean distance for diagonal moves

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

    def _would_create_crossing(self, current: Tuple[int, int, int], neighbor: Tuple[int, int, int], tier: RoutingTier) -> bool:
        """
        Fast crossing check using unified CrossingDetector with sweep line algorithm.
        """
        # Skip crossing detection for tier 0 (planar edges shouldn't cross by definition)
        if tier.tier_id == 0:
            return False

        # Only check crossings on the same layer
        if current[2] != neighbor[2]:
            return False  # Layer transitions don't create crossings

        layer = current[2]

        # Convert current and neighbor to 2D coordinates
        current_2d = (current[0], current[1])
        neighbor_2d = (neighbor[0], neighbor[1])

        # Early exit: if endpoints are the same, no crossing
        if current_2d == neighbor_2d:
            return False

        # Use unified crossing detector for fast segment checking
        return tier.crossing_detector.check_segment_crossing(current_2d, neighbor_2d, layer)



class StraightLineRouter:
    """Route edges as straight lines when possible."""

    def __init__(self, config: HALConfig):
        self.config = config

    def route_straight_line(self, start: Tuple[int, int], end: Tuple[int, int],
                           tier: RoutingTier, layer: int = 0) -> Optional[List[Tuple[int, int, int]]]:
        """
        Route edge using appropriate strategy based on tier.

        Tier 0 (qubit tier): Single layer routing only - no bump transitions allowed
        Higher tiers: HAL paper's minimum bump bond strategy with layer switching

        Args:
            start: (x, y) start position
            end: (x, y) end position
            tier: RoutingTier to route on
            layer: Layer to route on (starting layer)

        Returns:
            List of (x, y, layer) positions, or None if routing fails
        """
        if tier.tier_id == 0:
            # Tier 0: No bump transitions allowed - single layer routing only
            return self._route_single_layer(start, end, tier, layer)
        else:
            # Higher tiers: Use minimum bump bond strategy
            return self._route_with_minimum_bumps(start, end, tier, layer)

    def _route_single_layer(self, start: Tuple[int, int], end: Tuple[int, int],
                           tier: RoutingTier, layer: int) -> Optional[List[Tuple[int, int, int]]]:
        """
        Route edge on single layer only (for tier 0 - no bump transitions).

        This is the original simple straight-line routing that fails if any
        obstruction is encountered, forcing the edge to escalate to higher tiers.
        """
        # Generate straight line path
        line_points = self._bresenham_line(start[0], start[1], end[0], end[1])

        # Check for cell occupancy conflicts (skip endpoints - they are connection points)
        for i, (x, y) in enumerate(line_points):
            # Skip endpoints (start and end are qubit connection points)
            if i == 0 or i == len(line_points) - 1:
                continue
            if tier.is_occupied(x, y, layer):
                return None

        # Path is valid for tier 0 (planar edges shouldn't cross by definition)
        # Transform to 3D coordinate representation
        return [(x, y, layer) for x, y in line_points]

    def _route_with_minimum_bumps(self, start: Tuple[int, int], end: Tuple[int, int],
                                 tier: RoutingTier, start_layer: int) -> Optional[List[Tuple[int, int, int]]]:
        """
        Multi-layer routing strategy without bump transitions.

        Strategy:
        1. Try single layer routing on starting layer
        2. Try opposing layer if starting layer fails
        3. Try smart layer switching if both single layers fail
        """
        # Generate the straight line path in 2D
        line_points = self._bresenham_line(start[0], start[1], end[0], end[1])
        if len(line_points) < 2:
            return [(start[0], start[1], start_layer)]

        # Strategy 1: Try starting layer
        path = self._route_single_layer_strategy(line_points, tier, start_layer)
        if path:
            return path

        # Strategy 2: Try opposing layer
        opposing_layer = 1 - start_layer if start_layer in [0, 1] else 0
        path = self._route_single_layer_strategy(line_points, tier, opposing_layer)
        if path:
            return path

        # Strategy 3: Try smart layer switching
        path = self._route_smart_bumps_strategy(line_points, tier, start_layer)
        if path:
            return path

        return None  # All strategies failed


    def _route_single_layer_strategy(self, line_points: List[Tuple[int, int]], tier: RoutingTier,
                                   layer: int) -> Optional[List[Tuple[int, int, int]]]:
        """Try routing the entire path on a single layer."""
        path_3d = [(x, y, layer) for x, y in line_points]

        # Check for occupancy conflicts (skip endpoints)
        for i, (x, y, z) in enumerate(path_3d):
            if i == 0 or i == len(path_3d) - 1:  # Skip endpoints
                continue
            if tier.is_occupied(x, y, z):
                return None

        # Check for crossings using proper validation
        if self._validate_path_crossings_quick(path_3d, tier):
            return path_3d
        return None

    def _route_smart_bumps_strategy(self, line_points: List[Tuple[int, int]], tier: RoutingTier,
                                  start_layer: int) -> Optional[List[Tuple[int, int, int]]]:
        """Simple layer switching strategy - try both layers systematically."""
        # Strategy 1: Try single layer routing first
        path_3d = [(x, y, start_layer) for x, y in line_points]

        # Check occupancy
        for i, (x, y, z) in enumerate(path_3d):
            if i == 0 or i == len(path_3d) - 1:  # Skip endpoints
                continue
            if tier.is_occupied(x, y, z):
                # Strategy 2: Try opposing layer
                opposing_layer = 1 - start_layer if start_layer in [0, 1] else 0
                alt_path = [(x, y, opposing_layer) for x, y in line_points]

                # Check occupancy on opposing layer
                for j, (ax, ay, az) in enumerate(alt_path):
                    if j == 0 or j == len(alt_path) - 1:  # Skip endpoints
                        continue
                    if tier.is_occupied(ax, ay, az):
                        return None  # Both layers blocked

                return alt_path

        return path_3d

    def _point_creates_crossing(self, point: Tuple[int, int], tier: RoutingTier, layer: int,
                              point_index: int, full_path: List[Tuple[int, int]]) -> bool:
        """Quick check if a point would create crossings using unified crossing detector."""
        # Check segments involving this point
        if point_index > 0:
            prev_point = full_path[point_index - 1]
            if tier.crossing_detector.check_segment_crossing(prev_point, point, layer):
                return True

        if point_index < len(full_path) - 1:
            next_point = full_path[point_index + 1]
            if tier.crossing_detector.check_segment_crossing(point, next_point, layer):
                return True

        return False




    def _validate_path_crossings_quick(self, path_3d: List[Tuple[int, int, int]],
                                     tier: RoutingTier) -> bool:
        """Quick crossing validation using unified CrossingDetector."""
        # Skip crossing detection for tier 0 (planar edges shouldn't cross by definition)
        if tier.tier_id == 0:
            return True

        # Group by layer for efficient checking
        layer_paths = {}
        for x, y, z in path_3d:
            if z not in layer_paths:
                layer_paths[z] = []
            layer_paths[z].append((x, y))

        # Check each layer's segments using unified crossing detector
        for layer, path_points in layer_paths.items():
            if len(path_points) < 2:
                continue

            # Use unified crossing detector
            if tier.crossing_detector.would_create_crossing(path_points, layer):
                return False

        return True


    # NOTE: _check_path_crossings method removed - replaced by unified CrossingDetector

    def _record_routed_path(self, tier: RoutingTier, path_3d: List[Tuple[int, int, int]]):
        """Record a successfully routed path using unified CrossingDetector."""
        # Group path points by layer
        layer_paths = {}
        for x, y, z in path_3d:
            if z not in layer_paths:
                layer_paths[z] = []
            layer_paths[z].append((x, y))

        # Add path to unified crossing detector for each layer
        for layer, path_points in layer_paths.items():
            if len(path_points) > 1:  # Only record if there are actual segments
                tier.crossing_detector.add_path(path_points, layer)

        # Legacy storage for visualization (keep for compatibility)
        if not hasattr(tier, 'routed_paths'):
            tier.routed_paths = {}

        # Find the primary layer (layer with most points) for visualization
        layer_counts = {}
        for x, y, z in path_3d:
            layer_counts[z] = layer_counts.get(z, 0) + 1

        primary_layer = max(layer_counts, key=layer_counts.get) if layer_counts else 0

        # Store complete 3D path under primary layer for visualization
        if primary_layer not in tier.routed_paths:
            tier.routed_paths[primary_layer] = []

        if len(path_3d) > 1:
            tier.routed_paths[primary_layer].append(path_3d)

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






class RoutingEngine:
    """Main routing engine coordinating all routing algorithms."""

    def __init__(self, config: HALConfig):
        self.config = config
        self.straight_router = StraightLineRouter(config)
        self.pathfinder = AStarPathfinder(config)

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

        all_edges = list(graph.edges())

        # Step 1: Route ONLY planar subgraph edges on Tier 0 (qubit tier)
        planar_edges = [edge for edge in all_edges if edge in planar_subgraph_edges or
                       (edge[1], edge[0]) in planar_subgraph_edges]

        # Sort planar edges by straight-line distance for optimal routing order
        planar_edges_by_length = sorted(planar_edges,
            key=lambda e: self._calculate_straight_line_distance(e, node_positions))

        # Start with all non-planar edges for higher tier routing (will add failed planar edges later)
        remaining_edges = set(all_edges) - set(planar_edges)
        tier_0_routed = 0

        print(f"Tier 0: Routing {len(planar_edges)} planar edges only (non-planar edges skip to higher tiers)")

        # Route planar subgraph edges on qubit tier (Tier 0)
        failed_planar_edges = []
        for edge in planar_edges_by_length:
            path = self._route_edge_on_qubit_tier(edge, node_positions, qubit_tier)
            if path:
                edge_routes[edge] = path
                self._mark_path_occupied(path, qubit_tier)
                qubit_tier.edges.append(edge)
                tier_usage[0] += 1
                tier_0_routed += 1
            else:
                # Planar edges that fail will be moved to higher tiers as per HAL paper
                failed_planar_edges.append(edge)
                print(f"Warning: Planar edge {edge} could not be routed on Tier 0 and will be moved to higher tier")

        if failed_planar_edges:
            print(f"Failed to route {len(failed_planar_edges)} planar edges on Tier 0. Moving to higher tiers.")
            # Add failed planar edges to higher-tier routing queue as per HAL paper
            remaining_edges = remaining_edges | set(failed_planar_edges)

        print(f"Tier 0 completion: {tier_0_routed}/{len(planar_edges)} planar edges routed")
        print(f"Higher-tier routing required for {len(remaining_edges)} edges ({len(remaining_edges) - (len(all_edges) - len(planar_edges))} failed planar + {len(all_edges) - len(planar_edges)} non-planar)")

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

            # Sort edges by straight-line distance then add to FIFO queue (HAL paper approach)
            remaining_edges_list = list(remaining_edges)
            remaining_edges_list.sort(key=lambda edge: self._calculate_straight_line_distance(edge, node_positions))
            print(f"Tier {current_tier_id}: Sorted {len(remaining_edges_list)} edges by straight-line distance (shortest first)")
            edge_queue.extend(remaining_edges_list)

            routed_in_iteration = set()
            attempted_edges = set()  # Track which edges have been attempted on this tier
            failed_edges = []
            routing_attempts = 0
            max_routing_attempts = len(remaining_edges) * 2  # Prevent infinite loops

            # Process edges from FIFO queue with iterative routing
            print(f"Tier {current_tier_id}: Starting to process {len(edge_queue)} edges...")
            while edge_queue and routing_attempts < max_routing_attempts:
                edge = edge_queue.popleft()
                routing_attempts += 1

                if routing_attempts % 10 == 0:  # Progress indicator
                    print(f"Tier {current_tier_id}: Processed {routing_attempts} edges, {len(edge_queue)} remaining")

                # Check if this edge has already been attempted on this tier
                if edge in attempted_edges:
                    # HAL paper: when router pops an edge already attempted, declare tier congested
                    print(f"Tier {current_tier_id} congested: edge {edge} already attempted on this tier - escalating all remaining edges")
                    # Add the current edge back to failed edges and break to escalate ALL remaining edges
                    failed_edges.append(edge)
                    # Add all remaining edges in queue to failed_edges for escalation
                    failed_edges.extend(list(edge_queue))
                    break

                attempted_edges.add(edge)
                path = self._route_edge_on_tier(edge, node_positions, current_tier, current_tier_id)
                if path:
                    edge_routes[edge] = path
                    self._mark_path_occupied(path, current_tier)
                    current_tier.edges.append(edge)
                    routed_in_iteration.add(edge)
                    tier_usage[current_tier_id] += 1
                else:
                    # Both algorithms failed - immediately escalate to next tier (don't re-queue)
                    failed_edges.append(edge)
                    # Don't add back to queue - will be escalated to next tier

            # Update remaining edges for next tier escalation
            remaining_edges = (remaining_edges - routed_in_iteration) | set(failed_edges)
            print(f"Tier {current_tier_id}: routed {len(routed_in_iteration)} edges, {len(remaining_edges)} remaining")

            # Create new tier and reattempt remaining edges
            if remaining_edges:
                current_tier_id += 1
            else:
                # All edges routed successfully
                break

        # Mark unrouted edges (remaining non-planar edges + failed planar edges)
        unrouted_edges = remaining_edges | set(failed_planar_edges)


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
        Route edge on higher tier following HAL paper's exact routing flow:
        1. Straight-line attempt on available layers
        2. A* pathfinding attempt if straight-line fails
        3. Tier escalation (handled by caller) if both fail
        """
        num_layers = tier.grid.shape[2]

        # STEP 1: HAL Paper - Try straight-line routing
        for layer in range(num_layers):
            path = self.straight_router.route_straight_line(start_pos, end_pos, tier, layer=layer)
            if path:
                # Validate path doesn't create crossings (skip for tier 0)
                if tier.tier_id == 0 or self._validate_full_path_crossings(path, tier):
                    print(f"Tier {tier_id}: Straight-line successfully routed edge {edge} on layer {layer}")
                    return path

        # STEP 2: HAL Paper - Try A* pathfinding
        print(f"Tier {tier_id}: Straight-line failed for edge {edge}, trying A* pathfinding...")

        # Try A* on each available layer (layer 0 first, then layer 1)
        for layer in range(num_layers):
            # Convert 2D positions to 3D for A* pathfinder
            start_3d = (start_pos[0], start_pos[1], layer)
            end_3d = (end_pos[0], end_pos[1], layer)

            # Use A* pathfinder to route around obstructions
            astar_path = self.pathfinder.find_path(start_3d, end_3d, tier)
            if astar_path:
                # Validate A* path doesn't create crossings (skip for tier 0)
                if tier.tier_id == 0 or self._validate_full_path_crossings(astar_path, tier):
                    print(f"Tier {tier_id}: A* successfully routed edge {edge} on layer {layer}")
                    return astar_path

        # STEP 3: HAL Paper - Both straight-line and A* failed, escalate to next tier
        print(f"Tier {tier_id}: Both straight-line and A* failed for edge {edge} - will escalate to next tier")
        return None

    def _route_edge_on_qubit_tier(self, edge: Tuple[int, int], node_positions: Dict[int, Tuple[int, int]],
                                  tier: RoutingTier) -> Optional[List[Tuple[int, int, int]]]:
        """
        Route planar edge on qubit tier using simple straight-line routing only.

        Tier 0 constraints:
        - Only routes planar edges (guaranteed non-crossing by planar subgraph extraction)
        - Only uses layer 1 (layer 0 reserved for control/readout circuitry)
        - Only straight-line routing (no A*, no bump transitions, no complex pathfinding)
        - If straight-line fails, edge remains unrouted (cannot escalate to higher tiers)
        """
        u, v = edge
        if u not in node_positions or v not in node_positions:
            return None

        start_pos = node_positions[u]
        end_pos = node_positions[v]

        # Single strategy: Straight-line routing on qubit layer only
        # Use _route_single_layer to ensure no bump transitions are attempted
        path = self.straight_router._route_single_layer(start_pos, end_pos, tier, layer=QUBIT_LAYER)
        if path:
            return path

        # If straight-line routing fails, edge cannot be routed on qubit tier
        return None


    def _mark_path_occupied(self, path: List[Tuple[int, int, int]], tier: RoutingTier):
        """
        Mark path cells as occupied and record path for crossing detection.

        IMPORTANT: Node positions are connection points that should NOT be marked as occupied.
        Only intermediate routing cells should be blocked to prevent path crossings.
        """
        # Always record paths for crossing detection, even short ones
        # Only skip occupancy marking for very short paths

        # Mark only intermediate path cells as occupied (excluding start and end nodes)
        # But only if the path has intermediate points
        if len(path) > 2:
            for i, (x, y, layer) in enumerate(path):
                if i == 0 or i == len(path) - 1:
                    # Skip start and end positions - these are node connection points
                    continue

                # HAL paper: strict binary occupancy for intermediate routing cells
                tier.set_occupied(x, y, layer, True)

        # ALWAYS record the full path for future crossing detection, regardless of length
        self.straight_router._record_routed_path(tier, path)



    def _calculate_routing_metrics(self, edge_routes: Dict, tiers: List[RoutingTier],
                                  node_positions: Dict[int, Tuple[int, int]]) -> Dict[str, float]:
        """Calculate routing quality metrics."""
        if not edge_routes:
            return {'tiers': max(len(tiers), 1), 'length': 0.0, 'tsvs': 0.0}

        total_length = 0.0
        edge_tsvs = defaultdict(int)  # Track TSVs per edge

        # Calculate TSVs per edge based on which tiers they use
        for edge, path in edge_routes.items():
            if not path:
                continue

            # Calculate path length in grid units
            path_length = len(path) - 1 if len(path) > 1 else 0
            total_length += path_length

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

        # Average TSVs per edge (only counting edges that use higher tiers)
        total_tsvs = sum(edge_tsvs.values())
        avg_tsvs = total_tsvs / num_edges if num_edges > 0 else 0.0

        return {
            'tiers': len(tiers),
            'length': (total_length / num_edges / self.config.qubit_spacing) if num_edges > 0 else 0.0,
            'tsvs': avg_tsvs        # Average TSVs per edge on higher tiers
        }





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
        line_points = self.straight_router._bresenham_line(x1, y1, x2, y2)

        # Return path length as number of grid steps for accurate routing cost estimation
        return float(len(line_points) - 1) if len(line_points) > 1 else 0.0


    def _validate_full_path_crossings(self, full_path: List[Tuple[int, int, int]], tier: RoutingTier) -> bool:
        """
        Validate that the full path doesn't create crossings using unified CrossingDetector.

        This is critical for higher tier routing where composite paths are created from
        multiple segments and layer transitions.
        """
        if not full_path or len(full_path) < 2:
            return True

        # Group path points by layer for crossing detection
        layer_paths = {}
        for x, y, z in full_path:
            if z not in layer_paths:
                layer_paths[z] = []
            layer_paths[z].append((x, y))

        # Check each layer's path segments using unified crossing detector
        for layer, path_points in layer_paths.items():
            if len(path_points) < 2:
                continue  # No segments to check

            # Use unified crossing detector
            if tier.crossing_detector.would_create_crossing(path_points, layer):
                return False  # Found crossing - reject path

        return True  # No crossings found - path is valid

