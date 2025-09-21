"""
Clean routing engine for HAL algorithm with multi-tier pathfinding.
Implements efficient routing using only edge_routes storage.
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


class StraightLineRouter:
    """Route edges as straight lines with bump transitions for intersections."""

    def __init__(self, config: HALConfig):
        self.config = config
        self.crossing_detector = CrossingDetector()

    def route_straight_line(self, start: Tuple[int, int], end: Tuple[int, int],
                           layer: int) -> List[Tuple[int, int, int]]:
        """
        Route edge using straight-line routing on specified layer.

        Args:
            start: (x, y) start position
            end: (x, y) end position
            layer: Layer to route on

        Returns:
            List of (x, y, layer) positions forming path
        """
        return [(start[0], start[1], layer), (end[0], end[1], layer)]

    def route_with_bump_transitions(self, start: Tuple[int, int], end: Tuple[int, int],
                                   existing_routes: Dict, tier: 'RoutingTier') -> Optional[List[Tuple[int, int, int]]]:
        """
        Route edge with bump transitions to avoid intersections.

        Args:
            start: (x, y) start position
            end: (x, y) end position
            existing_routes: Dictionary of already routed edges
            tier: Current routing tier

        Returns:
            List of (x, y, layer) positions forming path with bump transitions, or None if exceeds limit
        """
        # Separate existing segments by layer
        segments_by_layer = {0: [], 1: []}

        for route_info in existing_routes.values():
            if route_info.get('tier') == tier.tier_id and route_info.get('path'):
                route_path = route_info['path']
                for i in range(len(route_path) - 1):
                    p1 = route_path[i]
                    p2 = route_path[i + 1]
                    # Only consider segments on same layer
                    if p1[2] == p2[2]:
                        segment = ((p1[0], p1[1]), (p2[0], p2[1]))
                        if p1[2] in segments_by_layer:
                            segments_by_layer[p1[2]].append(segment)

        # Build path with iterative bump transitions
        path = []
        current_layer = 0
        current_pos = start
        remaining_end = end
        bump_count = 0

        path.append((start[0], start[1], current_layer))

        # Iteratively route segments, switching layers at crossings
        while current_pos != remaining_end and bump_count < self.config.max_bump_transitions:
            # Current segment to route
            current_segment = ((current_pos[0], current_pos[1]), (remaining_end[0], remaining_end[1]))

            # Check for intersections on current layer
            test_segments = segments_by_layer[current_layer] + [current_segment]
            layer_segments = {current_layer: test_segments}
            intersections = self.crossing_detector.find_intersections_by_layer(layer_segments)

            # Find intersections on our current segment
            segment_intersections = []
            for ix, iy in intersections.get(current_layer, []):
                if self._point_on_segment(ix, iy, current_pos, remaining_end):
                    # Exclude start/end points
                    if not (abs(ix - current_pos[0]) < 1e-6 and abs(iy - current_pos[1]) < 1e-6) and \
                       not (abs(ix - remaining_end[0]) < 1e-6 and abs(iy - remaining_end[1]) < 1e-6):
                        segment_intersections.append((ix, iy))

            if not segment_intersections:
                # No intersections, route directly to end
                path.append((remaining_end[0], remaining_end[1], current_layer))
                break

            # Sort by distance from current position
            segment_intersections.sort(key=lambda p: ((p[0] - current_pos[0])**2 + (p[1] - current_pos[1])**2)**0.5)

            # Take the closest intersection - use exact bentley-ottmann coordinates
            ix, iy = segment_intersections[0]

            # Route to bump position on current layer using exact intersection coordinates
            path.append((ix, iy, current_layer))

            # Add bump transition (switch layer) at exact same coordinates
            current_layer = 1 - current_layer
            path.append((ix, iy, current_layer))
            bump_count += 1

            # Update current position to exact bump location
            current_pos = (ix, iy)

        # Check if we exceeded bump limit
        if bump_count >= self.config.max_bump_transitions and current_pos != remaining_end:
            return None

        return path

    def _point_on_segment(self, px: float, py: float, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """Check if point lies on line segment."""
        x1, y1 = start
        x2, y2 = end

        # Check if point is on line (within tolerance)
        tolerance = 1e-6
        cross_product = abs((py - y1) * (x2 - x1) - (px - x1) * (y2 - y1))
        if cross_product > tolerance:
            return False

        # Check if point is within segment bounds
        dot_product = (px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)
        squared_distance = (x2 - x1) ** 2 + (y2 - y1) ** 2

        if squared_distance == 0:
            return abs(px - x1) < tolerance and abs(py - y1) < tolerance

        parameter = dot_product / squared_distance
        return 0 <= parameter <= 1


class AStarPathfinder:
    """A* pathfinding algorithm for 3D grid routing."""

    def __init__(self, config: HALConfig):
        self.config = config

    def find_path(self, start: Tuple[int, int, int], end: Tuple[int, int, int],
                  grid_size: Tuple[int, int, int]) -> Optional[List[Tuple[int, int, int]]]:
        """
        Find path from start to end using A* algorithm.

        Args:
            start: (x, y, layer) starting position
            end: (x, y, layer) ending position
            grid_size: (width, height, num_layers) grid dimensions

        Returns:
            List of (x, y, layer) positions forming path, or None if no path found
        """
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

            # Check if we reached the target
            if current[:2] == end[:2]:  # Allow flexible layer matching
                return self._reconstruct_path(came_from, current)

            for neighbor in self._get_neighbors(current, grid_size):
                if neighbor in visited:
                    continue

                # Compute movement cost
                tentative_g = g_score[current] + self._movement_cost(current, neighbor)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], tentative_g, neighbor))

        return None  # No path found

    def _heuristic(self, pos1: Tuple[int, int, int], pos2: Tuple[int, int, int]) -> float:
        """Euclidean distance heuristic."""
        x1, y1, l1 = pos1
        x2, y2, l2 = pos2
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def _movement_cost(self, pos1: Tuple[int, int, int], pos2: Tuple[int, int, int]) -> float:
        """Cost of moving from pos1 to pos2."""
        x1, y1, l1 = pos1
        x2, y2, l2 = pos2

        # Base movement cost
        cost = 1.0

        # Diagonal movement cost
        if abs(x1 - x2) + abs(y1 - y2) > 1:
            cost *= 1.414  # sqrt(2) for diagonal moves

        # Layer transition cost (bump bond) - no penalty for layer changes
        # if l1 != l2:
        #     cost += 0.1  # Small penalty for layer changes

        return cost

    def _get_neighbors(self, pos: Tuple[int, int, int],
                      grid_size: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Get valid neighboring positions in 8-connected grid + layer transitions."""
        x, y, layer = pos
        width, height, num_layers = grid_size
        neighbors = []

        # 8-connected movement within same layer
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < width and 0 <= new_y < height:
                    neighbors.append((new_x, new_y, layer))

        # Layer transitions (bump bonds)
        for new_layer in range(num_layers):
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


class RoutingEngine:
    """Main routing engine implementing clean HAL algorithm."""

    def __init__(self, config: HALConfig):
        self.config = config
        self.straight_router = StraightLineRouter(config)
        self.pathfinder = AStarPathfinder(config)
        self.crossing_detector = CrossingDetector()

    def route_edges(self, graph: nx.Graph, node_positions: Dict[int, Tuple[int, int]],
                   planar_subgraph_edges: Set[Tuple[int, int]]) -> RoutingResult:
        """
        Route all edges across multiple tiers using clean HAL algorithm.

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

        # Initialize tier 0 (qubit tier)
        qubit_tier = self._create_tier(0, node_positions)
        tiers.append(qubit_tier)

        all_edges = list(graph.edges())

        # Step 1: Route planar edges on tier 0 (layer 1 only)
        planar_edges = [edge for edge in all_edges if edge in planar_subgraph_edges or
                       (edge[1], edge[0]) in planar_subgraph_edges]

        # Initialize FIFO with all edges, sorted by straight-line distance
        # Put planar edges first, then non-planar edges
        planar_edges.sort(key=lambda e: self._calculate_straight_line_distance(e, node_positions))
        non_planar_edges = [e for e in all_edges if e not in planar_edges]
        non_planar_edges.sort(key=lambda e: self._calculate_straight_line_distance(e, node_positions))

        edge_queue = deque(planar_edges + non_planar_edges)

        print(f"Tier 0: Processing {len(edge_queue)} edges ({len(planar_edges)} planar, {len(non_planar_edges)} non-planar)")

        # Route edges on qubit tier using straight-line only
        attempted_edges = set()
        while edge_queue:
            edge = edge_queue.popleft()

            # Check for tier congestion (edge attempted twice)
            if edge in attempted_edges:
                print(f"Tier 0 congested - edge {edge} already attempted")
                # Put this edge and all remaining back for next tier
                edge_queue.appendleft(edge)
                break

            attempted_edges.add(edge)

            # Only try straight-line routing on tier 0 (no bump transitions)
            path = self._route_edge_on_qubit_tier_simple(edge, node_positions, edge_routes)
            if path:
                edge_routes[edge] = {
                    'path': path,
                    'tier': 0,
                    'routing_method': 'straight_line'
                }
                tier_usage[0] += 1
            else:
                # Failed edge goes back to queue for next tier
                edge_queue.append(edge)

        remaining_edges = set(edge_queue)
        print(f"Tier 0: Routed {tier_usage[0]} edges, {len(remaining_edges)} remaining")

        # Step 2: Route remaining edges on higher tiers
        current_tier_id = 1

        while remaining_edges and current_tier_id < self.config.max_tiers:
            print(f"Creating tier {current_tier_id} for {len(remaining_edges)} edges")

            # Create new tier
            current_tier = self._create_tier(current_tier_id, node_positions)
            tiers.append(current_tier)

            # Add TSVs for incident nodes
            self._add_tsvs_for_edges(remaining_edges, current_tier, node_positions)

            # Process edges from previous tier's queue + any failed edges
            # Sort by straight-line distance (FIFO order by length)
            edge_queue = deque(sorted(remaining_edges,
                                    key=lambda e: self._calculate_straight_line_distance(e, node_positions)))

            routed_this_tier = set()
            attempted_edges = set()
            failed_edges = set()

            while edge_queue:
                edge = edge_queue.popleft()

                # Check for tier congestion (edge attempted twice on same tier)
                if edge in attempted_edges:
                    print(f"Tier {current_tier_id} congested - escalating edge {edge} to next tier")
                    # Put edge back for next tier
                    failed_edges.add(edge)
                    continue

                attempted_edges.add(edge)
                u, v = edge
                path, routing_method = self._route_edge_on_higher_tier(edge, node_positions, current_tier, edge_routes)

                if path:
                    edge_routes[edge] = {
                        'path': path,
                        'tier': current_tier_id,
                        'routing_method': routing_method
                    }

                    # Track bump transitions in tier
                    bump_transitions = self._extract_bump_transitions(path)
                    for bump_pos in bump_transitions:
                        current_tier.bump_transitions[bump_pos] = edge

                    routed_this_tier.add(edge)
                    tier_usage[current_tier_id] += 1
                else:
                    # Failed edge goes back to end of queue for retry on this tier
                    edge_queue.append(edge)

            # Combine remaining edges from queue and failed edges for next tier
            remaining_edges = set(edge_queue) | failed_edges
            print(f"Tier {current_tier_id}: Routed {len(routed_this_tier)} edges")

            current_tier_id += 1

        # Mark any remaining edges as unrouted
        unrouted_edges = remaining_edges

        # Calculate metrics
        metrics = self._calculate_metrics(edge_routes, tiers, tier_usage)

        return RoutingResult(
            edge_routes=edge_routes,
            tiers=tiers,
            unrouted_edges=unrouted_edges,
            metrics=metrics
        )

    def _create_tier(self, tier_id: int, node_positions: Dict[int, Tuple[int, int]]) -> RoutingTier:
        """Create a new routing tier."""
        coords = list(node_positions.values())
        if not coords:
            grid_size = (10, 10, 2)
        else:
            max_x = max(x for x, y in coords)
            max_y = max(y for x, y in coords)
            grid_size = (max_x + 3, max_y + 3, 2)  # Small margin

        return RoutingTier(
            tier_id=tier_id,
            grid=np.zeros(grid_size, dtype=bool),
            edges=[],
            tsvs=set(),
            bump_transitions={}
        )

    def _route_edge_on_qubit_tier_simple(self, edge: Tuple[int, int],
                                        node_positions: Dict[int, Tuple[int, int]],
                                        existing_routes: Dict) -> Optional[List[Tuple[int, int, int]]]:
        """Route edge on qubit tier using only straight-line on layer 1 with crossing detection."""
        u, v = edge
        if u not in node_positions or v not in node_positions:
            return None

        start_pos = node_positions[u]
        end_pos = node_positions[v]

        # Simple straight-line routing on QUBIT_LAYER
        proposed_path = self.straight_router.route_straight_line(start_pos, end_pos, QUBIT_LAYER)

        # Check for crossings with existing routes - if crossing detected, fail this edge
        if self._would_cross_existing_routes(proposed_path, existing_routes, 0):
            return None  # Edge fails and goes to FIFO queue for higher tier

        return proposed_path

    def _route_edge_on_higher_tier(self, edge: Tuple[int, int],
                                  node_positions: Dict[int, Tuple[int, int]],
                                  tier: RoutingTier, existing_routes: Dict) -> Tuple[Optional[List[Tuple[int, int, int]]], str]:
        """Route edge on higher tier using straight-line with bump transitions then A*."""
        u, v = edge
        if u not in node_positions or v not in node_positions:
            return None, 'failed'

        start_pos = node_positions[u]
        end_pos = node_positions[v]

        # Try straight-line routing with bump transitions
        path = self.straight_router.route_with_bump_transitions(start_pos, end_pos, existing_routes, tier)
        if path:
            return path, 'straight_line_bumps'

        # Try A* pathfinding with vertical moves
        start_3d = (start_pos[0], start_pos[1], 0)  # Start on layer 0
        end_3d = (end_pos[0], end_pos[1], 0)      # End on layer 0

        path = self.pathfinder.find_path(start_3d, end_3d, tier.grid.shape)
        if path:
            # Validate A* path doesn't create crossings
            if self._would_cross_existing_routes(path, existing_routes, tier.tier_id):
                return None, 'failed'  # A* path would create crossings
            return path, 'astar'
        else:
            return None, 'failed'

    def _add_tsvs_for_edges(self, edges: Set[Tuple[int, int]], tier: RoutingTier,
                           node_positions: Dict[int, Tuple[int, int]]):
        """Add TSVs for all nodes incident to edges."""
        incident_nodes = set()
        for u, v in edges:
            incident_nodes.add(u)
            incident_nodes.add(v)

        for node in incident_nodes:
            if node in node_positions:
                x, y = node_positions[node]
                tier.tsvs.add((x, y))

    def _would_cross_existing_routes(self, proposed_path: List[Tuple[int, int, int]],
                                   existing_routes: Dict, tier_id: int) -> bool:
        """Check if proposed path would cross any existing routes on the same tier."""
        if not proposed_path or len(proposed_path) < 2:
            return False

        # Get existing segments on the same tier, grouped by layer
        segments_by_layer = {0: [], 1: []}

        for route_info in existing_routes.values():
            if route_info.get('tier') == tier_id and route_info.get('path'):
                route_path = route_info['path']
                for i in range(len(route_path) - 1):
                    p1 = route_path[i]
                    p2 = route_path[i + 1]
                    if p1[2] == p2[2]:  # Same layer segment
                        segment = ((p1[0], p1[1]), (p2[0], p2[1]))
                        if p1[2] in segments_by_layer:
                            segments_by_layer[p1[2]].append(segment)

        # Convert proposed path to segments by layer
        proposed_segments_by_layer = {0: [], 1: []}
        for i in range(len(proposed_path) - 1):
            p1 = proposed_path[i]
            p2 = proposed_path[i + 1]
            if p1[2] == p2[2]:  # Same layer segment
                segment = ((p1[0], p1[1]), (p2[0], p2[1]))
                if p1[2] in proposed_segments_by_layer:
                    proposed_segments_by_layer[p1[2]].append(segment)

        # Check for intersections on each layer
        for layer in [0, 1]:
            if proposed_segments_by_layer[layer] and segments_by_layer[layer]:
                # Check if proposed segments cross existing segments on this layer
                proposed_segments = proposed_segments_by_layer[layer]
                existing_segments = segments_by_layer[layer]

                # Test proposed segments against existing segments
                test_segments = existing_segments + proposed_segments
                layer_segments = {layer: test_segments}
                intersections = self.crossing_detector.find_intersections_by_layer(layer_segments)

                if intersections.get(layer, []):
                    return True  # Found crossing

        return False

    def _extract_bump_transitions(self, path: List[Tuple[int, int, int]]) -> List[Tuple[int, int]]:
        """Extract bump transition positions from path."""
        bump_positions = []
        for i in range(len(path) - 1):
            if path[i][2] != path[i + 1][2]:  # Layer change
                bump_positions.append((path[i][0], path[i][1]))
        return bump_positions

    def _calculate_straight_line_distance(self, edge: Tuple[int, int],
                                        node_positions: Dict[int, Tuple[int, int]]) -> float:
        """Calculate straight-line distance for edge ordering."""
        u, v = edge
        if u not in node_positions or v not in node_positions:
            return float('inf')

        x1, y1 = node_positions[u]
        x2, y2 = node_positions[v]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def _calculate_metrics(self, edge_routes: Dict, tiers: List[RoutingTier],
                          tier_usage: Dict[int, int]) -> Dict[str, float]:
        """Calculate routing quality metrics."""
        if not edge_routes:
            return {'tiers': len(tiers), 'length': 0.0, 'tsvs': 0.0, 'bumps': 0.0}

        total_length = 0.0
        total_bumps = 0
        total_tsvs = 0

        for edge, route_info in edge_routes.items():
            if not route_info or not route_info.get('path'):
                continue

            path = route_info['path']

            # Path length
            total_length += len(path) - 1 if len(path) > 1 else 0

            # Count bump transitions (layer changes within path)
            for i in range(len(path) - 1):
                if path[i][2] != path[i + 1][2]:  # Layer change
                    total_bumps += 1

        # TSVs based on tier usage (edges on higher tiers need TSVs)
        for tier_id, edge_count in tier_usage.items():
            if tier_id > 0:
                total_tsvs += edge_count * 2  # 2 TSVs per edge on higher tiers

        num_edges = len(edge_routes)

        return {
            'tiers': len(tiers),
            'length': (total_length / num_edges / self.config.qubit_spacing) if num_edges > 0 else 0.0,
            'bumps': total_bumps / num_edges if num_edges > 0 else 0.0,
            'tsvs': total_tsvs / num_edges if num_edges > 0 else 0.0
        }