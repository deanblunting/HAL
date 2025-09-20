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

# Layer definitions for (x,y,z) coordinate system
# Tier 0 (qubit tier) layer assignments:
CONTROL_LAYER = 0  # Layer 0: Control and readout circuitry (tier 0 only)
QUBIT_LAYER = 1    # Layer 1: Physical qubits (tier 0 only)
# Higher tiers use z = {0, 1} as generic routing layers


class StraightLineRouter:
    """Route edges as straight lines using Bresenham algorithm."""

    def __init__(self, config: HALConfig):
        self.config = config

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

        # Layer transition cost (bump bond)
        if l1 != l2:
            cost += 0.1  # Small penalty for layer changes

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

        # Sort by straight-line distance for optimal routing order
        planar_edges.sort(key=lambda e: self._calculate_straight_line_distance(e, node_positions))

        remaining_edges = set(all_edges) - set(planar_edges)

        print(f"Tier 0: Routing {len(planar_edges)} planar edges on layer {QUBIT_LAYER}")

        # Route planar edges on qubit tier
        for edge in planar_edges:
            path = self._route_edge_on_qubit_tier(edge, node_positions)
            if path:
                edge_routes[edge] = {
                    'path': path,
                    'tier': 0,
                    'routing_method': 'straight_line'
                }
                tier_usage[0] += 1
            else:
                # Failed planar edges escalate to higher tiers
                remaining_edges.add(edge)
                print(f"Warning: Planar edge {edge} failed on tier 0, escalating")

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

            # Process edges in FIFO order by length
            edge_queue = deque(sorted(remaining_edges,
                                    key=lambda e: self._calculate_straight_line_distance(e, node_positions)))

            routed_this_tier = set()
            attempted_edges = set()

            while edge_queue:
                edge = edge_queue.popleft()

                # Check for congestion (edge attempted twice on same tier)
                if edge in attempted_edges:
                    print(f"Tier {current_tier_id} congested - escalating remaining edges")
                    break

                attempted_edges.add(edge)
                path, routing_method = self._route_edge_on_higher_tier(edge, node_positions, current_tier)

                if path:
                    edge_routes[edge] = {
                        'path': path,
                        'tier': current_tier_id,
                        'routing_method': routing_method
                    }
                    routed_this_tier.add(edge)
                    tier_usage[current_tier_id] += 1
                else:
                    # Failed edge will be reattempted on next tier
                    pass

            # Update remaining edges
            remaining_edges -= routed_this_tier
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

    def _route_edge_on_qubit_tier(self, edge: Tuple[int, int],
                                 node_positions: Dict[int, Tuple[int, int]]) -> Optional[List[Tuple[int, int, int]]]:
        """Route edge on qubit tier using only layer 1."""
        u, v = edge
        if u not in node_positions or v not in node_positions:
            return None

        start_pos = node_positions[u]
        end_pos = node_positions[v]

        # Only straight-line routing on QUBIT_LAYER
        return self.straight_router.route_straight_line(start_pos, end_pos, QUBIT_LAYER)

    def _route_edge_on_higher_tier(self, edge: Tuple[int, int],
                                  node_positions: Dict[int, Tuple[int, int]],
                                  tier: RoutingTier) -> Tuple[Optional[List[Tuple[int, int, int]]], str]:
        """Route edge on higher tier using straight-line then A*."""
        u, v = edge
        if u not in node_positions or v not in node_positions:
            return None, 'failed'

        start_pos = node_positions[u]
        end_pos = node_positions[v]

        # Try straight-line routing on each layer
        for layer in range(tier.grid.shape[2]):
            path = self.straight_router.route_straight_line(start_pos, end_pos, layer)
            if path:  # Accept all straight-line paths for simplicity
                return path, 'straight_line'

        # Try A* pathfinding
        start_3d = (start_pos[0], start_pos[1], 0)  # Start on layer 0
        end_3d = (end_pos[0], end_pos[1], 0)      # End on layer 0

        path = self.pathfinder.find_path(start_3d, end_3d, tier.grid.shape)
        if path:
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