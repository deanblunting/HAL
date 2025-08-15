"""
Placement engine for HAL algorithm with spring layout and rasterization.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
from scipy.optimize import minimize
import heapq
from collections import defaultdict

from graph_utils import CommunityDetector, PlanarityTester, GraphAnalyzer
from data_structures import PlacementResult
from config import HALConfig


class SpringLayout:
    """Kamada-Kawai spring layout optimization."""
    
    def __init__(self, graph: nx.Graph, config: HALConfig):
        self.graph = graph
        self.config = config
        self.analyzer = GraphAnalyzer(graph)
        
    def compute_layout(self, planar_subgraph_edges: Set[Tuple[int, int]]) -> Dict[int, Tuple[float, float]]:
        """Compute spring layout positions for nodes."""
        nodes = list(self.graph.nodes())
        n = len(nodes)
        
        if n == 0:
            return {}
        if n == 1:
            return {nodes[0]: (0.0, 0.0)}
        
        # Get shortest path distances
        distances = self.analyzer.compute_all_pairs_shortest_paths()
        
        # Create planar subgraph for distance calculations
        planar_graph = nx.Graph()
        planar_graph.add_nodes_from(self.graph.nodes())
        planar_graph.add_edges_from(planar_subgraph_edges)
        
        # Compute k parameter (average distance scaling)
        if self.config.spring_layout_k is None:
            total_dist = sum(distances.get((u, v), float('inf')) for u in nodes for v in nodes if u != v and distances.get((u, v), float('inf')) != float('inf'))
            k = np.sqrt(1.0 / n) if total_dist == 0 else np.sqrt(1.0 / n) * np.sqrt(total_dist / (n * (n - 1)))
        else:
            k = self.config.spring_layout_k
        
        # Initialize positions randomly
        np.random.seed(self.config.random_seed)
        initial_pos = np.random.rand(n, 2) * 2 - 1  # Random positions in [-1, 1]
        
        # Optimize using Kamada-Kawai energy function
        def energy_function(pos):
            pos = pos.reshape(n, 2)
            energy = 0.0
            
            for i, u in enumerate(nodes):
                for j, v in enumerate(nodes):
                    if i >= j:
                        continue
                        
                    d_uv = distances.get((u, v), float('inf'))
                    if d_uv == float('inf') or d_uv == 0:
                        continue
                    
                    euclidean_dist = np.linalg.norm(pos[i] - pos[j])
                    ideal_dist = k * d_uv
                    
                    # Spring energy: (actual_distance - ideal_distance)^2
                    energy += (euclidean_dist - ideal_dist) ** 2
            
            return energy
        
        try:
            result = minimize(
                energy_function, 
                initial_pos.flatten(),
                method='L-BFGS-B',
                options={'maxiter': self.config.spring_layout_iterations}
            )
            
            optimized_pos = result.x.reshape(n, 2)
        except:
            # Fallback to initial positions if optimization fails
            optimized_pos = initial_pos
        
        # Convert to dictionary
        positions = {}
        for i, node in enumerate(nodes):
            positions[node] = (float(optimized_pos[i, 0]), float(optimized_pos[i, 1]))
        
        return positions


class GridRasterizer:
    """Convert continuous positions to integer grid coordinates."""
    
    def __init__(self, config: HALConfig):
        self.config = config
        
    def rasterize_positions(self, positions: Dict[int, Tuple[float, float]]) -> Dict[int, Tuple[int, int]]:
        """Convert floating-point positions to integer grid coordinates."""
        if not positions:
            return {}
        
        # Normalize positions to fit in grid
        pos_array = np.array(list(positions.values()))
        min_pos = np.min(pos_array, axis=0)
        max_pos = np.max(pos_array, axis=0)
        
        # Scale to fit in grid with margin
        margin = 5
        available_size = (
            self.config.grid_size[0] - 2 * margin,
            self.config.grid_size[1] - 2 * margin
        )
        
        range_pos = max_pos - min_pos
        # Avoid division by zero
        range_pos[range_pos == 0] = 1.0
        
        scale = min(available_size[0] / range_pos[0], available_size[1] / range_pos[1])
        
        # Phase 1: Naive rounding
        grid_positions = {}
        occupied = set()
        conflicts = []
        
        nodes = list(positions.keys())
        for node in nodes:
            x, y = positions[node]
            # Normalize and scale
            x_norm = (x - min_pos[0]) * scale + margin
            y_norm = (y - min_pos[1]) * scale + margin
            
            # Round to integer
            grid_x = int(round(x_norm))
            grid_y = int(round(y_norm))
            
            # Clamp to grid bounds
            grid_x = max(0, min(self.config.grid_size[0] - 1, grid_x))
            grid_y = max(0, min(self.config.grid_size[1] - 1, grid_y))
            
            if (grid_x, grid_y) not in occupied:
                grid_positions[node] = (grid_x, grid_y)
                occupied.add((grid_x, grid_y))
            else:
                conflicts.append((node, positions[node], (grid_x, grid_y)))
        
        # Phase 2: Resolve conflicts using priority queue
        if conflicts:
            grid_positions.update(self._resolve_conflicts(conflicts, occupied))
        
        return grid_positions
    
    def _resolve_conflicts(self, conflicts: List, occupied: Set[Tuple[int, int]]) -> Dict[int, Tuple[int, int]]:
        """Resolve placement conflicts using nearest available position."""
        resolved = {}
        
        for node, original_pos, target_grid in conflicts:
            best_pos = self._find_nearest_free_position(target_grid, occupied)
            if best_pos:
                resolved[node] = best_pos
                occupied.add(best_pos)
            else:
                # Fallback: place at original target (allowing overlap)
                resolved[node] = target_grid
        
        return resolved
    
    def _find_nearest_free_position(self, target: Tuple[int, int], occupied: Set[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Find nearest free position to target using BFS."""
        target_x, target_y = target
        queue = [(0, target_x, target_y)]  # (distance, x, y)
        visited = {target}
        
        max_search_radius = 20  # Limit search to avoid infinite loops
        
        while queue:
            dist, x, y = heapq.heappop(queue)
            
            if dist > max_search_radius:
                break
                
            if (x, y) not in occupied and 0 <= x < self.config.grid_size[0] and 0 <= y < self.config.grid_size[1]:
                return (x, y)
            
            # Add neighbors
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                new_x, new_y = x + dx, y + dy
                if (new_x, new_y) not in visited:
                    visited.add((new_x, new_y))
                    new_dist = abs(new_x - target_x) + abs(new_y - target_y)
                    heapq.heappush(queue, (new_dist, new_x, new_y))
        
        return None
    
    def compact_grid(self, positions: Dict[int, Tuple[int, int]]) -> Dict[int, Tuple[int, int]]:
        """Compact grid by removing empty rows and columns."""
        if not positions:
            return positions
        
        coords = list(positions.values())
        min_x = min(x for x, y in coords)
        min_y = min(y for x, y in coords)
        
        # Shift all positions to start from (0, 0)
        compacted = {}
        for node, (x, y) in positions.items():
            compacted[node] = (x - min_x, y - min_y)
        
        return compacted


class PlacementEngine:
    """Main placement engine combining all placement algorithms."""
    
    def __init__(self, config: HALConfig):
        self.config = config
        self.spring_layout = None
        self.rasterizer = GridRasterizer(config)
        
    def place_nodes(self, graph: nx.Graph, custom_positions: Optional[Dict[int, Tuple[int, int]]] = None) -> PlacementResult:
        """
        Place nodes using either custom positions or spring layout.
        
        Args:
            graph: Input connectivity graph
            custom_positions: Optional pre-specified positions
            
        Returns:
            PlacementResult containing node positions and metadata
        """
        if custom_positions:
            # Use custom positions directly
            node_positions = custom_positions.copy()
            planar_edges = set(graph.edges())  # Assume all edges can be planar with custom positions
            communities = {node: 0 for node in graph.nodes()}  # Single community
        else:
            # Use algorithmic placement
            node_positions, planar_edges, communities = self._algorithmic_placement(graph)
        
        # Compute grid bounds
        if node_positions:
            coords = list(node_positions.values())
            bounds = (
                min(x for x, y in coords),
                max(x for x, y in coords),
                min(y for x, y in coords),
                max(y for x, y in coords)
            )
        else:
            bounds = (0, 0, 0, 0)
        
        return PlacementResult(
            node_positions=node_positions,
            planar_subgraph_edges=planar_edges,
            grid_bounds=bounds,
            communities=communities
        )
    
    def _algorithmic_placement(self, graph: nx.Graph) -> Tuple[Dict[int, Tuple[int, int]], Set[Tuple[int, int]], Dict[int, int]]:
        """Perform algorithmic placement using community detection and spring layout."""
        
        # Step 1: Community detection (enhanced with Louvain as per paper)
        community_detector = CommunityDetector(graph, self.config)
        communities = community_detector.detect_communities()
        
        # Step 2: Extract planar subgraph
        analyzer = GraphAnalyzer(graph)
        edge_priorities = analyzer.get_edge_priorities(communities)
        
        planarity_tester = PlanarityTester(graph)
        planar_edges = planarity_tester.get_planar_subgraph(edge_priorities)
        
        # Step 3: Spring layout on planar subgraph
        self.spring_layout = SpringLayout(graph, self.config)
        continuous_positions = self.spring_layout.compute_layout(planar_edges)
        
        # Step 4: Rasterize to integer grid
        grid_positions = self.rasterizer.rasterize_positions(continuous_positions)
        
        # Step 5: Compact grid
        final_positions = self.rasterizer.compact_grid(grid_positions)
        
        return final_positions, planar_edges, communities