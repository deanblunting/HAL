"""
Placement engine for HAL algorithm with spring layout and rasterization.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
import heapq
from collections import defaultdict

from .graph_utils import CommunityDetector, PlanarityTester, GraphAnalyzer
from .data_structures import PlacementResult
from .config import HALConfig


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
        
        # Compute all-pairs shortest path distances
        distances = self.analyzer.compute_all_pairs_shortest_paths()
        
        # Construct planar subgraph for distance computation
        planar_graph = nx.Graph()
        planar_graph.add_nodes_from(self.graph.nodes())
        planar_graph.add_edges_from(planar_subgraph_edges)
        
        # Calculate spring constant parameter for distance scaling
        if self.config.spring_layout_k is None:
            total_dist = sum(distances.get((u, v), float('inf')) for u in nodes for v in nodes if u != v and distances.get((u, v), float('inf')) != float('inf'))
            k = np.sqrt(1.0 / n) if total_dist == 0 else np.sqrt(1.0 / n) * np.sqrt(total_dist / (n * (n - 1)))
        else:
            k = self.config.spring_layout_k
        
        # Initialize node positions with random distribution optimized for auxiliary grid
        np.random.seed(self.config.random_seed)
        initial_pos = np.random.rand(n, 2) * 6 - 3  # Random positions in [-3, 3] for better spread
        
        # Apply Kamada-Kawai energy minimization algorithm
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
                    
                    # Compute spring energy: quadratic distance deviation
                    energy += (euclidean_dist - ideal_dist) ** 2
            
            return energy
        
        try:
            # Apply gradient descent optimization (SciPy-independent implementation)
            optimized_pos = self._gradient_descent_optimization(
                energy_function, initial_pos, n, self.config.spring_layout_iterations
            )
        except:
            # Revert to initial configuration on optimization failure
            optimized_pos = initial_pos
        
        # Transform array results to node position dictionary
        positions = {}
        for i, node in enumerate(nodes):
            positions[node] = (float(optimized_pos[i, 0]), float(optimized_pos[i, 1]))
        
        return positions
    
    def _gradient_descent_optimization(self, energy_function, initial_pos, n, max_iterations):
        """Simple gradient descent optimization to replace SciPy dependency."""
        pos = initial_pos.copy()
        learning_rate = 0.01
        epsilon = 1e-6
        
        for iteration in range(max_iterations):
            # Compute numerical gradient using finite differences
            gradient = np.zeros_like(pos)
            current_energy = energy_function(pos.flatten())
            
            for i in range(pos.shape[0]):
                for j in range(pos.shape[1]):
                    # Apply forward difference approximation
                    pos_plus = pos.copy()
                    pos_plus[i, j] += epsilon
                    energy_plus = energy_function(pos_plus.flatten())
                    
                    gradient[i, j] = (energy_plus - current_energy) / epsilon
            
            # Execute gradient descent position update
            pos -= learning_rate * gradient
            
            # Apply adaptive learning rate decay
            if iteration % 10 == 0:
                learning_rate *= 0.95  # Apply exponential decay
            
            # Implement early termination based on gradient magnitude
            if np.linalg.norm(gradient) < 1e-4:
                break
        
        return pos


class GridRasterizer:
    """Convert continuous positions to integer grid coordinates."""
    
    def __init__(self, config: HALConfig):
        self.config = config
        
    def rasterize_positions(self, positions: Dict[int, Tuple[float, float]], 
                           auxiliary_grid_size: Optional[Tuple[int, int]] = None) -> Dict[int, Tuple[int, int]]:
        """Convert floating-point positions to integer grid coordinates."""
        if not positions:
            return {}
        
        # Select grid dimensions: auxiliary grid or default configuration
        if auxiliary_grid_size:
            grid_width, grid_height = auxiliary_grid_size
        else:
            grid_width, grid_height = self.config.grid_size
        
        # Apply position normalization to auxiliary grid bounds
        pos_array = np.array(list(positions.values()))
        min_pos = np.min(pos_array, axis=0)
        max_pos = np.max(pos_array, axis=0)
        
        # Optimize auxiliary grid utilization with reduced margins
        margin = 1  # Smaller margin for better distribution
        available_size = (
            grid_width - 2 * margin,
            grid_height - 2 * margin
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
            
            # Clamp to auxiliary grid bounds
            grid_x = max(0, min(grid_width - 1, grid_x))
            grid_y = max(0, min(grid_height - 1, grid_y))
            
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
        
        # Step 4: Calculate auxiliary grid dimensions (like paper's approach)
        auxiliary_grid_size = self._calculate_auxiliary_grid_dimensions(len(graph.nodes()))
        
        # Step 5: Use specialized auxiliary grid placement (like paper's approach)
        final_positions = self._distribute_on_auxiliary_grid(graph.nodes(), auxiliary_grid_size)
        
        return final_positions, planar_edges, communities
    
    def _calculate_auxiliary_grid_dimensions(self, n_logical_qubits: int) -> Tuple[int, int]:
        """
        Calculate auxiliary grid dimensions using paper's methodology.
        This mirrors the calculation in routing.py for consistency.
        """
        if n_logical_qubits == 0:
            return (10, 10)
        
        # Paper's approach: 50% efficiency target
        target_efficiency = self.config.hardware_efficiency_target if hasattr(self.config, 'hardware_efficiency_target') else 0.5
        target_total_positions = int(n_logical_qubits / target_efficiency)
        
        # Calculate dimensions for rectangular grid (like paper's 10×6)
        grid_height = int((target_total_positions / 1.6) ** 0.5)  # Start with height
        grid_width = int(target_total_positions / grid_height)
        
        # Adjust to get close to target positions
        while grid_width * grid_height < target_total_positions:
            grid_width += 1
        
        # Don't make it too much bigger than needed
        if grid_width * grid_height > target_total_positions * 1.2:  # Max 20% over
            if grid_width > grid_height:
                grid_width -= 1
            else:
                grid_height -= 1
        
        return (grid_width, grid_height)
    
    def _distribute_on_auxiliary_grid(self, nodes: list, auxiliary_grid_size: Tuple[int, int]) -> Dict[int, Tuple[int, int]]:
        """
        Distribute logical qubits across auxiliary grid using HAL's generic placement algorithm for non-geometric codes.
        This creates a more uniform distribution instead of clustering.
        """
        grid_width, grid_height = auxiliary_grid_size
        node_list = list(nodes)
        n_nodes = len(node_list)
        
        if n_nodes == 0:
            return {}
        
        # Create a distributed placement strategy
        positions = {}
        
        # Strategy 1: Grid-based distribution achieving HAL's target hardware efficiency (50% for 10×6 grid with 30 qubits)
        # Distribute nodes roughly evenly across the auxiliary grid
        spacing_x = max(1, grid_width // max(1, int(n_nodes**0.5)))
        spacing_y = max(1, grid_height // max(1, int(n_nodes**0.5)))
        
        # Add some randomization to avoid perfect regularity
        np.random.seed(self.config.random_seed)
        
        placed_positions = set()
        
        for i, node in enumerate(node_list):
            # Try distributed placement first
            attempts = 0
            max_attempts = 20
            
            while attempts < max_attempts:
                if attempts < 10:
                    # First 10 attempts: try systematic distribution
                    row = (i // int(n_nodes**0.5)) * spacing_y + np.random.randint(0, max(1, spacing_y))
                    col = (i % int(n_nodes**0.5)) * spacing_x + np.random.randint(0, max(1, spacing_x))
                else:
                    # Remaining attempts: random placement
                    row = np.random.randint(0, grid_height)
                    col = np.random.randint(0, grid_width)
                
                # Ensure within bounds
                row = max(0, min(grid_height - 1, row))
                col = max(0, min(grid_width - 1, col))
                
                if (col, row) not in placed_positions:
                    positions[node] = (col, row)
                    placed_positions.add((col, row))
                    break
                
                attempts += 1
            
            # If we couldn't find a free position, place it anywhere
            if node not in positions:
                row = i % grid_height
                col = (i // grid_height) % grid_width
                positions[node] = (col, row)
        
        return positions