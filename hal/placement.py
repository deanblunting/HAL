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
        
        # Construct planar subgraph for distance computation
        planar_graph = nx.Graph()
        planar_graph.add_nodes_from(self.graph.nodes())
        planar_graph.add_edges_from(planar_subgraph_edges or [])
        
        # Compute distances on planar subgraph if provided, otherwise use full graph
        if planar_subgraph_edges:
            distances = dict(nx.all_pairs_shortest_path_length(planar_graph))
        else:
            distances = self.analyzer.compute_all_pairs_shortest_paths()
        
        # Calculate spring constant parameter for distance scaling
        if self.config.spring_layout_k is None:
            total_dist = sum(distances.get((u, v), float('inf')) for u in nodes for v in nodes if u != v and distances.get((u, v), float('inf')) != float('inf'))
            k = np.sqrt(1.0 / n) if total_dist == 0 else np.sqrt(1.0 / n) * np.sqrt(total_dist / (n * (n - 1)))
        else:
            k = self.config.spring_layout_k
        
        # Initialize node positions with random distribution optimized for auxiliary grid
        np.random.seed(self.config.random_seed)
        initial_pos = np.random.rand(n, 2) * 6 - 3  # Random positions in [-3, 3] for better spread
        
        # Apply exact Kamada-Kawai energy minimization algorithm
        def energy_function(pos):
            """
            Exact Kamada-Kawai global energy function that minimizes squared differences
            between graph-theoretic distances and their corresponding Euclidean distances.
            """
            pos = pos.reshape(n, 2)
            total_energy = 0.0
            
            for i, u in enumerate(nodes):
                for j, v in enumerate(nodes):
                    if i >= j:
                        continue
                        
                    # Graph-theoretic distance between nodes u and v
                    d_ij = distances.get((u, v), float('inf'))
                    if d_ij == float('inf') or d_ij == 0:
                        continue
                    
                    # Current Euclidean distance in layout
                    euclidean_dist = np.linalg.norm(pos[i] - pos[j])
                    
                    # Ideal distance scaled by spring constant
                    l_ij = k * d_ij
                    
                    # Spring constant between nodes i and j
                    k_ij = 1.0 / (d_ij * d_ij) if d_ij > 0 else 0.0
                    
                    # Exact Kamada-Kawai energy: k_ij * (euclidean_dist - l_ij)^2
                    energy_contribution = k_ij * (euclidean_dist - l_ij) ** 2
                    total_energy += energy_contribution
            
            return total_energy
        
        try:
            # Apply conjugate gradient optimization as specified in HAL paper
            optimized_pos = self._conjugate_gradient_optimization(
                energy_function, initial_pos, n, self.config.spring_layout_iterations
            )
        except:
            # Graceful fallback to gradient descent on optimization failure
            optimized_pos = self._gradient_descent_optimization(
                energy_function, initial_pos, n, self.config.spring_layout_iterations
            )
        
        # Transform array results to node position dictionary
        positions = {}
        for i, node in enumerate(nodes):
            positions[node] = (float(optimized_pos[i, 0]), float(optimized_pos[i, 1]))
        
        return positions
    
    def _conjugate_gradient_optimization(self, energy_function, initial_pos, n, max_iterations):
        """Conjugate gradient optimization for Kamada-Kawai energy minimization.
        
        Implements conjugate gradient method as referenced in HAL paper for
        minimizing spring layout energy function.
        """
        pos = initial_pos.copy().flatten()
        
        def compute_gradient(x):
            """Compute gradient using finite differences."""
            grad = np.zeros_like(x)
            epsilon = 1e-6
            base_energy = energy_function(x)
            
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += epsilon
                grad[i] = (energy_function(x_plus) - base_energy) / epsilon
                
            return grad
        
        # Initialize conjugate gradient variables
        gradient = compute_gradient(pos)
        search_direction = -gradient.copy()
        
        for iteration in range(max_iterations):
            # Line search to find optimal step size
            step_size = self._line_search(energy_function, pos, search_direction)
            
            if step_size == 0:
                break
                
            # Update position
            new_pos = pos + step_size * search_direction
            new_gradient = compute_gradient(new_pos)
            
            # Check convergence
            if np.linalg.norm(new_gradient) < 1e-4:
                pos = new_pos
                break
            
            # Compute conjugate coefficient using Polak-Ribiere formula
            beta = max(0, np.dot(new_gradient, new_gradient - gradient) / np.dot(gradient, gradient))
            
            # Update search direction
            search_direction = -new_gradient + beta * search_direction
            
            # Update variables for next iteration
            pos = new_pos
            gradient = new_gradient
        
        return pos.reshape(n, 2)
    
    def _line_search(self, energy_function, pos, direction, alpha_max=1.0, c1=1e-4, max_iter=20):
        """Simple backtracking line search for conjugate gradient method."""
        alpha = alpha_max
        base_energy = energy_function(pos)
        gradient_direction_dot = np.dot(self._compute_simple_gradient(energy_function, pos), direction)
        
        for _ in range(max_iter):
            new_energy = energy_function(pos + alpha * direction)
            
            # Armijo condition for sufficient decrease
            if new_energy <= base_energy + c1 * alpha * gradient_direction_dot:
                return alpha
            
            alpha *= 0.5
        
        return 0.0
    
    def _compute_simple_gradient(self, energy_function, pos):
        """Simple gradient computation for line search."""
        grad = np.zeros_like(pos)
        epsilon = 1e-6
        base_energy = energy_function(pos)
        
        for i in range(len(pos)):
            pos_plus = pos.copy()
            pos_plus[i] += epsilon
            grad[i] = (energy_function(pos_plus) - base_energy) / epsilon
            
        return grad
    
    def _gradient_descent_optimization(self, energy_function, initial_pos, n, max_iterations):
        """Gradient descent fallback optimization method."""
        pos = initial_pos.copy()
        learning_rate = 0.01
        epsilon = 1e-6
        
        for iteration in range(max_iterations):
            # Compute numerical gradient using finite differences
            gradient = np.zeros_like(pos)
            current_energy = energy_function(pos.flatten())
            
            for i in range(pos.shape[0]):
                for j in range(pos.shape[1]):
                    pos_plus = pos.copy()
                    pos_plus[i, j] += epsilon
                    energy_plus = energy_function(pos_plus.flatten())
                    
                    gradient[i, j] = (energy_plus - current_energy) / epsilon
            
            # Update position with gradient descent
            pos -= learning_rate * gradient
            
            # Adaptive learning rate adjustment
            if iteration % 10 == 0:
                learning_rate *= 0.95
            
            # Early termination on convergence
            if np.linalg.norm(gradient) < 1e-4:
                break
        
        return pos


class GridRasterizer:
    """Convert continuous positions to integer grid coordinates."""
    
    def __init__(self, config: HALConfig):
        self.config = config
        
    def rasterize_positions(self, positions: Dict[int, Tuple[float, float]], 
                           auxiliary_grid_size: Optional[Tuple[int, int]] = None) -> Dict[int, Tuple[int, int]]:
        """Convert continuous positions to integer grid coordinates using two-phase rasterization.
        
        Implements the exact algorithm from HAL paper Section A.1.b:
        Phase 1: Naive rounding with immediate acceptance
        Phase 2: Priority conflict resolution using min-heap with Euclidean distance keys
        """
        if not positions:
            return {}
        
        # Determine target grid dimensions
        if auxiliary_grid_size:
            grid_width, grid_height = auxiliary_grid_size
        else:
            grid_width, grid_height = self.config.grid_size
        
        # Normalize positions to grid coordinate system
        pos_array = np.array(list(positions.values()))
        min_pos = np.min(pos_array, axis=0)
        max_pos = np.max(pos_array, axis=0)
        
        # Calculate scaling parameters with margin for grid utilization
        margin = 1
        available_size = (
            grid_width - 2 * margin,
            grid_height - 2 * margin
        )
        
        range_pos = max_pos - min_pos
        range_pos[range_pos == 0] = 1.0  # Prevent division by zero
        
        scale = min(available_size[0] / range_pos[0], available_size[1] / range_pos[1])
        
        # Phase 1: Naive rounding and immediate acceptance
        grid_positions = {}
        occupied = set()
        conflicts = []
        
        nodes = list(positions.keys())
        for node in nodes:
            x, y = positions[node]
            
            # Transform to grid coordinates
            x_norm = (x - min_pos[0]) * scale + margin
            y_norm = (y - min_pos[1]) * scale + margin
            
            # Map to nearest lattice point
            grid_x = int(np.floor(x_norm + 0.5))
            grid_y = int(np.floor(y_norm + 0.5))
            
            # Enforce grid boundaries
            grid_x = max(0, min(grid_width - 1, grid_x))
            grid_y = max(0, min(grid_height - 1, grid_y))
            
            # Accept placement if site is available
            if (grid_x, grid_y) not in occupied:
                grid_positions[node] = (grid_x, grid_y)
                occupied.add((grid_x, grid_y))
            else:
                # Queue node for conflict resolution in Phase 2
                conflicts.append({
                    'node': node,
                    'original_pos': positions[node],
                    'preferred_grid_pos': (grid_x, grid_y)
                })
        
        # Phase 2: Priority conflict resolution
        if conflicts:
            grid_positions.update(self._resolve_conflicts_with_heap(conflicts, occupied, grid_width, grid_height))
        
        return grid_positions
    
    def _resolve_conflicts_with_heap(self, conflicts: List[Dict], occupied: Set[Tuple[int, int]], 
                                    grid_width: int, grid_height: int) -> Dict[int, Tuple[int, int]]:
        """Phase 2 priority conflict resolution using min-heap algorithm.
        
        Implements exact algorithm from HAL paper: nodes enter min-heap keyed by
        Euclidean distance to nearest free site, processed greedily with distance
        key updates as sites become occupied.
        """
        resolved = {}
        
        if not conflicts:
            return resolved
        
        # Initialize min-heap with distance-keyed conflicts
        heap = []
        
        for conflict in conflicts:
            node = conflict['node']
            preferred_pos = conflict['preferred_grid_pos']
            
            # Find nearest available site and compute distance
            nearest_free_site, distance = self._find_nearest_free_site_with_distance(
                preferred_pos, occupied, grid_width, grid_height
            )
            
            if nearest_free_site is not None:
                heapq.heappush(heap, (distance, node, nearest_free_site))
        
        # Process heap greedily: closest nodes get priority
        while heap:
            distance, node, nearest_site = heapq.heappop(heap)
            
            # Verify site availability (may have been claimed)
            if nearest_site not in occupied:
                resolved[node] = nearest_site
                occupied.add(nearest_site)
                
                # Update remaining heap elements with new distances
                self._update_heap_distances(heap, occupied, grid_width, grid_height)
            else:
                # Find alternative site for displaced node
                new_nearest_site, new_distance = self._find_nearest_free_site_with_distance(
                    nearest_site, occupied, grid_width, grid_height
                )
                
                if new_nearest_site is not None:
                    heapq.heappush(heap, (new_distance, node, new_nearest_site))
        
        return resolved
    
    def _find_nearest_free_site_with_distance(self, target: Tuple[int, int], occupied: Set[Tuple[int, int]], 
                                             grid_width: int, grid_height: int) -> Tuple[Optional[Tuple[int, int]], float]:
        """Find nearest free lattice site using expanding square shells method."""
        target_x, target_y = target
        
        # Expand search radius using square shells
        for radius in range(max(grid_width, grid_height)):
            # Check positions on shell perimeter
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    # Only examine shell boundary positions
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                    
                    x, y = target_x + dx, target_y + dy
                    
                    # Validate grid boundaries and availability
                    if 0 <= x < grid_width and 0 <= y < grid_height:
                        if (x, y) not in occupied:
                            distance = np.sqrt(dx*dx + dy*dy)
                            return (x, y), distance
        
        return None, float('inf')
    
    def _update_heap_distances(self, heap: List, occupied: Set[Tuple[int, int]], 
                              grid_width: int, grid_height: int):
        """Update distance keys for remaining heap elements in-place."""
        updated_heap = []
        
        for distance, node, old_site in heap:
            # Recompute nearest free site and distance
            new_nearest_site, new_distance = self._find_nearest_free_site_with_distance(
                old_site, occupied, grid_width, grid_height
            )
            
            if new_nearest_site is not None:
                updated_heap.append((new_distance, node, new_nearest_site))
        
        # Rebuild heap with updated distances
        heap.clear()
        for item in updated_heap:
            heapq.heappush(heap, item)
    
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
        """Compact grid using monotone coordinate remapping as specified in HAL paper.
        
        Paper (Section A.1.c): "The set of distinct x-coordinates is sorted, and the 
        i-th element is mapped to i; the same is done for the y-coordinates. The monotone 
        remap preserves the embedding planarity and relative edge lengths measured in grid units."
        """
        if not positions:
            return positions
        
        # Extract and sort distinct coordinates
        coords = list(positions.values())
        x_coords = sorted(set(x for x, y in coords))
        y_coords = sorted(set(y for x, y in coords))
        
        # Create monotone coordinate mappings
        x_mapping = {old_x: new_x for new_x, old_x in enumerate(x_coords)}
        y_mapping = {old_y: new_y for new_y, old_y in enumerate(y_coords)}
        
        # Apply monotone remapping to preserve planarity
        compacted = {}
        for node, (x, y) in positions.items():
            compacted[node] = (x_mapping[x], y_mapping[y])
        
        # Translate to positive quadrant (final paper step)
        if compacted:
            min_x = min(x for x, y in compacted.values())
            min_y = min(y for x, y in compacted.values())
            
            # Ensure all coordinates are non-negative
            if min_x < 0 or min_y < 0:
                translated = {}
                for node, (x, y) in compacted.items():
                    translated[node] = (x - min_x, y - min_y)
                compacted = translated
        
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
            # Always rasterize and compact custom positions as per HAL paper spec
            snapped = self.rasterizer.rasterize_positions(custom_positions)
            node_positions = self.rasterizer.compact_grid(snapped)
            
            # Add spacing between qubits for routing infrastructure
            node_positions = self._add_qubit_spacing(node_positions, spacing=10)
            
            # Run normal MPS extraction on the given graph
            # Step 1: Community detection for edge prioritization
            community_detector = CommunityDetector(graph, self.config)
            communities = community_detector.detect_communities()
            
            # Step 2: Extract planar subgraph using position-dependent edge priorities
            analyzer = GraphAnalyzer(graph)
            edge_priorities = analyzer.get_edge_priorities(communities, node_positions)
            
            planarity_tester = PlanarityTester(graph)
            planar_edges = planarity_tester.get_planar_subgraph(edge_priorities)
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
        
        # Step 2: Initial spring layout for position-dependent edge prioritization
        self.spring_layout = SpringLayout(graph, self.config)
        # Use empty planar subgraph initially to get rough positions
        initial_positions = self.spring_layout.compute_layout(set())
        
        # Step 3: Extract planar subgraph using position-dependent edge priorities
        analyzer = GraphAnalyzer(graph)
        edge_priorities = analyzer.get_edge_priorities(communities, initial_positions)
        
        planarity_tester = PlanarityTester(graph)
        planar_edges = planarity_tester.get_planar_subgraph(edge_priorities)
        
        # Step 4: Refined spring layout on extracted planar subgraph
        continuous_positions = self.spring_layout.compute_layout(planar_edges)
        
        # Step 4: Calculate auxiliary grid dimensions (like paper's approach)
        auxiliary_grid_size = self._calculate_auxiliary_grid_dimensions(len(graph.nodes()))
        
        # Step 5: Rasterize continuous positions to grid coordinates
        final_positions = self.rasterizer.rasterize_positions(continuous_positions, auxiliary_grid_size)
        
        # Step 6: Apply grid compaction with monotone remapping
        final_positions = self.rasterizer.compact_grid(final_positions)
        
        # Step 7: Add spacing between qubits for routing infrastructure
        final_positions = self._add_qubit_spacing(final_positions, spacing=10)
        
        return final_positions, planar_edges, communities
    
    def _calculate_auxiliary_grid_dimensions(self, n_logical_qubits: int) -> Tuple[int, int]:
        """
        Return initial grid dimensions for rasterization.
        The actual grid size emerges naturally from the placement optimization process.
        This is just an initial canvas that gets compacted after placement.
        """
        if n_logical_qubits == 0:
            return (10, 10)
        
        # Start with generous initial canvas - will be compacted after placement
        # Paper's rasterization process determines actual space needs naturally
        initial_size = max(10, int((n_logical_qubits) ** 0.6) + 5)
        return (initial_size, initial_size)
    
    def _add_qubit_spacing(self, positions: Dict[int, Tuple[int, int]], spacing: int) -> Dict[int, Tuple[int, int]]:
        """
        Add spacing between qubits to allow for routing infrastructure.
        
        Args:
            positions: Current qubit positions
            spacing: Units of space to add between each qubit
            
        Returns:
            New positions with spacing applied
        """
        spaced_positions = {}
        for node, (x, y) in positions.items():
            spaced_positions[node] = (x * spacing, y * spacing)
        
        return spaced_positions


class AspectRatioAnalyzer:
    """Aspect ratio analysis for bivariate bicycle code layout selection."""
    
    def __init__(self, config: HALConfig):
        self.config = config
        
    def calculate_aspect_ratio(self, node_positions: Dict[int, Tuple[int, int]]) -> float:
        """Calculate aspect ratio of qubit lattice (height/width where height >= width)."""
        if not node_positions:
            return 1.0
            
        coords = list(node_positions.values())
        x_coords = [x for x, y in coords]
        y_coords = [y for x, y in coords]
        
        width = max(x_coords) - min(x_coords) + 1
        height = max(y_coords) - min(y_coords) + 1
        
        # Ensure aspect ratio >= 1 (height/width where height >= width)
        return max(height, width) / min(height, width)
    
    def should_use_spring_layout(self, aspect_ratio: float) -> bool:
        """Determine if spring layout should be used based on aspect ratio analysis."""
        # Low aspect ratio (< 4): Square grid wins by ~30%
        if aspect_ratio < 4.0:
            return False
            
        # Cross-over regime (4-8): Mixed performance, prefer square for consistency
        elif aspect_ratio <= 8.0:
            return False
            
        # High aspect ratio (> 8): Spring layout wins by up to 4× cost reduction
        else:
            return True
    
    def select_optimal_layout_strategy(self, graph: nx.Graph, node_positions: Dict[int, Tuple[int, int]]) -> str:
        """Select optimal layout strategy based on aspect ratio analysis."""
        aspect_ratio = self.calculate_aspect_ratio(node_positions)
        
        if self.should_use_spring_layout(aspect_ratio):
            return "spring_layout"
        else:
            return "square_grid"


class BivariateBicycleLayoutSelector:
    """Layout strategy selector specifically for bivariate bicycle codes."""
    
    def __init__(self, placement_engine: PlacementEngine):
        self.placement_engine = placement_engine
        self.aspect_analyzer = AspectRatioAnalyzer(placement_engine.config)
        
    def select_optimal_placement(self, graph: nx.Graph, 
                                custom_square_positions: Optional[Dict[int, Tuple[int, int]]] = None) -> PlacementResult:
        """
        Select optimal placement strategy for BB codes using aspect ratio analysis.
        Runs both square grid and spring layout, then picks the better result.
        """
        results = []
        
        # Strategy 1: Square grid layout (if custom positions provided)
        if custom_square_positions:
            try:
                # Create temporary config with custom positions
                square_config = HALConfig(**self.placement_engine.config.__dict__)
                square_config.custom_positions = custom_square_positions
                
                square_engine = PlacementEngine(square_config)
                square_result = square_engine.place_nodes(graph)
                square_result.strategy_used = "square_grid"
                results.append(square_result)
                
                # Calculate aspect ratio for decision guidance
                aspect_ratio = self.aspect_analyzer.calculate_aspect_ratio(custom_square_positions)
                
            except Exception as e:
                print(f"Square grid placement failed: {e}")
        
        # Strategy 2: Spring layout
        try:
            spring_result = self.placement_engine.place_nodes(graph)
            spring_result.strategy_used = "spring_layout"
            results.append(spring_result)
        except Exception as e:
            print(f"Spring layout placement failed: {e}")
        
        # Select best result (if multiple available)
        if len(results) == 1:
            return results[0]
        elif len(results) == 2:
            # Compare results and select optimal based on expected performance
            square_result, spring_result = results
            aspect_ratio = self.aspect_analyzer.calculate_aspect_ratio(custom_square_positions)
            
            # Use aspect ratio analysis to predict better strategy
            if self.aspect_analyzer.should_use_spring_layout(aspect_ratio):
                print(f"High aspect ratio ({aspect_ratio:.1f}) detected: selecting spring layout")
                return spring_result
            else:
                print(f"Low aspect ratio ({aspect_ratio:.1f}) detected: selecting square grid")
                return square_result
        else:
            raise ValueError("Both placement strategies failed")
    
