"""
Main HAL (Hardware-Aware Layout) algorithm implementation.
"""

import networkx as nx
from typing import Dict, List, Tuple, Optional
import numpy as np
import time

from .config import HALConfig
from .data_structures import QECCLayout
from .placement import PlacementEngine
from .routing import RoutingEngine
from .cost import HardwareCostCalculator
from .visualizer import HALVisualizer
from .graph_utils import create_qecc_graph_from_edges


class HAL:
    """
    Main HAL algorithm implementation.
    
    Coordinates placement, routing, cost calculation, and visualization
    for quantum error correcting code layouts on multi-tier hardware.
    """
    
    def __init__(self, config: Optional[HALConfig] = None):
        """
        Initialize HAL algorithm with configuration.
        
        Args:
            config: HALConfig object, uses defaults if None
        """
        self.config = config or HALConfig()
        self.config.validate()
        
        # Initialize algorithm components
        self.placement_engine = PlacementEngine(self.config)
        self.routing_engine = RoutingEngine(self.config)
        self.cost_calculator = HardwareCostCalculator(self.config)
        self.visualizer = HALVisualizer(self.config)
        
        # Initialize random number generator for deterministic results
        np.random.seed(self.config.random_seed)
    
    def layout_code(self, connectivity_graph: nx.Graph, 
                   custom_positions: Optional[Dict[int, Tuple[int, int]]] = None,
                   verbose: bool = False) -> QECCLayout:
        """
        Execute complete HAL algorithm pipeline: placement, routing, and cost optimization.
        
        Args:
            connectivity_graph: Graph representing QECC connectivity
            custom_positions: Optional pre-specified node positions
            verbose: Whether to print progress information
            
        Returns:
            Complete QECCLayout with placement, routing, and cost information
        """
        start_time = time.time()
        
        if verbose:
            print(f"Starting HAL layout for graph with {connectivity_graph.number_of_nodes()} nodes, "
                  f"{connectivity_graph.number_of_edges()} edges")
        
        # Phase 1: Node Placement
        if verbose:
            print("Phase 1: Executing node placement...")
        placement_start = time.time()
        
        placement_result = self.placement_engine.place_nodes(
            connectivity_graph, custom_positions
        )
        
        placement_time = time.time() - placement_start
        if verbose:
            print(f"  Placement completed in {placement_time:.2f}s")
            print(f"  Grid bounds: {placement_result.grid_bounds}")
            print(f"  Planar edges: {len(placement_result.planar_subgraph_edges)}/{connectivity_graph.number_of_edges()}")
        
        # Phase 2: Multi-Tier Edge Routing
        if verbose:
            print("Phase 2: Executing multi-tier routing...")
        routing_start = time.time()
        
        routing_result = self.routing_engine.route_edges(
            connectivity_graph,
            placement_result.node_positions,
            placement_result.planar_subgraph_edges,
            placement_result.grid_bounds
        )
        
        routing_time = time.time() - routing_start
        if verbose:
            print(f"  Routing completed in {routing_time:.2f}s")
            print(f"  Routed edges: {len(routing_result.edge_routes)}/{connectivity_graph.number_of_edges()}")
            print(f"  Unrouted edges: {len(routing_result.unrouted_edges)}")
            print(f"  Tiers used: {len(routing_result.tiers)}")
        
        # Phase 3: Hardware Cost Calculation
        if verbose:
            print("Phase 3: Computing normalized hardware cost...")
        
        hardware_cost = self.cost_calculator.calculate_cost(routing_result.metrics)
        
        if verbose:
            print(f"  Hardware cost: {hardware_cost:.3f}")
            print(f"  Metrics: {routing_result.metrics}")
        
        # Phase 4: Construct Comprehensive Layout Representation
        layout = QECCLayout(
            node_positions=placement_result.node_positions,
            edge_routes=routing_result.edge_routes,
            tiers=routing_result.tiers,  # Keep full RoutingTier objects
            metrics=routing_result.metrics,
            hardware_cost=hardware_cost
        )
        
        total_time = time.time() - start_time
        if verbose:
            print(f"\nTotal HAL execution time: {total_time:.2f}s")
            
            # Print improvement suggestions
            suggestions = self.cost_calculator.suggest_improvements(routing_result.metrics)
            if suggestions:
                print("\nImprovement suggestions:")
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"  {i}. {suggestion}")
        
        return layout
    
    def layout_code_from_edges(self, edges: List[Tuple[int, int]], 
                              custom_positions: Optional[Dict[int, Tuple[int, int]]] = None,
                              verbose: bool = False) -> QECCLayout:
        """
        Layout code from edge list.
        
        Args:
            edges: List of (node1, node2) tuples representing connectivity
            custom_positions: Optional pre-specified node positions
            verbose: Whether to print progress information
            
        Returns:
            Complete QECCLayout
        """
        graph = create_qecc_graph_from_edges(edges)
        return self.layout_code(graph, custom_positions, verbose)
    
    def batch_layout_codes(self, graphs: List[nx.Graph], 
                          labels: Optional[List[str]] = None,
                          custom_positions: Optional[List[Dict[int, Tuple[int, int]]]] = None,
                          verbose: bool = False,
                          parallel: bool = False,
                          n_processes: Optional[int] = None) -> Dict[str, QECCLayout]:
        """
        Execute batch processing of multiple QECC layouts with comparative analysis.
        Supports both sequential and parallel processing.
        
        Args:
            graphs: List of connectivity graphs
            labels: Optional labels for each graph
            custom_positions: Optional list of position dictionaries
            verbose: Whether to print progress information
            parallel: Enable parallel processing using multiprocessing
            n_processes: Number of processes for parallel execution
            
        Returns:
            Dictionary mapping labels to QECCLayout results
        """
        if labels is None:
            labels = [f"Code_{i+1}" for i in range(len(graphs))]
        
        if custom_positions is None:
            custom_positions = [None] * len(graphs)
        
        results = {}
        
        # Use parallel processing if requested
        if parallel:
            from .parallel import ParallelHAL
            
            # Prepare code information for parallel processing
            code_infos = []
            for i, (graph, label, positions) in enumerate(zip(graphs, labels, custom_positions)):
                code_infos.append({
                    'graph': graph,
                    'label': label,
                    'positions': positions,
                    'index': i
                })
            
            def graph_creator(code_info):
                return code_info['graph']  # Graph is already created
            
            # Process in parallel
            parallel_hal = ParallelHAL(n_processes=n_processes, verbose=verbose)
            parallel_results = parallel_hal.process_batch(code_infos, graph_creator)
            
            # Convert results back to expected format
            for result in parallel_results:
                if result['success']:
                    label = result['code_info']['label']
                    results[label] = result['layout']
                elif verbose:
                    label = result['code_info']['label']
                    print(f"Failed to process {label}: {result['error']}")
        else:
            # Sequential processing (original behavior)
            for i, (graph, label, positions) in enumerate(zip(graphs, labels, custom_positions)):
                if verbose:
                    print(f"\n{'='*50}")
                    print(f"Executing layout algorithm for {label} ({i+1}/{len(graphs)})")
                    print(f"{'='*50}")
                
                layout = self.layout_code(graph, positions, verbose)
                results[label] = layout
        
        if verbose:
            print(f"\n{'='*50}")
            print("Batch Processing Complete")
            print(f"{'='*50}")
            
            # Print comparison summary
            print("\nCost Comparison:")
            sorted_results = sorted(results.items(), key=lambda x: x[1].hardware_cost)
            for label, layout in sorted_results:
                print(f"  {label}: {layout.hardware_cost:.3f}")
        
        return results
    
    def analyze_layout(self, layout: QECCLayout, detailed: bool = False) -> Dict[str, any]:
        """
        Perform comprehensive layout quality assessment with detailed performance metrics.
        
        Args:
            layout: QECCLayout to analyze
            detailed: Whether to include detailed breakdown
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'basic_metrics': layout.metrics.copy(),
            'hardware_cost': layout.hardware_cost,
            'nodes': len(layout.node_positions),
            'edges_routed': len(layout.edge_routes),
            'tiers_used': len([t for t in layout.tiers if t]),
            'grid_utilization': self._calculate_grid_utilization(layout)
        }
        
        if detailed:
            analysis['cost_breakdown'] = self.cost_calculator.calculate_detailed_cost_breakdown(layout.metrics)
            analysis['cost_sensitivity'] = self.cost_calculator.analyze_cost_sensitivity(layout.metrics)
            analysis['improvement_suggestions'] = self.cost_calculator.suggest_improvements(layout.metrics)
            analysis['routing_efficiency'] = self._calculate_routing_efficiency(layout)
        
        return analysis
    
    def _calculate_grid_utilization(self, layout: QECCLayout) -> Dict[str, float]:
        """Calculate grid utilization statistics."""
        if not layout.tiers:
            return {'overall': 0.0, 'per_tier': []}
        
        tier_utilizations = []
        total_cells = 0
        total_occupied = 0
        
        for tier_id, tier in enumerate(layout.tiers):
            if not tier:
                tier_utilizations.append(0.0)
                continue
            
            # Calculate tier size and occupied cells from RoutingTier grid
            if tier and hasattr(tier, 'grid'):
                tier_size = tier.grid.shape[0] * tier.grid.shape[1]
                tier_occupied = np.sum(tier.grid)
            else:
                tier_size = 1
                tier_occupied = 0
            
            utilization = tier_occupied / tier_size if tier_size > 0 else 0.0
            tier_utilizations.append(utilization)
            
            total_cells += tier_size
            total_occupied += tier_occupied
        
        overall_utilization = total_occupied / total_cells if total_cells > 0 else 0.0
        
        return {
            'overall': overall_utilization,
            'per_tier': tier_utilizations,
            'avg_tier_utilization': np.mean(tier_utilizations) if tier_utilizations else 0.0
        }
    
    def _calculate_routing_efficiency(self, layout: QECCLayout) -> Dict[str, float]:
        """Calculate routing efficiency metrics."""
        if not layout.edge_routes:
            return {'avg_path_efficiency': 0.0, 'total_overhead': 0.0}
        
        efficiencies = []
        total_actual = 0
        total_optimal = 0
        
        for edge, path in layout.edge_routes.items():
            if not path or len(path) < 2:
                continue
            
            # Calculate actual path length
            actual_length = len(path) - 1
            
            # Calculate Manhattan distance (optimal)
            start = path[0]
            end = path[-1]
            optimal_length = abs(start[0] - end[0]) + abs(start[1] - end[1])
            
            if optimal_length > 0:
                efficiency = optimal_length / actual_length
                efficiencies.append(efficiency)
                
                total_actual += actual_length
                total_optimal += optimal_length
        
        avg_efficiency = np.mean(efficiencies) if efficiencies else 0.0
        total_overhead = (total_actual - total_optimal) / total_optimal if total_optimal > 0 else 0.0
        
        return {
            'avg_path_efficiency': avg_efficiency,
            'total_overhead': total_overhead,
            'min_efficiency': np.min(efficiencies) if efficiencies else 0.0,
            'max_efficiency': np.max(efficiencies) if efficiencies else 0.0
        }
    
    def visualize_layout(self, layout: QECCLayout, n: int = None, k: int = None, d: int = None) -> None:
        """
        Visualize layout using built-in visualizer.

        Args:
            layout: QECCLayout to visualize
            n: Total number of qubits
            k: Number of logical qubits
            d: Minimum distance
        """
        self.visualizer.plot_layout(layout, n, k, d)
    
    def visualize_cost_analysis(self, layout: QECCLayout, show_breakdown: bool = True) -> None:
        """Generate hardware cost analysis visualization using integrated plotting system."""
        self.visualizer.plot_cost_analysis(layout, show_breakdown)
    
    def compare_layouts(self, layouts: List[QECCLayout], labels: Optional[List[str]] = None) -> Dict[str, any]:
        """
        Execute comparative analysis of multiple layouts with integrated visualization.
        
        Args:
            layouts: List of QECCLayout objects to compare
            labels: Optional labels for each layout
            
        Returns:
            Comparison results dictionary
        """
        comparison = self.cost_calculator.compare_layouts(layouts, labels)
        self.visualizer.compare_layouts(layouts, labels)
        
        return comparison