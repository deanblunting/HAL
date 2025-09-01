#!/usr/bin/env python3
"""
Interactive HAL script for quantum error correcting code layout generation.
Asks user for [n,k,d] parameters and generates optimal hardware layout.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import networkx as nx
from hal import HAL, HALConfig
from hal.graph_utils import (
    create_bicycle_code_graph,
    create_tile_code_graph,
    create_radial_code_graph,
    create_hypergraph_code_graph,
    get_code_custom_positions
)


def get_user_input():
    """Acquire quantum code parameters and type specification from user interface."""
    print("HAL Quantum Error Correcting Code Layout Generator")
    print("=" * 55)
    print()
    
    print("Please provide the code parameters [n,k,d] where:")
    print("  n = total number of qubits")
    print("  k = number of logical qubits")  
    print("  d = minimum distance")
    print()
    
    # Get parameters
    while True:
        try:
            n = int(input("Enter n (total qubits): "))
            if n <= 0:
                print("Error: n must be positive")
                continue
            break
        except ValueError:
            print("Error: Please enter a valid integer")
    
    while True:
        try:
            k = int(input("Enter k (logical qubits): "))
            if k < 0 or k >= n:
                print("Error: k must be non-negative and less than n")
                continue
            break
        except ValueError:
            print("Error: Please enter a valid integer")
    
    while True:
        try:
            d = int(input("Enter d (minimum distance): "))
            if d <= 0:
                print("Error: d must be positive")
                continue
            break
        except ValueError:
            print("Error: Please enter a valid integer")
    
    return n, k, d


def select_code_type():
    """Present quantum code type selection interface for user specification."""
    print("\nSelect quantum code type:")
    print("1. BB code (bivariate bicycle, toric structure)")
    print("2. Tile code (open-boundary, O(1) locality)")
    print("3. Radial code (concentric rings)")
    print("4. Custom hypergraph code")
    
    while True:
        try:
            choice = int(input("\nEnter choice (1-4): "))
            if 1 <= choice <= 4:
                return choice
            else:
                print("Error: Please enter a number between 1 and 4")
        except ValueError:
            print("Error: Please enter a valid number")


def generate_code_graph(n, k, d, code_type):
    """Construct quantum code connectivity graph according to specified parameters and type."""
    
    if code_type == 1:  # BB code
        # Determine BB code construction parameters
        n1 = int(n**0.5)
        n2 = (n + n1 - 1) // n1
        a, b = 1, 1  # Simple shift parameters
        print(f"Generating BB code with n1={n1}, n2={n2}, a={a}, b={b}...")
        graph = create_bicycle_code_graph(n1, n2, a, b)
        return graph, 'bicycle', {'n1': n1, 'n2': n2}
    
    elif code_type == 2:  # Tile code
        # Determine tile code parameters from qubit count
        tile_size = 3  # Standard 3x3 tiles
        total_tiles = max(1, n // (tile_size * tile_size))
        tiles_per_side = max(1, int(total_tiles**0.5))
        tiles_x = tiles_per_side
        tiles_y = (total_tiles + tiles_per_side - 1) // tiles_per_side
        print(f"Generating tile code with {tiles_x}x{tiles_y} tiles of size {tile_size}x{tile_size}...")
        graph = create_tile_code_graph(tiles_x, tiles_y, tile_size)
        return graph, 'tile', {'tiles_x': tiles_x, 'tiles_y': tiles_y, 'tile_size': tile_size}
    
    elif code_type == 3:  # Radial code
        # Derive radial code parameters from qubit count constraint
        # Optimize radial code parameters for target qubit count
        best_r, best_s = 2, max(1, n // 8)
        for r in range(2, int(n**0.33) + 2):
            s = n // (2 * r * r)
            if s >= 1 and abs(2 * r * r * s - n) < abs(2 * best_r * best_r * best_s - n):
                best_r, best_s = r, s
        print(f"Generating radial code with r={best_r}, s={best_s} (n≈{2*best_r*best_r*best_s})...")
        graph = create_radial_code_graph(best_r, best_s)
        return graph, 'radial', {'r': best_r, 's': best_s}
    
    elif code_type == 4:  # Custom hypergraph code
        avg_degree = min(6, n // 2)  # Reasonable default
        print(f"Generating hypergraph code with average degree {avg_degree}...")
        graph = create_hypergraph_code_graph(n, avg_degree, 'structured')
        return graph, 'hypergraph', {}


def configure_hal():
    """Initialize HAL algorithm configuration interface."""
    print("\nHAL Algorithm Configuration:")
    print("1. Use default configuration (recommended)")
    print("2. Custom configuration")
    
    while True:
        try:
            choice = int(input("\nEnter choice (1-2): "))
            if choice == 1:
                return HAL()
            elif choice == 2:
                return configure_custom_hal()
            else:
                print("Error: Please enter 1 or 2")
        except ValueError:
            print("Error: Please enter a valid number")


def configure_custom_hal():
    """Establish custom HAL algorithm parameter configuration."""
    print("\nCustom HAL Configuration:")
    
    # Get max tiers
    while True:
        try:
            max_tiers = int(input("Maximum routing tiers (default 10): ") or "10")
            if max_tiers > 0:
                break
            print("Error: Must be positive")
        except ValueError:
            print("Error: Please enter a valid integer")
    
    # Get cost weights
    print("\nCost weights (higher = more penalty):")
    try:
        tier_weight = float(input("Tier weight (default 1.0): ") or "1.0")
        length_weight = float(input("Length weight (default 1.0): ") or "1.0")
        bump_weight = float(input("Bump weight (default 1.0): ") or "1.0")
        tsv_weight = float(input("TSV weight (default 1.0): ") or "1.0")
    except ValueError:
        print("Using default weights...")
        tier_weight = length_weight = bump_weight = tsv_weight = 1.0
    
    config = HALConfig(
        max_tiers=max_tiers,
        cost_weights={
            'tiers': tier_weight,
            'length': length_weight,
            'bumps': bump_weight,
            'tsvs': tsv_weight
        },
        random_seed=42
    )
    
    return HAL(config)


def analyze_and_display_results(hal, layout, graph, n, k, d):
    """Execute comprehensive layout analysis and present performance metrics."""
    print("\n" + "=" * 55)
    print("LAYOUT RESULTS")
    print("=" * 55)
    
    print(f"\nCode Parameters: [n={n}, k={k}, d={d}]")
    print(f"Graph Properties:")
    print(f"  Nodes: {graph.number_of_nodes()}")
    print(f"  Edges: {graph.number_of_edges()}")
    print(f"  Density: {nx.density(graph):.3f}")
    
    print(f"\nHardware Layout:")
    print(f"  Hardware Cost: {layout.hardware_cost:.3f}")
    print(f"  Tiers Used: {len(layout.tiers)}")
    print(f"  Edges Routed: {len(layout.edge_routes)}/{graph.number_of_edges()}")
    
    # Determine grid utilization bounds from node placement
    if layout.node_positions:
        x_coords = [pos[0] for pos in layout.node_positions.values()]
        y_coords = [pos[1] for pos in layout.node_positions.values()]
        grid_width = max(x_coords) - min(x_coords) + 1
        grid_height = max(y_coords) - min(y_coords) + 1
        print(f"  Grid Size: {grid_width}x{grid_height}")
    else:
        print(f"  Grid Size: Unknown")
    
    if layout.metrics:
        print(f"\nDetailed Metrics:")
        for metric, value in layout.metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")
    
    # Execute comprehensive performance analysis
    try:
        analysis = hal.analyze_layout(layout, detailed=True)
        print(f"\nLayout Analysis:")
        if 'grid_utilization' in analysis:
            print(f"  Grid Utilization: {analysis['grid_utilization']['overall']:.2%}")
        if 'routing_efficiency' in analysis:
            print(f"  Routing Efficiency: {analysis['routing_efficiency']['avg_path_efficiency']:.3f}")
        if 'improvement_suggestions' in analysis:
            print(f"  Improvement Suggestions: {len(analysis['improvement_suggestions'])}")
            for suggestion in analysis['improvement_suggestions'][:3]:  # Show top 3
                print(f"    - {suggestion}")
    except Exception as e:
        print(f"Analysis error: {e}")


def offer_visualization(hal, layout):
    """Present visualization interface options for user selection."""
    print("\n" + "=" * 55)
    print("VISUALIZATION OPTIONS")
    print("=" * 55)
    
    while True:
        print("\nVisualization options:")
        print("1. Create layout visualization")
        print("2. Create cost analysis plot")
        print("3. Skip visualization")
        print("4. Exit")
        
        try:
            choice = int(input("\nEnter choice (1-4): "))
            
            if choice == 1:
                try:
                    print("Creating layout visualization...")
                    hal.visualize_layout(layout, show_tiers=True, interactive=False)
                    print("Layout visualization created!")
                except Exception as e:
                    print(f"Visualization error: {e}")
                    
            elif choice == 2:
                try:
                    print("Creating cost analysis...")
                    hal.visualize_cost_analysis(layout, show_breakdown=True)
                    print("Cost analysis created!")
                except Exception as e:
                    print(f"Cost analysis error: {e}")
                    
            elif choice == 3:
                print("Skipping visualization.")
                break
                
            elif choice == 4:
                return True  # Signal to exit
                
            else:
                print("Error: Please enter a number between 1 and 4")
                
        except ValueError:
            print("Error: Please enter a valid number")
    
    return False


def save_results(layout, n, k, d):
    """Provide file output interface for result persistence."""
    print("\n" + "=" * 55)
    print("SAVE RESULTS")
    print("=" * 55)
    
    save = input("\nSave results to file? (y/n): ").lower().strip()
    if save in ['y', 'yes']:
        filename = f"hal_layout_n{n}_k{k}_d{d}.txt"
        try:
            with open(filename, 'w') as f:
                f.write(f"HAL Layout Results for [{n},{k},{d}] Code\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Hardware Cost: {layout.hardware_cost:.3f}\n")
                f.write(f"Tiers Used: {len(layout.tiers)}\n")
                
                # Compute and record grid utilization metrics
                if layout.node_positions:
                    x_coords = [pos[0] for pos in layout.node_positions.values()]
                    y_coords = [pos[1] for pos in layout.node_positions.values()]
                    grid_width = max(x_coords) - min(x_coords) + 1
                    grid_height = max(y_coords) - min(y_coords) + 1
                    f.write(f"Grid Size: {grid_width}x{grid_height}\n")
                else:
                    f.write(f"Grid Size: Unknown\n")
                f.write(f"Node Positions:\n")
                for node, pos in layout.node_positions.items():
                    f.write(f"  Node {node}: {pos}\n")
                f.write(f"\nRouting Information:\n")
                f.write(f"  Total routes: {len(layout.edge_routes)}\n")
                if layout.metrics:
                    f.write(f"\nMetrics:\n")
                    for metric, value in layout.metrics.items():
                        f.write(f"  {metric}: {value}\n")
            print(f"Results saved to {filename}")
        except Exception as e:
            print(f"Error saving file: {e}")


def main():
    """Execute primary interactive application workflow."""
    try:
        while True:
            # Acquire user specification parameters
            n, k, d = get_user_input()
            code_type = select_code_type()
            
            # Construct quantum code connectivity graph
            try:
                graph, code_family, params = generate_code_graph(n, k, d, code_type)
                print(f"\nGenerated graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
            except Exception as e:
                print(f"Error generating code graph: {e}")
                continue
            
            # Generate optimized positions for structured code families
            custom_positions = get_code_custom_positions(graph, code_family, **params)
            if custom_positions:
                print(f"Using custom {code_family} code positions for structured layout")
            else:
                print(f"Using spring layout for {code_family} code")
            
            # Initialize HAL algorithm configuration
            hal = configure_hal()
            
            # Execute HAL layout algorithm
            print("\nGenerating optimal layout...")
            try:
                layout = hal.layout_code(graph, custom_positions=custom_positions, verbose=True)
                print("Layout generation complete!")
            except Exception as e:
                print(f"Error generating layout: {e}")
                continue
            
            # Perform results analysis and presentation
            analyze_and_display_results(hal, layout, graph, n, k, d)
            
            # Present visualization interface
            should_exit = offer_visualization(hal, layout)
            if should_exit:
                break
            
            # Execute result persistence
            save_results(layout, n, k, d)
            
            # Query user for workflow continuation
            print("\n" + "=" * 55)
            continue_choice = input("\nGenerate another layout? (y/n): ").lower().strip()
            if continue_choice not in ['y', 'yes']:
                break
                
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    
    print("\nThank you for using HAL!")


if __name__ == "__main__":
    main()