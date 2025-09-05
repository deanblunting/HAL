#!/usr/bin/env python3
"""
Simplified interactive HAL script for quantum error correcting code layout generation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hal import HAL
from hal.graph_utils import (
    create_bicycle_code_graph,
    create_tile_code_graph,
    create_radial_code_graph,
    create_hypergraph_code_graph,
    get_code_custom_positions
)


def get_user_input():
    """Get quantum code parameters from user."""
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
    """Select quantum code type."""
    print("\nSelect quantum code type:")
    print("1. BB code (bivariate bicycle)")
    print("2. Tile code")
    print("3. Radial code")
    print("4. Hypergraph code")
    
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
    """Generate quantum code graph."""
    
    if code_type == 1:  # BB code
        print(f"Generating BB code with parameters [n={n}, k={k}, d={d}]...")
        graph = create_bicycle_code_graph(n, k, d)
        return graph, 'bicycle', {'n': n, 'k': k, 'd': d}
    
    elif code_type == 2:  # Tile code
        print(f"Generating tile code with parameters [n={n}, k={k}, d={d}]...")
        graph = create_tile_code_graph(n, k, d)
        return graph, 'tile', {'n': n, 'k': k, 'd': d}
    
    elif code_type == 3:  # Radial code
        print(f"Generating radial code with parameters [n={n}, k={k}, d={d}]...")
        graph = create_radial_code_graph(n, k, d)
        return graph, 'radial', {'n': n, 'k': k, 'd': d}
    
    elif code_type == 4:  # Hypergraph code
        print(f"Generating hypergraph code with parameters [n={n}, k={k}, d={d}]...")
        graph = create_hypergraph_code_graph(n, k, 'structured')
        return graph, 'hypergraph', {'n': n, 'k': k, 'd': d}


def main():
    """Main application workflow."""
    try:
        # Get user input
        n, k, d = get_user_input()
        code_type = select_code_type()
        
        # Generate code graph
        try:
            graph, code_family, params = generate_code_graph(n, k, d, code_type)
            print(f"\nGenerated graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        except Exception as e:
            print(f"Error generating code graph: {e}")
            return
        
        # Get custom positions if available
        custom_positions = get_code_custom_positions(graph, code_family, **params)
        if custom_positions:
            print(f"Using custom {code_family} code positions")
        else:
            print(f"Using spring layout for {code_family} code")
        
        # Create HAL layout
        hal = HAL()
        print("\nGenerating optimal layout...")
        try:
            layout = hal.layout_code(graph, custom_positions=custom_positions, verbose=True)
            print("Layout generation complete!")
        except Exception as e:
            print(f"Error generating layout: {e}")
            return
        
        # Show results
        print(f"\nResults:")
        print(f"  Hardware Cost: {layout.hardware_cost:.3f}")
        print(f"  Tiers Used: {len(layout.tiers)}")
        print(f"  Edges Routed: {len(layout.edge_routes)}/{graph.number_of_edges()}")
        
        # Create visualization
        print("\nCreating visualization...")
        try:
            hal.visualize_layout(layout)
            print("Visualization complete!")
        except Exception as e:
            print(f"Visualization error: {e}")
            
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    
    print("\nThank you for using HAL!")


if __name__ == "__main__":
    main()