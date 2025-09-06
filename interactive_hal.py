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
    create_radial_code_graph_from_nkd,
    create_hypergraph_code_graph,
    get_code_custom_positions
)


def get_bb_parameters():
    """Get BB code parameters (rows and columns)."""
    print("\nBB Code Parameters:")
    print("Enter the grid dimensions for the bicycle code")
    
    while True:
        try:
            n1 = int(input("Enter number of rows: "))
            if n1 <= 0:
                print("Error: number of rows must be positive")
                continue
            break
        except ValueError:
            print("Error: Please enter a valid integer")
    
    while True:
        try:
            n2 = int(input("Enter number of columns: "))
            if n2 <= 0:
                print("Error: number of columns must be positive")
                continue
            break
        except ValueError:
            print("Error: Please enter a valid integer")
    
    return n1, n2


def get_standard_parameters():
    """Get standard [n,k,d] parameters for other codes."""
    print("\nCode Parameters [n,k,d]:")
    print("  n = total number of qubits")
    print("  k = number of logical qubits")  
    print("  d = minimum distance")
    
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
    print("HAL Quantum Error Correcting Code Layout Generator")
    print("=" * 55)
    print()
    print("Select quantum code type:")
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


def generate_code_graph(code_type, params):
    """Generate quantum code graph based on type and parameters."""
    
    if code_type == 1:  # BB code
        n1, n2 = params
        print(f"Generating BB code with dimensions {n1}×{n2} ({n1*n2} qubits)...")
        graph = create_bicycle_code_graph(n1, n2)
        return graph, 'bicycle', {'n1': n1, 'n2': n2}
    
    elif code_type == 2:  # Tile code
        n, k, d = params
        print(f"Generating tile code with parameters [n={n}, k={k}, d={d}]...")
        graph = create_tile_code_graph(n, k, d)
        return graph, 'tile', {'n': n, 'k': k, 'd': d}
    
    elif code_type == 3:  # Radial code
        n, k, d = params
        print(f"Generating radial code with parameters [n={n}, k={k}, d={d}]...")
        graph = create_radial_code_graph_from_nkd(n, k, d)
        return graph, 'radial', {'n': n, 'k': k, 'd': d}
    
    elif code_type == 4:  # Hypergraph code
        n, k, d = params
        print(f"Generating hypergraph code with parameters [n={n}, k={k}, d={d}]...")
        graph = create_hypergraph_code_graph(n, k, 'structured')
        return graph, 'hypergraph', {'n': n, 'k': k, 'd': d}


def main():
    """Main application workflow."""
    try:
        # Select code type first
        code_type = select_code_type()
        
        # Get parameters based on code type
        if code_type == 1:  # BB code
            params = get_bb_parameters()
        else:  # Other codes use [n,k,d]
            params = get_standard_parameters()
        
        # Generate code graph
        try:
            graph, code_family, graph_params = generate_code_graph(code_type, params)
            print(f"\nGenerated graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        except Exception as e:
            print(f"Error generating code graph: {e}")
            return
        
        # Get custom positions if available
        custom_positions = get_code_custom_positions(graph, code_family, **graph_params)
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