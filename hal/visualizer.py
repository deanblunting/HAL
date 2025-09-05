"""
Simplified visualization components for HAL algorithm results - static mode only.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

from .data_structures import QECCLayout, RoutingTier
from .config import HALConfig


class HALVisualizer:
    """Simplified visualization tools for HAL algorithm results."""
    
    def __init__(self, config: HALConfig = None):
        self.config = config or HALConfig()
        
    def plot_layout(self, layout: QECCLayout) -> None:
        """
        Create static visualization with separate tier plots.
        
        Args:
            layout: QECCLayout object to visualize
        """
        if len(layout.tiers) > 1:
            self._plot_separate_tier_layouts_static(layout)
        else:
            self._plot_single_tier_static(layout)

    def _plot_separate_tier_layouts_static(self, layout: QECCLayout) -> None:
        """Create separate 2D static plots for each tier."""
        if not layout.node_positions:
            print("No layout to visualize")
            return
        
        n_tiers = len(layout.tiers)
        # Create multi-row layout for better visibility when many tiers
        if n_tiers <= 5:
            rows, cols = 1, n_tiers
        else:
            rows = 2
            cols = (n_tiers + 1) // 2  # Ceiling division
        
        # Make each tier subplot larger
        subplot_size = 5  # 5x5 inches per subplot
        fig, axes = plt.subplots(rows, cols, figsize=(subplot_size * cols, subplot_size * rows))
        
        # Handle axis indexing for both single row and multi-row cases
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = list(axes) if hasattr(axes, '__iter__') else [axes]
        else:
            axes = axes.flatten()
        
        # Separate logical and infrastructure routes
        logical_routes = []
        infrastructure_routes = []
        
        for edge, path in layout.edge_routes.items():
            if not path:
                continue
            u, v = edge
            if u >= 1000 or v >= 1000:
                infrastructure_routes.append((edge, path))
            else:
                logical_routes.append((edge, path))
        
        for tier_id, tier in enumerate(layout.tiers):
            ax = axes[tier_id]
            
            # Show grid
            grid_width, grid_height = tier.grid.shape[0], tier.grid.shape[1]
            
            # Grid lines
            for x in range(grid_width + 1):
                ax.plot([x, x], [0, grid_height], 'lightgray', alpha=0.3, linewidth=0.5)
            for y in range(grid_height + 1):
                ax.plot([0, grid_width], [y, y], 'lightgray', alpha=0.3, linewidth=0.5)
            
            # Plot qubits (only on tier 0) - use blue color
            if tier_id == 0:
                node_x = [pos[0] for pos in layout.node_positions.values()]
                node_y = [pos[1] for pos in layout.node_positions.values()]
                node_ids = list(layout.node_positions.keys())
                
                ax.scatter(node_x, node_y, c='blue', s=100, zorder=5)
                for i, node_id in enumerate(node_ids):
                    ax.annotate(str(node_id), (node_x[i], node_y[i]), 
                               xytext=(2, 2), textcoords='offset points', fontsize=8)
            
            # Plot logical routes for this tier
            for edge, path in logical_routes:
                if edge not in tier.edges:
                    continue
                
                # Plot all path segments (project to 2D)
                path_x = [pos[0] for pos in path]
                path_y = [pos[1] for pos in path]
                
                if len(path_x) > 1:
                    # Color edges by layer: black for layer 0, yellow for layer 1
                    layer_counts = {}
                    for pos in path:
                        layer = pos[2] % 2  # Layer within tier (0 or 1)
                        layer_counts[layer] = layer_counts.get(layer, 0) + 1
                    
                    # Use the layer where most of the path is
                    primary_layer = max(layer_counts, key=layer_counts.get) if layer_counts else 0
                    if primary_layer == 0:
                        edge_color = 'black'  # Layer 0
                    else:
                        edge_color = 'yellow'  # Layer 1
                    
                    ax.plot(path_x, path_y, color=edge_color, linewidth=2, alpha=0.8)
                
                # Highlight bump bonds (layer changes) with green squares
                for i in range(len(path) - 1):
                    current_pos = path[i]
                    next_pos = path[i + 1]
                    if current_pos[2] != next_pos[2]:  # Layer change = bump bond
                        ax.scatter([current_pos[0], next_pos[0]], 
                                  [current_pos[1], next_pos[1]], 
                                  c='green', s=50, marker='s', alpha=1.0, zorder=10)
            
            # Add TSV markers in red
            if tier.tsvs:
                tsv_x = [pos[0] for pos in tier.tsvs]
                tsv_y = [pos[1] for pos in tier.tsvs]
                ax.scatter(tsv_x, tsv_y, c='red', s=30, marker='x', alpha=0.8, zorder=8)
            
            ax.set_title(f'Tier {tier_id}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        # Hide unused subplots if we have a multi-row layout
        if rows * cols > n_tiers:
            for i in range(n_tiers, rows * cols):
                if i < len(axes):
                    axes[i].set_visible(False)
        
        logical_count = len(logical_routes)
        
        plt.suptitle(f'HAL Multi-Tier Layout - {len(layout.node_positions)} qubits, '
                    f'{logical_count} logical edges\n'
                    f'{len(layout.tiers)} tiers, Hardware Cost: {layout.hardware_cost:.2f}',
                    fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def _plot_single_tier_static(self, layout: QECCLayout) -> None:
        """Create static 2D matplotlib visualization for single tier."""
        plt.figure(figsize=(10, 8))
        
        # Plot edges first (so they appear behind nodes)
        for edge, path in layout.edge_routes.items():
            if not path:
                continue
            path_x = [pos[0] for pos in path]
            path_y = [pos[1] for pos in path]
            
            # Color edges by layer: black for layer 0, yellow for layer 1
            layer_counts = {}
            for pos in path:
                layer = pos[2] % 2  # Layer within tier (0 or 1)
                layer_counts[layer] = layer_counts.get(layer, 0) + 1
            
            # Use the layer where most of the path is
            primary_layer = max(layer_counts, key=layer_counts.get) if layer_counts else 0
            if primary_layer == 0:
                edge_color = 'black'  # Layer 0
            else:
                edge_color = 'yellow'  # Layer 1
            
            plt.plot(path_x, path_y, color=edge_color, linewidth=2, alpha=0.8)
            
            # Highlight bump bonds (layer changes) with green squares
            for i in range(len(path) - 1):
                current_pos = path[i]
                next_pos = path[i + 1]
                if current_pos[2] != next_pos[2]:  # Layer change = bump bond
                    plt.scatter([current_pos[0], next_pos[0]], 
                              [current_pos[1], next_pos[1]], 
                              c='green', s=50, marker='s', alpha=1.0, zorder=10)
        
        # Plot nodes (qubits) in blue
        node_x = [pos[0] for pos in layout.node_positions.values()]
        node_y = [pos[1] for pos in layout.node_positions.values()]
        node_ids = list(layout.node_positions.keys())
        
        plt.scatter(node_x, node_y, c='blue', s=100, zorder=5)
        
        # Add node labels
        for i, node_id in enumerate(node_ids):
            plt.annotate(str(node_id), (node_x[i], node_y[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add TSV markers if they exist
        if layout.tiers and layout.tiers[0].tsvs:
            tsv_x = [pos[0] for pos in layout.tiers[0].tsvs]
            tsv_y = [pos[1] for pos in layout.tiers[0].tsvs]
            plt.scatter(tsv_x, tsv_y, c='red', s=30, marker='x', alpha=0.8, zorder=8)
        
        plt.title(f"HAL Layout - {len(layout.node_positions)} qubits, "
                 f"{len(layout.edge_routes)} edges\n"
                 f"Hardware Cost: {layout.hardware_cost:.2f}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()