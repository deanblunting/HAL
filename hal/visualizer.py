"""
Simplified visualization components for HAL algorithm results - static mode only.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

from .data_structures import QECCLayout, RoutingTier
from .config import HALConfig
from .crossing_detector import CrossingDetector


class HALVisualizer:
    """Simplified visualization tools for HAL algorithm results."""
    
    def __init__(self, config: HALConfig = None):
        self.config = config or HALConfig()
        
    def plot_layout(self, layout: QECCLayout, n: int = None, k: int = None, d: int = None) -> None:
        """
        Create static visualization with separate tier plots.

        Args:
            layout: QECCLayout object to visualize
            n: Total number of qubits
            k: Number of logical qubits
            d: Minimum distance
        """
        if len(layout.tiers) > 1:
            self._plot_separate_tier_layouts_static(layout, n, k, d)
        else:
            self._plot_single_tier_static(layout, n, k, d)

    def _plot_separate_tier_layouts_static(self, layout: QECCLayout, n: int = None, k: int = None, d: int = None) -> None:
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
        
        # Get infrastructure routes from edge_routes if needed
        infrastructure_routes = []
        for edge, path in layout.edge_routes.items():
            if not path:
                continue
            u, v = edge
            if u >= 1000 or v >= 1000:
                infrastructure_routes.append((edge, path))

        # Get grid dimensions from tier 0 to use for all subplots
        tier0_grid_width, tier0_grid_height = layout.tiers[0].grid.shape[0], layout.tiers[0].grid.shape[1]

        for tier_id, tier in enumerate(layout.tiers):
            ax = axes[tier_id]

            # Use tier 0 grid size for plot bounds on all tiers
            grid_width, grid_height = tier0_grid_width, tier0_grid_height

            # Plot qubits (only on tier 0) - use blue color
            if tier_id == 0:
                node_x = [pos[0] for pos in layout.node_positions.values()]
                node_y = [pos[1] for pos in layout.node_positions.values()]
                node_ids = list(layout.node_positions.keys())

                ax.scatter(node_x, node_y, c='blue', s=70, zorder=5)

            # Plot edges from edge_routes for this tier
            for edge, route_info in layout.edge_routes.items():
                if not route_info or not route_info.get('path'):
                    continue

                path = route_info['path']
                edge_tier = route_info.get('tier', 0)

                # Only show edges that belong to this tier
                if edge_tier != tier_id:
                    continue

                # Use all path points since this edge belongs to this tier
                tier_path_points = [(x, y, layer) for x, y, layer in path]

                if len(tier_path_points) < 2:
                    continue

                # Group consecutive points by layer for proper line drawing
                current_layer = tier_path_points[0][2]
                current_segment = [tier_path_points[0]]

                for point in tier_path_points[1:]:
                    if point[2] == current_layer:
                        current_segment.append(point)
                    else:
                        # Draw the current segment
                        if len(current_segment) >= 2:
                            x_coords = [p[0] for p in current_segment]
                            y_coords = [p[1] for p in current_segment]
                            color = 'black' if current_layer == 0 else 'orange'
                            ax.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.8)

                        # Start new segment
                        current_layer = point[2]
                        current_segment = [current_segment[-1], point]  # Connect segments

                # Draw final segment
                if len(current_segment) >= 2:
                    x_coords = [p[0] for p in current_segment]
                    y_coords = [p[1] for p in current_segment]
                    color = 'black' if current_layer == 0 else 'orange'
                    ax.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.8)
            
            # Add intersection points using edge_routes data
            if hasattr(tier, 'crossing_detector') and tier.crossing_detector:
                # Convert edge_routes to segments for this tier
                segments_by_layer = {}
                for edge, route_info in layout.edge_routes.items():
                    if not route_info or not route_info.get('path'):
                        continue

                    path = route_info['path']
                    edge_tier = route_info.get('tier', 0)

                    # Only process edges that belong to this tier
                    if edge_tier != tier_id:
                        continue

                    # Filter path points for this tier and group by layer
                    for i in range(len(path) - 1):
                        current_point = path[i]
                        next_point = path[i + 1]

                        # Only include segments on the same layer
                        if current_point[2] == next_point[2]:
                            layer = current_point[2]
                            if layer not in segments_by_layer:
                                segments_by_layer[layer] = []

                            segment = ((current_point[0], current_point[1]),
                                     (next_point[0], next_point[1]))
                            segments_by_layer[layer].append(segment)

                # Find intersections for each layer
                for layer, segments in segments_by_layer.items():
                    if len(segments) > 1:
                        # Get qubit positions to exclude from intersection detection
                        qubit_positions = set(layout.node_positions.values()) if layout.node_positions else set()
                        intersections = tier.crossing_detector._find_intersections_from_segments(segments, qubit_positions)

                        if intersections:
                            # Filter out intersections where bump bonds already exist
                            bump_positions = set()
                            for edge, route_info in layout.edge_routes.items():
                                if not route_info or not route_info.get('path'):
                                    continue
                                edge_tier = route_info.get('tier', 0)
                                if edge_tier != tier_id:
                                    continue
                                path = route_info['path']
                                for i in range(len(path) - 1):
                                    if path[i][2] != path[i + 1][2]:  # Layer change = bump bond
                                        bump_positions.add((path[i][0], path[i][1]))

                            # Only show intersections that don't have bump bonds
                            # Use more precise coordinate matching
                            filtered_intersections = []
                            for pt in intersections:
                                has_bump = False
                                for bump_x, bump_y in bump_positions:
                                    # Use exact coordinate comparison (bump bonds should be at exact intersection points)
                                    if (abs(pt[0] - bump_x) < 1e-10 and abs(pt[1] - bump_y) < 1e-10) or \
                                       (pt[0] == bump_x and pt[1] == bump_y):
                                        has_bump = True
                                        break
                                if not has_bump:
                                    filtered_intersections.append(pt)

                            if filtered_intersections:
                                intersection_x = [pt[0] for pt in filtered_intersections]
                                intersection_y = [pt[1] for pt in filtered_intersections]
                                ax.scatter(intersection_x, intersection_y, c='magenta', s=50,
                                         marker='o', alpha=1.0, zorder=15, edgecolor='black', linewidth=1)

            # Add TSV markers in red
            if tier.tsvs:
                tsv_x = [pos[0] for pos in tier.tsvs]
                tsv_y = [pos[1] for pos in tier.tsvs]
                ax.scatter(tsv_x, tsv_y, c='red', s=15, marker='o', alpha=0.8, zorder=8)

            # Add bump bonds (layer transitions) as green squares
            bump_positions = []
            for edge, route_info in layout.edge_routes.items():
                if not route_info or not route_info.get('path'):
                    continue

                edge_tier = route_info.get('tier', 0)
                if edge_tier != tier_id:
                    continue

                path = route_info['path']
                # Find layer transitions (bump bonds)
                for i in range(len(path) - 1):
                    if path[i][2] != path[i + 1][2]:  # Layer change
                        bump_positions.append((path[i][0], path[i][1]))

            if bump_positions:
                bump_x = [pos[0] for pos in bump_positions]
                bump_y = [pos[1] for pos in bump_positions]
                ax.scatter(bump_x, bump_y, c='green', s=10, marker='s', alpha=0.8, zorder=10,
                          edgecolor='darkgreen', linewidth=1)

            
            ax.set_title(f'Tier {tier_id}')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_aspect('equal')
            # Set same axis limits for all tiers based on tier 0 grid with 20% margins
            margin_x = grid_width * 0.2
            margin_y = grid_height * 0.2
            ax.set_xlim(-0.5 - margin_x, grid_width - 0.5 + margin_x)
            ax.set_ylim(-0.5 - margin_y, grid_height - 0.5 + margin_y)
        
        # Hide unused subplots if we have a multi-row layout
        if rows * cols > n_tiers:
            for i in range(n_tiers, rows * cols):
                if i < len(axes):
                    axes[i].set_visible(False)
        
        # Calculate total logical routes from edge_routes
        logical_count = len(layout.edge_routes)

        # Add title with code parameters if provided
        if n is not None and k is not None and d is not None:
            plt.suptitle(f'[[{n},{k},{d}]]', fontsize=16, y=0.98)

        # Adjust layout to leave space for suptitle
        if n is not None and k is not None and d is not None:
            plt.tight_layout(rect=[0, 0, 1, 0.93])
        else:
            plt.tight_layout()
        plt.show()
    
    def _plot_single_tier_static(self, layout: QECCLayout, n: int = None, k: int = None, d: int = None) -> None:
        """Create static 2D matplotlib visualization for single tier."""
        plt.figure(figsize=(10, 8))
        
        # Plot edges from edge_routes
        for edge, route_info in layout.edge_routes.items():
            if not route_info or not route_info.get('path'):
                continue

            path = route_info['path']

            # Group consecutive points by layer for proper line drawing
            if len(path) < 2:
                continue

            current_layer = path[0][2]
            current_segment = [path[0]]

            for point in path[1:]:
                if point[2] == current_layer:
                    current_segment.append(point)
                else:
                    # Draw the current segment
                    if len(current_segment) >= 2:
                        x_coords = [p[0] for p in current_segment]
                        y_coords = [p[1] for p in current_segment]
                        color = 'black' if current_layer == 0 else 'orange'
                        plt.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.8)

                    # Start new segment
                    current_layer = point[2]
                    current_segment = [current_segment[-1], point]  # Connect segments

            # Draw final segment
            if len(current_segment) >= 2:
                x_coords = [p[0] for p in current_segment]
                y_coords = [p[1] for p in current_segment]
                color = 'black' if current_layer == 0 else 'orange'
                plt.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.8)
        
        # Plot nodes (qubits) in blue
        node_x = [pos[0] for pos in layout.node_positions.values()]
        node_y = [pos[1] for pos in layout.node_positions.values()]
        node_ids = list(layout.node_positions.keys())
        
        plt.scatter(node_x, node_y, c='blue', s=70, zorder=5)

        # Add intersection points using edge_routes data
        if layout.tiers and hasattr(layout.tiers[0], 'crossing_detector') and layout.tiers[0].crossing_detector:
            tier = layout.tiers[0]
            # Convert edge_routes to segments grouped by layer
            segments_by_layer = {}
            for edge, route_info in layout.edge_routes.items():
                if not route_info or not route_info.get('path'):
                    continue

                path = route_info['path']

                # Create segments from consecutive points on same layer
                for i in range(len(path) - 1):
                    current_point = path[i]
                    next_point = path[i + 1]

                    # Only include segments on the same layer
                    if current_point[2] == next_point[2]:
                        layer = current_point[2]
                        if layer not in segments_by_layer:
                            segments_by_layer[layer] = []

                        segment = ((current_point[0], current_point[1]),
                                 (next_point[0], next_point[1]))
                        segments_by_layer[layer].append(segment)

            # Find intersections for each layer
            for layer, segments in segments_by_layer.items():
                if len(segments) > 1:
                    # Get qubit positions to exclude from intersection detection
                    qubit_positions = set(layout.node_positions.values()) if layout.node_positions else set()
                    intersections = tier.crossing_detector._find_intersections_from_segments(segments, qubit_positions)
                    if intersections:
                        # Filter out intersections where bump bonds already exist
                        bump_positions = set()
                        for edge, route_info in layout.edge_routes.items():
                            if not route_info or not route_info.get('path'):
                                continue
                            path = route_info['path']
                            for i in range(len(path) - 1):
                                if path[i][2] != path[i + 1][2]:  # Layer change = bump bond
                                    bump_positions.add((path[i][0], path[i][1]))

                        # Only show intersections that don't have bump bonds
                        # Use more precise coordinate matching
                        filtered_intersections = []
                        for pt in intersections:
                            has_bump = False
                            for bump_x, bump_y in bump_positions:
                                # Use exact coordinate comparison (bump bonds should be at exact intersection points)
                                if (abs(pt[0] - bump_x) < 1e-10 and abs(pt[1] - bump_y) < 1e-10) or \
                                   (pt[0] == bump_x and pt[1] == bump_y):
                                    has_bump = True
                                    break
                            if not has_bump:
                                filtered_intersections.append(pt)

                        if filtered_intersections:
                            intersection_x = [pt[0] for pt in filtered_intersections]
                            intersection_y = [pt[1] for pt in filtered_intersections]
                            plt.scatter(intersection_x, intersection_y, c='magenta', s=50,
                                       marker='o', alpha=1.0, zorder=15, edgecolor='black', linewidth=1)

        # Add TSV markers if they exist
        if layout.tiers and layout.tiers[0].tsvs:
            tsv_x = [pos[0] for pos in layout.tiers[0].tsvs]
            tsv_y = [pos[1] for pos in layout.tiers[0].tsvs]
            plt.scatter(tsv_x, tsv_y, c='red', s=15, marker='o', alpha=0.8, zorder=8)

        # Add bump bonds (layer transitions) as green squares
        bump_positions = []
        for edge, route_info in layout.edge_routes.items():
            if not route_info or not route_info.get('path'):
                continue

            path = route_info['path']
            # Find layer transitions (bump bonds)
            for i in range(len(path) - 1):
                if path[i][2] != path[i + 1][2]:  # Layer change
                    bump_positions.append((path[i][0], path[i][1]))

        if bump_positions:
            bump_x = [pos[0] for pos in bump_positions]
            bump_y = [pos[1] for pos in bump_positions]
            plt.scatter(bump_x, bump_y, c='green', s=10, marker='s', alpha=0.8, zorder=10,
                       edgecolor='darkgreen', linewidth=1)
        
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.axis('equal')

        # Add 20% margins around the plot
        if layout.tiers:
            grid_width, grid_height = layout.tiers[0].grid.shape[0], layout.tiers[0].grid.shape[1]
            margin_x = grid_width * 0.2
            margin_y = grid_height * 0.2
            plt.xlim(-0.5 - margin_x, grid_width - 0.5 + margin_x)
            plt.ylim(-0.5 - margin_y, grid_height - 0.5 + margin_y)

        # Add title with code parameters if provided
        if n is not None and k is not None and d is not None:
            plt.title(f'[[{n},{k},{d}]]', fontsize=16)

        plt.show()