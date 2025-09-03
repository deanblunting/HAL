"""
Visualization components for HAL algorithm results.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
import networkx as nx

from .data_structures import QECCLayout, RoutingTier
from .config import HALConfig


class HALVisualizer:
    """Visualization tools for HAL algorithm results."""
    
    def __init__(self, config: HALConfig = None):
        self.config = config or HALConfig()
        
    def plot_layout(self, layout: QECCLayout, show_tiers: bool = True, 
                   interactive: bool = True, separate_tier_plots: bool = True) -> None:
        """
        Create visualization of multi-tier layout.
        
        Args:
            layout: QECCLayout object to visualize
            show_tiers: Whether to show multiple tiers
            interactive: Whether to create interactive plotly plot
            separate_tier_plots: Whether to create separate 2D plots for each tier (clearer than 3D)
        """
        if interactive:
            if show_tiers and separate_tier_plots and len(layout.tiers) > 1:
                self._plot_separate_tier_layouts(layout)
            else:
                self._plot_interactive_layout(layout, show_tiers)
        else:
            if show_tiers and separate_tier_plots and len(layout.tiers) > 1:
                self._plot_separate_tier_layouts_static(layout)
            else:
                self._plot_static_layout(layout, show_tiers)
    
    def _plot_interactive_layout(self, layout: QECCLayout, show_tiers: bool) -> None:
        """Create interactive plotly visualization."""
        if not layout.node_positions:
            print("No layout to visualize")
            return
        
        if show_tiers and len(layout.tiers) > 1:
            # Multi-tier 3D visualization
            fig = self._create_3d_plot(layout)
        else:
            # Single-tier 2D visualization
            fig = self._create_2d_plot(layout)
        
        fig.show()
    
    def _plot_static_layout(self, layout: QECCLayout, show_tiers: bool) -> None:
        """Create static matplotlib visualization."""
        if not layout.node_positions:
            print("No layout to visualize")
            return
        
        if show_tiers and len(layout.tiers) > 1:
            self._plot_static_3d(layout)
        else:
            self._plot_static_2d(layout)
        
        plt.show()
    
    def _create_3d_plot(self, layout: QECCLayout) -> go.Figure:
        """Create 3D plotly visualization for multi-tier layout."""
        fig = go.Figure()
        
        # Initialize tier-specific color palette
        tier_colors = px.colors.qualitative.Set1
        
        # Render auxiliary grid infrastructure for multi-tier superconducting implementation
        if layout.tiers and len(layout.tiers) > 0:
            tier0 = layout.tiers[0]
            grid_width, grid_height = tier0.grid.shape[0], tier0.grid.shape[1]
            
            # Generate grid line visualization for multi-tier superconducting methodology
            # Render vertical grid lines
            for x in range(grid_width + 1):
                fig.add_trace(go.Scatter3d(
                    x=[x, x], y=[0, grid_height], z=[0, 0],
                    mode='lines',
                    line=dict(width=1, color='lightgray'),
                    showlegend=False,
                    opacity=0.3
                ))
            
            # Render horizontal grid lines  
            for y in range(grid_height + 1):
                fig.add_trace(go.Scatter3d(
                    x=[0, grid_width], y=[y, y], z=[0, 0],
                    mode='lines',
                    line=dict(width=1, color='lightgray'),
                    showlegend=False,
                    opacity=0.3
                ))
            
            # Generate legend entry for auxiliary grid
            fig.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode='markers',
                marker=dict(size=0, color='lightgray'),
                name=f"Auxiliary Grid ({grid_width}×{grid_height})",
                showlegend=True
            ))
        
        # Render logical qubit positions on tier 0
        node_x = [pos[0] for pos in layout.node_positions.values()]
        node_y = [pos[1] for pos in layout.node_positions.values()]
        node_z = [0] * len(layout.node_positions)
        node_ids = list(layout.node_positions.keys())
        
        fig.add_trace(go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(size=8, color='red', symbol='circle'),
            text=node_ids,
            textposition="middle center",
            name="Logical Qubits (Tier 0)"
        ))
        
        # Plot edge routes across tiers
        logical_routes = []
        infrastructure_routes = []
        
        # Classify edge routes: logical connectivity versus auxiliary infrastructure
        for edge, path in layout.edge_routes.items():
            if not path:
                continue
            
            # Identify infrastructure routing by edge identifier range
            u, v = edge
            if u >= 1000 or v >= 1000:
                infrastructure_routes.append((edge, path))
            else:
                logical_routes.append((edge, path))
        
        # Plot logical qubit connections (main QECC connectivity)
        for edge, path in logical_routes:
            # Find which tier this edge was routed on
            tier_id = 0
            for i, tier in enumerate(layout.tiers):
                if edge in tier.edges:
                    tier_id = i
                    break
            
            color = tier_colors[tier_id % len(tier_colors)]
            
            path_x = [pos[0] for pos in path]
            path_y = [pos[1] for pos in path]
            
            # Show routing with clear tier separation for visualization
            if tier_id == 0:
                # Tier 0: Clean grid routing at z=0 (qubit tier)
                path_z = [0.0 for pos in path]
                line_width = 3
                opacity = 1.0
            else:
                # Higher tiers: Clear elevation for visual distinction
                base_z = tier_id * 1.0  # Full unit separation between tiers
                path_z = [base_z for pos in path]  # Keep paths flat within each tier
                line_width = 2
                opacity = 0.8
                
                # Show minimal TSV connections to indicate tier usage
                u, v = edge
                if u in layout.node_positions and v in layout.node_positions:
                    u_pos = layout.node_positions[u]
                    v_pos = layout.node_positions[v]
                    
                    # Subtle connection lines to show higher tier access
                    fig.add_trace(go.Scatter3d(
                        x=[u_pos[0], path_x[0]], 
                        y=[u_pos[1], path_y[0]], 
                        z=[0, path_z[0]],
                        mode='lines',
                        line=dict(width=1, color='lightgray', dash='dot'),
                        showlegend=False,
                        opacity=0.4
                    ))
                    fig.add_trace(go.Scatter3d(
                        x=[v_pos[0], path_x[-1]], 
                        y=[v_pos[1], path_y[-1]], 
                        z=[0, path_z[-1]],
                        mode='lines',
                        line=dict(width=1, color='lightgray', dash='dot'),
                        showlegend=False,
                        opacity=0.4
                    ))
            
            # Show the grid-based routing path
            fig.add_trace(go.Scatter3d(
                x=path_x, y=path_y, z=path_z,
                mode='lines+markers',
                line=dict(width=line_width, color=color),
                marker=dict(size=2, color=color, opacity=opacity),
                name=f"Edge {edge[0]}-{edge[1]} (Tier {tier_id})",
                showlegend=False,
                opacity=opacity
            ))
        
        # Plot auxiliary infrastructure routing for multi-tier superconducting methodology
        infrastructure_colors = {
            'control': 'green',    # Control lines
            'readout': 'orange',   # Readout lines  
            'tsv': 'purple',       # TSV connections
            'cross_tier': 'cyan'   # Cross-tier routing
        }
        
        for edge, path in infrastructure_routes:
            u, v = edge
            
            # Determine infrastructure type based on edge ID ranges
            if 1000 <= u < 2000:
                infra_type = 'control'
                line_style = 'dash'
            elif 2000 <= u < 4000:
                infra_type = 'readout'
                line_style = 'dot'
            elif 4000 <= u < 6000:
                infra_type = 'tsv'
                line_style = 'dashdot'
            else:
                infra_type = 'cross_tier'
                line_style = 'solid'
            
            color = infrastructure_colors.get(infra_type, 'gray')
            
            path_x = [pos[0] for pos in path]
            path_y = [pos[1] for pos in path]
            path_z = [pos[2] * 0.1 for pos in path]  # Slight elevation for infrastructure
            
            # Show auxiliary infrastructure with distinct styling
            fig.add_trace(go.Scatter3d(
                x=path_x, y=path_y, z=path_z,
                mode='lines',
                line=dict(width=1, color=color, dash=line_style),
                name=f"Infrastructure ({infra_type})",
                showlegend=infra_type not in [trace.name.split('(')[-1].rstrip(')') for trace in fig.data if hasattr(trace, 'name') and 'Infrastructure' in str(trace.name)],  # Show legend only once per type
                opacity=0.3
            ))
        
        # Add tier planes for reference
        if len(layout.tiers) > 1:
            for tier_id in range(len(layout.tiers)):
                tier = layout.tiers[tier_id]
                if tier.edges:  # Only if tier has routed edges
                    # Show occupied cells in the tier
                    occupied_cells = []
                    for x in range(tier.grid.shape[0]):
                        for y in range(tier.grid.shape[1]):
                            for layer in range(tier.grid.shape[2]):
                                if tier.is_occupied(x, y, layer):
                                    occupied_cells.append((x, y, tier_id))
                    
                    if occupied_cells:
                        x_coords = [cell[0] for cell in occupied_cells]
                        y_coords = [cell[1] for cell in occupied_cells]
                        z_coords = [tier_id * 1.0 for cell in occupied_cells]  # Use proper tier elevation
                        
                        fig.add_trace(go.Scatter3d(
                            x=x_coords, y=y_coords, z=z_coords,
                            mode='markers',
                            marker=dict(size=2, color=tier_colors[tier_id % len(tier_colors)], opacity=0.3),
                            name=f"Tier {tier_id} routing",
                            showlegend=tier_id < 5  # Show legend for first few tiers only
                        ))
        
        # Count logical vs infrastructure routes
        logical_count = len(logical_routes)
        infrastructure_count = len(infrastructure_routes)
        
        fig.update_layout(
            title=f"HAL Layout - {len(layout.node_positions)} qubits, "
                  f"{logical_count} logical edges, {infrastructure_count} infrastructure routes<br>"
                  f"{len(layout.tiers)} tiers, Hardware Cost: {layout.hardware_cost:.2f}<br>"
                  f"<i>Auxiliary infrastructure routing for multi-tier superconducting methodology</i>",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y", 
                zaxis_title="Tier",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def _plot_separate_tier_layouts(self, layout: QECCLayout) -> None:
        """Create separate 2D interactive plots for each tier (much clearer than 3D)."""
        if not layout.node_positions:
            print("No layout to visualize")
            return
        
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
        
        # Create a subplot for each tier with multi-row layout for better visibility
        n_tiers = len(layout.tiers)
        if n_tiers <= 5:
            rows, cols = 1, n_tiers
        else:
            rows = 2
            cols = (n_tiers + 1) // 2  # Ceiling division
        
        # Create subplot grid
        subplot_titles = [f"Tier {i}" for i in range(n_tiers)]
        # Pad with empty titles for unused subplots
        while len(subplot_titles) < rows * cols:
            subplot_titles.append("")
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            specs=[[{"type": "scatter"} for _ in range(cols)] for _ in range(rows)]
        )
        
        for tier_id, tier in enumerate(layout.tiers):
            # Calculate row and column for this tier
            row = (tier_id // cols) + 1
            col = (tier_id % cols) + 1
            
            # Show auxiliary grid lines for this tier
            grid_width, grid_height = tier.grid.shape[0], tier.grid.shape[1]
            
            # Add grid lines
            for x in range(grid_width + 1):
                fig.add_trace(go.Scatter(
                    x=[x, x], y=[0, grid_height],
                    mode='lines',
                    line=dict(width=1, color='lightgray'),
                    showlegend=False,
                    opacity=0.3
                ), row=row, col=col)
            
            for y in range(grid_height + 1):
                fig.add_trace(go.Scatter(
                    x=[0, grid_width], y=[y, y],
                    mode='lines',
                    line=dict(width=1, color='lightgray'),
                    showlegend=False,
                    opacity=0.3
                ), row=row, col=col)
            
            # Plot qubits (only on tier 0)
            if tier_id == 0:
                node_x = [pos[0] for pos in layout.node_positions.values()]
                node_y = [pos[1] for pos in layout.node_positions.values()]
                node_ids = list(layout.node_positions.keys())
                
                fig.add_trace(go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    marker=dict(size=10, color='red', symbol='circle'),
                    text=node_ids,
                    textposition="middle center",
                    name="Logical Qubits",
                    showlegend=(tier_id == 0)
                ), row=row, col=col)
            
            # Plot logical routes for this tier
            tier_colors = px.colors.qualitative.Set1
            for edge, path in logical_routes:
                # Check if this edge belongs to this tier
                if edge not in tier.edges:
                    continue
                
                path_x = [pos[0] for pos in path if pos[2] == 0]  # Only show paths on layer 0
                path_y = [pos[1] for pos in path if pos[2] == 0]
                
                if len(path_x) > 1:
                    color = tier_colors[tier_id % len(tier_colors)]
                    fig.add_trace(go.Scatter(
                        x=path_x, y=path_y,
                        mode='lines+markers',
                        line=dict(width=3, color=color),
                        marker=dict(size=4, color=color),
                        name=f"T{tier_id} Edge {edge[0]}-{edge[1]}",
                        showlegend=False
                    ), row=row, col=col)
            
            # Plot infrastructure routes for this tier
            infrastructure_colors = {
                'control': 'green',
                'readout': 'orange',
                'tsv': 'purple',
                'cross_tier': 'cyan'
            }
            
            for edge, path in infrastructure_routes:
                u, v = edge
                
                # Filter paths that should appear on this tier
                tier_z_level = tier_id * 0.1
                relevant_path = [(pos[0], pos[1]) for pos in path if abs(pos[2] - tier_z_level) < 0.05]
                
                if len(relevant_path) < 2:
                    continue
                
                # Determine infrastructure type
                if 1000 <= u < 2000:
                    infra_type = 'control'
                    line_style = 'dash'
                elif 2000 <= u < 4000:
                    infra_type = 'readout'
                    line_style = 'dot'
                elif 4000 <= u < 6000:
                    infra_type = 'tsv'
                    line_style = 'dashdot'
                else:
                    infra_type = 'cross_tier'
                    line_style = 'solid'
                
                color = infrastructure_colors.get(infra_type, 'gray')
                
                path_x = [pos[0] for pos in relevant_path]
                path_y = [pos[1] for pos in relevant_path]
                
                fig.add_trace(go.Scatter(
                    x=path_x, y=path_y,
                    mode='lines',
                    line=dict(width=1, color=color, dash=line_style),
                    name=f"Infrastructure ({infra_type})",
                    showlegend=(tier_id == 0 and infra_type not in [trace.name.split('(')[-1].rstrip(')') for trace in fig.data if hasattr(trace, 'name') and 'Infrastructure' in str(trace.name)]),
                    opacity=0.4
                ), row=row, col=col)
            
            # Show occupied cells in this tier
            occupied_cells = []
            for x in range(tier.grid.shape[0]):
                for y in range(tier.grid.shape[1]):
                    for layer in range(tier.grid.shape[2]):
                        if tier.is_occupied(x, y, layer):
                            occupied_cells.append((x, y))
            
            if occupied_cells:
                x_coords = [cell[0] for cell in occupied_cells]
                y_coords = [cell[1] for cell in occupied_cells]
                
                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords,
                    mode='markers',
                    marker=dict(size=3, color=tier_colors[tier_id % len(tier_colors)], opacity=0.5),
                    name=f"Tier {tier_id} occupied",
                    showlegend=(tier_id < 3)
                ), row=row, col=col)
        
        # Count routes
        logical_count = len(logical_routes)
        infrastructure_count = len(infrastructure_routes)
        
        # Calculate better sizing for multi-row layout
        subplot_width = 400  # Wider subplots for better visibility
        subplot_height = 400
        total_width = subplot_width * cols
        total_height = subplot_height * rows
        
        fig.update_layout(
            title=f"HAL Multi-Tier Layout - {len(layout.node_positions)} qubits, "
                  f"{logical_count} logical edges, {infrastructure_count} infrastructure routes<br>"
                  f"{len(layout.tiers)} tiers, Hardware Cost: {layout.hardware_cost:.2f}<br>"
                  f"<i>Each tier shown separately for enhanced visualization clarity</i>",
            height=total_height,
            width=total_width
        )
        
        # Update axes for each subplot
        for i in range(n_tiers):
            # Calculate row and column for this tier
            subplot_row = (i // cols) + 1
            subplot_col = (i % cols) + 1
            fig.update_xaxes(title_text="X", row=subplot_row, col=subplot_col)
            fig.update_yaxes(title_text="Y", row=subplot_row, col=subplot_col)
        
        fig.show()
    
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
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_tiers))
        
        for tier_id, tier in enumerate(layout.tiers):
            ax = axes[tier_id]
            
            # Show grid
            grid_width, grid_height = tier.grid.shape[0], tier.grid.shape[1]
            
            # Grid lines
            for x in range(grid_width + 1):
                ax.plot([x, x], [0, grid_height], 'lightgray', alpha=0.3, linewidth=0.5)
            for y in range(grid_height + 1):
                ax.plot([0, grid_width], [y, y], 'lightgray', alpha=0.3, linewidth=0.5)
            
            # Plot qubits (only on tier 0)
            if tier_id == 0:
                node_x = [pos[0] for pos in layout.node_positions.values()]
                node_y = [pos[1] for pos in layout.node_positions.values()]
                node_ids = list(layout.node_positions.keys())
                
                ax.scatter(node_x, node_y, c='red', s=100, zorder=5)
                for i, node_id in enumerate(node_ids):
                    ax.annotate(str(node_id), (node_x[i], node_y[i]), 
                               xytext=(2, 2), textcoords='offset points', fontsize=8)
            
            # Plot logical routes for this tier
            for edge, path in logical_routes:
                if edge not in tier.edges:
                    continue
                
                path_x = [pos[0] for pos in path if pos[2] == 0]
                path_y = [pos[1] for pos in path if pos[2] == 0]
                
                if len(path_x) > 1:
                    ax.plot(path_x, path_y, color=colors[tier_id], linewidth=2, alpha=0.8)
            
            # Plot infrastructure routes
            infrastructure_colors = {
                'control': 'green',
                'readout': 'orange', 
                'tsv': 'purple',
                'cross_tier': 'cyan'
            }
            
            for edge, path in infrastructure_routes:
                u, v = edge
                
                tier_z_level = tier_id * 0.1
                relevant_path = [(pos[0], pos[1]) for pos in path if abs(pos[2] - tier_z_level) < 0.05]
                
                if len(relevant_path) < 2:
                    continue
                
                # Determine type
                if 1000 <= u < 2000:
                    infra_type = 'control'
                    line_style = '--'
                elif 2000 <= u < 4000:
                    infra_type = 'readout'
                    line_style = ':'
                elif 4000 <= u < 6000:
                    infra_type = 'tsv'
                    line_style = '-.'
                else:
                    infra_type = 'cross_tier'
                    line_style = '-'
                
                color = infrastructure_colors.get(infra_type, 'gray')
                path_x = [pos[0] for pos in relevant_path]
                path_y = [pos[1] for pos in relevant_path]
                
                ax.plot(path_x, path_y, color=color, linestyle=line_style, 
                       linewidth=1, alpha=0.4)
            
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
        infrastructure_count = len(infrastructure_routes)
        
        plt.suptitle(f'HAL Multi-Tier Layout - {len(layout.node_positions)} qubits, '
                    f'{logical_count} logical edges, {infrastructure_count} infrastructure routes\n'
                    f'{len(layout.tiers)} tiers, Hardware Cost: {layout.hardware_cost:.2f}',
                    fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def _create_2d_plot(self, layout: QECCLayout) -> go.Figure:
        """Create 2D plotly visualization for single-tier layout."""
        fig = go.Figure()
        
        # Plot nodes
        node_x = [pos[0] for pos in layout.node_positions.values()]
        node_y = [pos[1] for pos in layout.node_positions.values()]
        node_ids = list(layout.node_positions.keys())
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(size=10, color='red', symbol='circle'),
            text=node_ids,
            textposition="middle center",
            name="Qubits"
        ))
        
        # Plot edges
        for edge, path in layout.edge_routes.items():
            if not path:
                continue
            
            path_x = [pos[0] for pos in path]
            path_y = [pos[1] for pos in path]
            
            fig.add_trace(go.Scatter(
                x=path_x, y=path_y,
                mode='lines',
                line=dict(width=2, color='blue'),
                name=f"Edge {edge[0]}-{edge[1]}",
                showlegend=False
            ))
        
        fig.update_layout(
            title=f"HAL Layout - {len(layout.node_positions)} qubits, "
                  f"{len(layout.edge_routes)} edges<br>"
                  f"Hardware Cost: {layout.hardware_cost:.2f}",
            xaxis_title="X",
            yaxis_title="Y",
            width=600,
            height=600
        )
        
        return fig
    
    def _plot_static_2d(self, layout: QECCLayout) -> None:
        """Create static 2D matplotlib visualization."""
        plt.figure(figsize=(10, 8))
        
        # Plot edges first (so they appear behind nodes)
        for edge, path in layout.edge_routes.items():
            if not path:
                continue
            path_x = [pos[0] for pos in path]
            path_y = [pos[1] for pos in path]
            plt.plot(path_x, path_y, 'b-', linewidth=1, alpha=0.6)
        
        # Plot nodes
        node_x = [pos[0] for pos in layout.node_positions.values()]
        node_y = [pos[1] for pos in layout.node_positions.values()]
        node_ids = list(layout.node_positions.keys())
        
        plt.scatter(node_x, node_y, c='red', s=100, zorder=5)
        
        # Add node labels
        for i, node_id in enumerate(node_ids):
            plt.annotate(str(node_id), (node_x[i], node_y[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.title(f"HAL Layout - {len(layout.node_positions)} qubits, "
                 f"{len(layout.edge_routes)} edges\n"
                 f"Hardware Cost: {layout.hardware_cost:.2f}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
    
    def _plot_static_3d(self, layout: QECCLayout) -> None:
        """Create static 3D matplotlib visualization."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Render auxiliary grid infrastructure for multi-tier superconducting implementation
        if layout.tiers and len(layout.tiers) > 0:
            tier0 = layout.tiers[0]
            grid_width, grid_height = tier0.grid.shape[0], tier0.grid.shape[1]
            
            # Add grid lines for clean visualization
            for x in range(grid_width + 1):
                ax.plot([x, x], [0, grid_height], [0, 0], 'lightgray', alpha=0.3, linewidth=0.5)
            for y in range(grid_height + 1):
                ax.plot([0, grid_width], [y, y], [0, 0], 'lightgray', alpha=0.3, linewidth=0.5)
            
            # Add dummy point for legend
            ax.plot([], [], [], 'lightgray', alpha=0.3, linewidth=0.5, label=f'Auxiliary Grid ({grid_width}×{grid_height})')
        
        # Plot nodes (qubits)
        node_x = [pos[0] for pos in layout.node_positions.values()]
        node_y = [pos[1] for pos in layout.node_positions.values()]
        node_z = [0] * len(layout.node_positions)
        
        ax.scatter(node_x, node_y, node_z, c='red', s=100, label='Logical Qubits')
        
        # Plot edges
        colors = plt.cm.tab10(np.linspace(0, 1, len(layout.tiers)))
        
        for edge, path in layout.edge_routes.items():
            if not path:
                continue
            
            # Find which tier this edge was routed on
            tier_id = 0
            for i, tier in enumerate(layout.tiers):
                if edge in tier.edges:
                    tier_id = i
                    break
            
            path_x = [pos[0] for pos in path]
            path_y = [pos[1] for pos in path]
            
            # Match the interactive visualization approach with clear tier separation
            if tier_id == 0:
                path_z = [0.0 for pos in path]  # Tier 0 at z=0
                linewidth = 2
                alpha = 0.8
            else:
                base_z = tier_id * 1.0  # Full unit separation between tiers
                path_z = [base_z for pos in path]  # Keep paths flat within each tier
                linewidth = 1.5
                alpha = 0.6
            
            color = colors[tier_id % len(colors)]
            ax.plot(path_x, path_y, path_z, color=color, linewidth=linewidth, alpha=alpha)
        
        ax.set_title(f"HAL Layout - {len(layout.node_positions)} qubits, "
                    f"{len(layout.edge_routes)} edges, {len(layout.tiers)} tiers\n"
                    f"Hardware Cost: {layout.hardware_cost:.2f}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Tier")
    
    def plot_cost_analysis(self, layout: QECCLayout, show_breakdown: bool = True) -> None:
        """Plot cost analysis and metrics breakdown."""
        if show_breakdown:
            self._plot_cost_breakdown(layout)
        else:
            self._plot_cost_summary(layout)
    
    def _plot_cost_breakdown(self, layout: QECCLayout) -> None:
        """Create detailed cost breakdown visualization."""
        metrics = layout.metrics
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Metric values
        metric_names = ['Tiers', 'Avg Length', 'Avg Bumps', 'Avg TSVs']
        metric_values = [
            metrics.get('tiers', 0),
            metrics.get('length', 0),
            metrics.get('bumps', 0),
            metrics.get('tsvs', 0)
        ]
        
        ax1.bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'orange', 'pink'])
        ax1.set_title('Raw Metric Values')
        ax1.set_ylabel('Value')
        
        # Cost contribution (normalized)
        from .cost import HardwareCostCalculator
        calculator = HardwareCostCalculator(self.config)
        breakdown = calculator.calculate_detailed_cost_breakdown(metrics)
        
        normalized_values = [
            breakdown.get('tiers_normalized', 0),
            breakdown.get('length_normalized', 0), 
            breakdown.get('bumps_normalized', 0),
            breakdown.get('tsvs_normalized', 0)
        ]
        
        ax2.bar(metric_names, normalized_values, color=['skyblue', 'lightgreen', 'orange', 'pink'])
        ax2.set_title('Normalized Cost Contributions')
        ax2.set_ylabel('Normalized Cost')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Cost weights
        weights = [
            self.config.cost_weights.get('tiers', 0),
            self.config.cost_weights.get('length', 0),
            self.config.cost_weights.get('bumps', 0),
            self.config.cost_weights.get('tsvs', 0)
        ]
        
        ax3.bar(metric_names, weights, color=['skyblue', 'lightgreen', 'orange', 'pink'])
        ax3.set_title('Cost Weights')
        ax3.set_ylabel('Weight')
        
        # Final weighted contributions
        weighted_values = [
            breakdown.get('tiers_weighted', 0),
            breakdown.get('length_weighted', 0),
            breakdown.get('bumps_weighted', 0),
            breakdown.get('tsvs_weighted', 0)
        ]
        
        ax4.bar(metric_names, weighted_values, color=['skyblue', 'lightgreen', 'orange', 'pink'])
        ax4.set_title('Weighted Cost Contributions')
        ax4.set_ylabel('Weighted Cost')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.suptitle(f'Cost Analysis - Total Hardware Cost: {layout.hardware_cost:.2f}')
        plt.tight_layout()
        plt.show()
    
    def _plot_cost_summary(self, layout: QECCLayout) -> None:
        """Create summary cost visualization."""
        metrics = layout.metrics
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Metrics radar chart
        categories = ['Tiers', 'Length', 'Bumps', 'TSVs']
        values = [
            metrics.get('tiers', 0),
            metrics.get('length', 0),
            metrics.get('bumps', 0),
            metrics.get('tsvs', 0)
        ]
        
        # Normalize values for radar chart
        max_vals = [10, 20, 5, 3]  # Reasonable max values for visualization
        normalized_values = [min(v/m, 1.0) for v, m in zip(values, max_vals)]
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        values_plot = normalized_values + [normalized_values[0]]  # Close the circle
        angles_plot = np.concatenate([angles, [angles[0]]])
        
        ax1.plot(angles_plot, values_plot, 'o-', linewidth=2, label='Current Layout')
        ax1.fill(angles_plot, values_plot, alpha=0.25)
        ax1.set_xticks(angles)
        ax1.set_xticklabels(categories)
        ax1.set_ylim(0, 1)
        ax1.set_title('Metrics Profile')
        ax1.grid(True)
        
        # Cost comparison bar
        ax2.bar(['Hardware Cost'], [layout.hardware_cost], color='steelblue', width=0.5)
        ax2.axhline(y=1.0, color='red', linestyle='--', label='Baseline Cost')
        ax2.set_title('Total Hardware Cost')
        ax2.set_ylabel('Cost')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def save_layout(self, layout: QECCLayout, filename: str) -> None:
        """Save layout visualization to file."""
        if filename.endswith('.html'):
            # Save interactive plot
            fig = self._create_2d_plot(layout) if len(layout.tiers) <= 1 else self._create_3d_plot(layout)
            fig.write_html(filename)
        else:
            # Save static plot
            self._plot_static_layout(layout, show_tiers=len(layout.tiers) > 1)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
    
    def compare_layouts(self, layouts: List[QECCLayout], labels: List[str] = None) -> None:
        """Create comparison visualization of multiple layouts."""
        if not layouts:
            return
        
        if labels is None:
            labels = [f"Layout {i+1}" for i in range(len(layouts))]
        
        # Cost comparison
        costs = [layout.hardware_cost for layout in layouts]
        
        plt.figure(figsize=(12, 8))
        
        # Cost comparison bar chart
        plt.subplot(2, 2, 1)
        bars = plt.bar(labels, costs, color='steelblue')
        plt.title('Hardware Cost Comparison')
        plt.ylabel('Hardware Cost')
        plt.xticks(rotation=45)
        
        # Add cost values on bars
        for bar, cost in zip(bars, costs):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{cost:.2f}', ha='center', va='bottom')
        
        # Metrics comparison - only show first 3 to fit in 2x2 subplot
        metrics_to_plot = ['tiers', 'length', 'bumps']
        metric_labels = ['Tiers', 'Avg Length', 'Avg Bumps']
        
        for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
            plt.subplot(2, 2, i+2)
            values = [layout.metrics.get(metric, 0) for layout in layouts]
            plt.bar(labels, values, color=plt.cm.Set3(i))
            plt.title(f'{label} Comparison')
            plt.ylabel(label)
            plt.xticks(rotation=45)
            
            # Add values on bars
            for j, (bar, value) in enumerate(zip(plt.gca().patches, values)):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.show()