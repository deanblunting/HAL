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

from data_structures import QECCLayout, RoutingTier
from config import HALConfig


class HALVisualizer:
    """Visualization tools for HAL algorithm results."""
    
    def __init__(self, config: HALConfig = None):
        self.config = config or HALConfig()
        
    def plot_layout(self, layout: QECCLayout, show_tiers: bool = True, 
                   interactive: bool = True) -> None:
        """
        Create visualization of multi-tier layout.
        
        Args:
            layout: QECCLayout object to visualize
            show_tiers: Whether to show multiple tiers
            interactive: Whether to create interactive plotly plot
        """
        if interactive:
            self._plot_interactive_layout(layout, show_tiers)
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
        
        # Color palette for different tiers
        tier_colors = px.colors.qualitative.Set1
        
        # Plot nodes (qubits) on tier 0
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
            name="Qubits (Tier 0)"
        ))
        
        # Plot edge routes across tiers
        for edge, path in layout.edge_routes.items():
            if not path:
                continue
            
            # Determine tier from path z-coordinates
            tier_id = max(pos[2] for pos in path) if path else 0
            color = tier_colors[tier_id % len(tier_colors)]
            
            path_x = [pos[0] for pos in path]
            path_y = [pos[1] for pos in path]
            path_z = [pos[2] for pos in path]
            
            fig.add_trace(go.Scatter3d(
                x=path_x, y=path_y, z=path_z,
                mode='lines+markers',
                line=dict(width=3, color=color),
                marker=dict(size=3, color=color),
                name=f"Edge {edge[0]}-{edge[1]} (Tier {tier_id})",
                showlegend=False
            ))
        
        # Add tier planes for reference
        if len(layout.tiers) > 1:
            for tier_id in range(len(layout.tiers)):
                if layout.tiers[tier_id]:  # Only if tier has content
                    occupied_cells = list(layout.tiers[tier_id])
                    if occupied_cells:
                        x_coords = [cell[0] for cell in occupied_cells]
                        y_coords = [cell[1] for cell in occupied_cells]
                        z_coords = [tier_id] * len(occupied_cells)
                        
                        fig.add_trace(go.Scatter3d(
                            x=x_coords, y=y_coords, z=z_coords,
                            mode='markers',
                            marker=dict(size=2, color=tier_colors[tier_id % len(tier_colors)], opacity=0.3),
                            name=f"Tier {tier_id} routing",
                            showlegend=tier_id < 5  # Show legend for first few tiers only
                        ))
        
        fig.update_layout(
            title=f"HAL Layout - {len(layout.node_positions)} qubits, "
                  f"{len(layout.edge_routes)} edges, "
                  f"{len(layout.tiers)} tiers<br>"
                  f"Hardware Cost: {layout.hardware_cost:.2f}",
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
        
        # Plot nodes
        node_x = [pos[0] for pos in layout.node_positions.values()]
        node_y = [pos[1] for pos in layout.node_positions.values()]
        node_z = [0] * len(layout.node_positions)
        
        ax.scatter(node_x, node_y, node_z, c='red', s=100, label='Qubits')
        
        # Plot edges
        colors = plt.cm.tab10(np.linspace(0, 1, len(layout.tiers)))
        
        for edge, path in layout.edge_routes.items():
            if not path:
                continue
            
            path_x = [pos[0] for pos in path]
            path_y = [pos[1] for pos in path]
            path_z = [pos[2] for pos in path]
            
            tier_id = max(path_z)
            color = colors[tier_id % len(colors)]
            
            ax.plot(path_x, path_y, path_z, color=color, linewidth=2, alpha=0.7)
        
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
        from cost import HardwareCostCalculator
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