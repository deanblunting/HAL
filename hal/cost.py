"""
Hardware cost calculation for HAL algorithm.
"""

import numpy as np
from typing import Dict, List
from .config import HALConfig
from .data_structures import QECCLayout, RoutingResult


class HardwareCostCalculator:
    """Calculate hardware implementation cost from layout metrics."""
    
    def __init__(self, config: HALConfig):
        self.config = config
        
    def calculate_cost(self, metrics: Dict[str, float]) -> float:
        """
        Compute normalized hardware implementation cost using the four-metric cost model.
        
        Paper formula (Section V): 
        ci = (qi - bi) / (pi - bi)
        Chw = 1 + (sum(wi*ci)) / (sum(wi))
        
        Where:
        - ci is the individual cost for metric i
        - qi is the raw quantity for metric i  
        - bi is the baseline value (surface code requirements)
        - pi is the optimistic value (advanced fabrication capabilities)
        - wi is the weight for metric i
        - Chw = 1 denotes ideal planar, single-tier, nearest-neighbor layout
        
        Args:
            metrics: Dictionary containing 'tiers', 'length', 'bumps', 'tsvs'
            
        Returns:
            Hardware cost as float (>= 1.0)
        """
        # Extract raw performance metrics from layout results
        qi_values = {}
        for metric_name in ['tiers', 'length', 'bumps', 'tsvs']:
            qi_values[metric_name] = metrics.get(metric_name, 0.0)
        
        # Compute normalized cost components using baseline rescaling 
        ci_values = {}
        for metric_name in ['tiers', 'length', 'bumps', 'tsvs']:
            qi = qi_values[metric_name]
            bi = self.config.cost_baselines[metric_name]  # baseline value
            pi = self.config.cost_optimistic[metric_name]  # optimistic value
            
            # Apply linear normalization transformation
            if pi == bi:
                ci_values[metric_name] = 0.0  # Handle degenerate case
            else:
                ci_values[metric_name] = (qi - bi) / (pi - bi)
        
        # Compute weighted arithmetic mean using the cost model from Section V
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric_name in ['tiers', 'length', 'bumps', 'tsvs']:
            wi = self.config.cost_weights[metric_name]
            ci = ci_values[metric_name]
            
            weighted_sum += wi * ci
            total_weight += wi
        
        # Calculate final normalized hardware cost
        if total_weight == 0:
            Chw = 1.0  # Default to optimal case for zero weights
        else:
            Chw = 1.0 + weighted_sum / total_weight
        
        # Enforce minimum cost constraint (optimal planar baseline)
        return max(Chw, 1.0)
    
    def calculate_detailed_cost_breakdown(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate detailed cost breakdown showing contribution of each metric.
        
        Args:
            metrics: Dictionary containing 'tiers', 'length', 'bumps', 'tsvs'
            
        Returns:
            Dictionary with normalized costs and weights for each metric
        """
        breakdown = {}
        
        for metric_name in ['tiers', 'length', 'bumps', 'tsvs']:
            if metric_name not in metrics:
                breakdown[f'{metric_name}_raw'] = 0.0
                breakdown[f'{metric_name}_normalized'] = 0.0
                breakdown[f'{metric_name}_weighted'] = 0.0
                continue
            
            value = metrics[metric_name]
            baseline = self.config.cost_baselines[metric_name]
            optimistic = self.config.cost_optimistic[metric_name]
            weight = self.config.cost_weights[metric_name]
            
            # Raw value
            breakdown[f'{metric_name}_raw'] = value
            
            # Normalized value
            if optimistic == baseline:
                normalized = 0.0
            else:
                normalized = (value - baseline) / (optimistic - baseline)
            breakdown[f'{metric_name}_normalized'] = normalized
            
            # Weighted value
            breakdown[f'{metric_name}_weighted'] = weight * normalized
        
        # Total cost
        total_weight = sum(self.config.cost_weights.values())
        if total_weight > 0:
            weighted_sum = sum(breakdown[f'{m}_weighted'] for m in ['tiers', 'length', 'bumps', 'tsvs'])
            breakdown['total_cost'] = 1.0 + weighted_sum / total_weight
        else:
            breakdown['total_cost'] = 1.0
        
        return breakdown
    
    def compare_layouts(self, layouts: List[QECCLayout], labels: List[str] = None) -> Dict[str, any]:
        """
        Compare multiple layouts and rank by hardware cost.
        
        Args:
            layouts: List of QECCLayout objects to compare
            labels: Optional labels for each layout
            
        Returns:
            Dictionary with comparison results
        """
        if not layouts:
            return {}
        
        if labels is None:
            labels = [f"Layout_{i+1}" for i in range(len(layouts))]
        
        # Calculate costs and breakdowns
        results = []
        for i, layout in enumerate(layouts):
            cost = self.calculate_cost(layout.metrics)
            breakdown = self.calculate_detailed_cost_breakdown(layout.metrics)
            
            results.append({
                'label': labels[i],
                'cost': cost,
                'metrics': layout.metrics.copy(),
                'breakdown': breakdown
            })
        
        # Sort by cost (lower is better)
        results.sort(key=lambda x: x['cost'])
        
        # Add rankings
        for i, result in enumerate(results):
            result['rank'] = i + 1
        
        comparison = {
            'results': results,
            'best_layout': results[0] if results else None,
            'cost_range': {
                'min': min(r['cost'] for r in results),
                'max': max(r['cost'] for r in results),
                'span': max(r['cost'] for r in results) - min(r['cost'] for r in results)
            } if results else None
        }
        
        return comparison
    
    def analyze_cost_sensitivity(self, metrics: Dict[str, float], 
                                perturbation: float = 0.1) -> Dict[str, float]:
        """
        Analyze sensitivity of cost to changes in each metric.
        
        Args:
            metrics: Base metrics
            perturbation: Fractional change to apply for sensitivity analysis
            
        Returns:
            Dictionary with sensitivity values for each metric
        """
        base_cost = self.calculate_cost(metrics)
        sensitivities = {}
        
        for metric_name in ['tiers', 'length', 'bumps', 'tsvs']:
            if metric_name not in metrics:
                sensitivities[metric_name] = 0.0
                continue
            
            # Perturb metric upward
            perturbed_metrics = metrics.copy()
            original_value = metrics[metric_name]
            perturbed_metrics[metric_name] = original_value * (1 + perturbation)
            
            perturbed_cost = self.calculate_cost(perturbed_metrics)
            
            # Calculate sensitivity (cost change per unit metric change)
            if original_value > 0:
                sensitivity = (perturbed_cost - base_cost) / (original_value * perturbation)
            else:
                sensitivity = 0.0
            
            sensitivities[metric_name] = sensitivity
        
        return sensitivities
    
    def suggest_improvements(self, metrics: Dict[str, float]) -> List[str]:
        """
        Suggest improvements based on cost analysis.
        
        Args:
            metrics: Current layout metrics
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        breakdown = self.calculate_detailed_cost_breakdown(metrics)
        
        # Find metrics with highest weighted contribution
        weighted_contributions = {
            'tiers': breakdown.get('tiers_weighted', 0),
            'length': breakdown.get('length_weighted', 0), 
            'bumps': breakdown.get('bumps_weighted', 0),
            'tsvs': breakdown.get('tsvs_weighted', 0)
        }
        
        # Sort by contribution (highest impact first)
        sorted_metrics = sorted(weighted_contributions.items(), 
                              key=lambda x: abs(x[1]), reverse=True)
        
        for metric_name, contribution in sorted_metrics:
            if contribution > 0.1:  # Only suggest if significant contribution
                if metric_name == 'tiers':
                    suggestions.append(f"Reduce number of routing tiers (current: {metrics.get('tiers', 0):.1f})")
                elif metric_name == 'length':
                    suggestions.append(f"Reduce average edge length (current: {metrics.get('length', 0):.2f})")
                elif metric_name == 'bumps':
                    suggestions.append(f"Reduce bump bond transitions (current: {metrics.get('bumps', 0):.2f} per edge)")
                elif metric_name == 'tsvs':
                    suggestions.append(f"Reduce through-silicon vias (current: {metrics.get('tsvs', 0):.2f} per edge)")
        
        if not suggestions:
            suggestions.append("Layout is well-optimized for current cost model")
        
        return suggestions