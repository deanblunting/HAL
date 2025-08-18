"""
Configuration classes for HAL algorithm.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class HALConfig:
    """Comprehensive HAL algorithm configuration with optimization parameters."""
    
    # Multi-tier routing configuration
    max_tiers: int = 10
    max_bump_transitions: int = 10
    max_tsvs_per_edge: int = 3
    edge_margin: int = 1
    node_size: int = 1
    grid_size: Tuple[int, int] = (100, 100)
    
    # Node placement optimization parameters
    spring_layout_iterations: int = 50
    spring_layout_k: float = None  # automatic spring constant calculation
    rasterization_conflicts_max_iterations: int = 100
    use_louvain_communities: bool = True  # enable Louvain community detection
    community_resolution: float = 1.0  # Louvain algorithm resolution parameter
    
    # Hardware cost model configuration
    cost_weights: Dict[str, float] = field(default_factory=lambda: {
        'tiers': 1.0,
        'length': 1.0, 
        'bumps': 1.0,
        'tsvs': 1.0
    })
    
    cost_baselines: Dict[str, float] = field(default_factory=lambda: {
        'tiers': 1,
        'length': 1.0,
        'bumps': 0.0,
        'tsvs': 0.0
    })
    
    cost_optimistic: Dict[str, float] = field(default_factory=lambda: {
        'tiers': 5,
        'length': 10.0,
        'bumps': 4.0,
        'tsvs': 3.0
    })
    
    # Core algorithm configuration parameters
    random_seed: int = 42
    debug_mode: bool = False
    
    # Pathfinding parameters
    astar_heuristic_weight: float = 1.0
    layer_change_cost: float = 2.0
    bump_bond_cost: float = 1.5
    
    # Auxiliary qubit grid sizing parameters
    hardware_efficiency_target: float = 0.5  # Target 50% efficiency like paper's 10×6 grid for 30 qubits
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.max_tiers < 1:
            raise ValueError("max_tiers must be >= 1")
        if self.max_bump_transitions < 0:
            raise ValueError("max_bump_transitions must be >= 0")
        if self.max_tsvs_per_edge < 0:
            raise ValueError("max_tsvs_per_edge must be >= 0")
        if self.grid_size[0] < 1 or self.grid_size[1] < 1:
            raise ValueError("grid_size dimensions must be >= 1")
        if not all(w >= 0 for w in self.cost_weights.values()):
            raise ValueError("cost_weights must be non-negative")
        if sum(self.cost_weights.values()) == 0:
            raise ValueError("at least one cost_weight must be positive")