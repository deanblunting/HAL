"""
Configuration classes for HAL algorithm.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Callable


@dataclass
class HALConfig:
    """Comprehensive HAL algorithm configuration with optimization parameters."""

    # Complete user-configurable parameters from Appendix A.3
    # Custom positions: explicit map for node placement override
    custom_positions: Optional[Dict[int, Tuple[int, int]]] = None

    # Edge margin: safety margin in grid cells around every routed trace
    edge_margin: int = 1

    # Node size: radius added around each node before routing starts
    node_size: int = 1

    # Qubit spacing: units of space between qubits for routing infrastructure
    qubit_spacing: int = 10

    # Grid size: overall device area, aspect ratio, and layout canvas granularity
    grid_size: Tuple[int, int] = (100, 100)
    device_aspect_ratio: float = 1.0
    layout_granularity: int = 1

    # Maximum bump transitions per coupler: limits bump bond usage per connection
    # Each edge can switch back and forth between layer 0 and 1 up to 4 times total
    max_bump_transitions: int = 4

    # Maximum TSVs per coupler: restricts through-silicon vias per connection
    max_tsvs_per_edge: int = 3

    # Maximum coupler length: limits connection length between qubits/nodes
    max_coupler_length: Optional[float] = None

    # Multi-tier routing configuration
    max_tiers: int = 10

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


    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.max_tiers < 1:
            raise ValueError("max_tiers must be >= 1")
        if self.max_bump_transitions < 0:
            raise ValueError("max_bump_transitions must be >= 0")
        if self.max_tsvs_per_edge < 0:
            raise ValueError("max_tsvs_per_edge must be >= 0")
        if self.edge_margin < 0:
            raise ValueError("edge_margin must be >= 0")
        if self.node_size < 0:
            raise ValueError("node_size must be >= 0")
        if self.grid_size[0] < 1 or self.grid_size[1] < 1:
            raise ValueError("grid_size dimensions must be >= 1")
        if self.device_aspect_ratio <= 0:
            raise ValueError("device_aspect_ratio must be > 0")
        if self.layout_granularity < 1:
            raise ValueError("layout_granularity must be >= 1")
        if self.max_coupler_length is not None and self.max_coupler_length <= 0:
            raise ValueError("max_coupler_length must be > 0")
        if not all(w >= 0 for w in self.cost_weights.values()):
            raise ValueError("cost_weights must be non-negative")
        if sum(self.cost_weights.values()) == 0:
            raise ValueError("at least one cost_weight must be positive")
