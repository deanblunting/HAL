#!/usr/bin/env python3
"""
Test HAL cost function directly with Table III data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hal.cost import HardwareCostCalculator
from hal.config import HALConfig

def test_directional_codes():
    """Test cost function with directional codes metrics from Table III."""
    
    config = HALConfig()
    calculator = HardwareCostCalculator(config)
    
    # All directional codes have same metrics: tiers=1, length=1.00, bumps=0.00, tsvs=0.00
    metrics = {
        "tiers": 1,
        "length": 1.00,
        "bumps": 0.00,
        "tsvs": 0.00
    }
    
    cost = calculator.calculate_cost(metrics)
    
    print(f"Directional codes metrics: {metrics}")
    print(f"Expected cost: 1.00")
    print(f"Our calculated cost: {cost}")
    print(f"Match: {abs(cost - 1.00) < 1e-10}")

if __name__ == "__main__":
    test_directional_codes()