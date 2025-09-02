#!/usr/bin/env python3
"""
Test HAL cost function directly with Table III data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hal.cost import HardwareCostCalculator
from hal.config import HALConfig

def test_cost_function_examples():
    """Test cost function with examples from BB, Tile, Radial, and Directional codes from Table III."""
    
    config = HALConfig()
    calculator = HardwareCostCalculator(config)
    
    # Examples from Table III - one from each code family
    test_cases = [
        # Directional code [36,4,4]
        ("Directional [36,4,4]", {"tiers": 1, "length": 1.00, "bumps": 0.00, "tsvs": 0.00}, 1.00),
        
        # BB code [24,4,4] 
        ("BB [24,4,4]", {"tiers": 3, "length": 3.65, "bumps": 2.50, "tsvs": 2.13}, 1.53),
        
        # Tile code [105,8,6]
        ("Tile [105,8,6]", {"tiers": 3, "length": 2.91, "bumps": 2.96, "tsvs": 2.14}, 1.54),
        
        # Radial code [16,2,4]
        ("Radial [16,2,4]", {"tiers": 2, "length": 2.76, "bumps": 1.78, "tsvs": 2.00}, 1.39)
    ]
    
    print("HAL Cost Function Validation - Multiple Code Families")
    print("=" * 60)
    print("Code Family      | Tiers | Length | Bumps | TSVs | Expected | Our Calc | Match")
    print("-" * 75)
    
    all_match = True
    
    for code_name, metrics, expected_cost in test_cases:
        cost = calculator.calculate_cost(metrics)
        matches = abs(cost - expected_cost) < 0.01
        
        if not matches:
            all_match = False
        
        match_status = "PASS" if matches else "FAIL"
        
        print(f"{code_name:16} |   {metrics['tiers']}   |  {metrics['length']:4.2f}  | {metrics['bumps']:5.2f} |{metrics['tsvs']:5.2f} |   {expected_cost:.2f}   |   {cost:.4f}   | {match_status}")
    
    print("-" * 75)
    
    if all_match:
        print("SUCCESS: All cost calculations match HAL paper results")
    else:
        print("FAILURE: Some cost calculations don't match paper results")
    
    return all_match

if __name__ == "__main__":
    success = test_cost_function_examples()
    sys.exit(0 if success else 1)