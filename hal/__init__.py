"""
HAL: Hardware-Aware Layout for Quantum Error Correcting Codes

A Python implementation of the HAL algorithm for placing and routing
quantum error correcting codes on multi-layer superconducting hardware.
"""

from hal import HAL
from config import HALConfig
from data_structures import QECCLayout, RoutingTier

__version__ = "0.1.0"
__all__ = ["HAL", "HALConfig", "QECCLayout", "RoutingTier"]