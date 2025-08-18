"""
HAL: Hardware-Aware Layout for Quantum Error Correcting Codes

Advanced Python implementation of the HAL algorithm for optimal placement
and routing of quantum error correcting codes on multi-tier superconducting
quantum processor architectures with comprehensive cost optimization.
"""

from .hal import HAL
from .config import HALConfig
from .data_structures import QECCLayout, RoutingTier

__version__ = "0.1.0"
__all__ = ["HAL", "HALConfig", "QECCLayout", "RoutingTier"]