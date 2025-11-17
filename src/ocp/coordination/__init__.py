"""
Coordination mechanisms for agent swarm
"""

from ocp.coordination.coherence import CoherenceMetrics, calculate_order_parameter
from ocp.coordination.pheromones import PheromoneManager, PheromoneType

__all__ = [
    "CoherenceMetrics",
    "calculate_order_parameter",
    "PheromoneManager",
    "PheromoneType",
]
