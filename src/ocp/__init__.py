"""
Omniscient Convergence Protocol (OCP)

A multi-agent AI system for coordinated intelligence and temporal wealth synthesis
through distributed autonomous agents with emergent swarm behavior.
"""

__version__ = "0.1.0"
__author__ = "OCP Research Team"

from ocp.core.agent import Agent, AgentState
from ocp.core.swarm import Swarm, SwarmCoordinator
from ocp.coordination.coherence import CoherenceMetrics, calculate_order_parameter
from ocp.temporal.synthesis import TemporalSynthesisEngine, FutureStateManifold

__all__ = [
    "Agent",
    "AgentState",
    "Swarm",
    "SwarmCoordinator",
    "CoherenceMetrics",
    "calculate_order_parameter",
    "TemporalSynthesisEngine",
    "FutureStateManifold",
]
