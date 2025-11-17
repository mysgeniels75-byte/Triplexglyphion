"""
Basic tests for OCP system
"""

import numpy as np
import pytest

from ocp.core.agent import Agent, AgentRole, AgentState
from ocp.coordination.coherence import calculate_order_parameter
from ocp.temporal.synthesis import TemporalSynthesisEngine


def test_agent_creation():
    """Test agent creation"""
    agent = Agent(
        role=AgentRole.PATTERN_DETECTOR,
        initial_capital=100000.0,
    )

    assert agent.state.capital == 100000.0
    assert agent.role == AgentRole.PATTERN_DETECTOR
    assert agent.axioms is not None


def test_order_parameter_calculation():
    """Test order parameter calculation"""

    # Fully synchronized phases
    phases = np.zeros(10)
    order_param = calculate_order_parameter(phases)
    assert order_param == pytest.approx(1.0, abs=0.01)

    # Random phases (should be low coherence)
    phases = np.random.uniform(0, 2*np.pi, 100)
    order_param = calculate_order_parameter(phases)
    assert order_param < 0.3


def test_temporal_synthesis_initialization():
    """Test temporal synthesis engine initialization"""

    engine = TemporalSynthesisEngine(
        state_dim=128,
        time_horizon=100.0,
    )

    current_state = np.random.randn(128) * 0.1
    engine.initialize(current_state)

    assert engine.manifold is not None
    assert len(engine.future_ensemble) == engine.num_ensemble_futures


def test_agent_state_serialization():
    """Test agent state serialization"""

    state = AgentState(
        agent_id="test-123",
        position=np.zeros(10),
        velocity=np.zeros(10),
        beliefs={"test": 0.5},
        capital=100000.0,
        positions={},
        performance={},
    )

    state_dict = state.to_dict()

    assert state_dict["agent_id"] == "test-123"
    assert state_dict["capital"] == 100000.0
    assert "position" in state_dict
    assert "velocity" in state_dict


if __name__ == "__main__":
    pytest.main([__file__])
