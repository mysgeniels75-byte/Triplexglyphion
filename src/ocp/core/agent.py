"""
Core Agent Architecture - Crystalline Intelligence with Adaptive Perception

Implements the fundamental agent structure with:
- Axiomatic Trust Matrix (crystalline core)
- Adaptive Perception Membranes (fluid sensory interface)
- State evolution dynamics
- Decision-making architecture
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray


class AgentRole(Enum):
    """Agent specialization roles"""
    PATTERN_DETECTOR = "pattern_detector"
    SENTIMENT_ANALYZER = "sentiment_analyzer"
    RISK_ASSESSOR = "risk_assessor"
    COORDINATOR = "coordinator"
    EXPLORER = "explorer"
    EXPLOITER = "exploiter"


@dataclass
class AxiomaticTrustMatrix:
    """
    The crystalline core of agent identity and values.
    These axioms are immutable and define the agent's fundamental behavior.
    """
    maximize_information_asymmetry: bool = True
    minimize_entropy: bool = True
    preserve_capital: bool = True
    maintain_swarm_coherence: bool = True
    respect_ethical_bounds: bool = True

    # Ethical constraints (hard boundaries)
    no_market_manipulation: bool = True
    regulatory_compliance: bool = True
    protect_retail_investors: bool = True

    # Risk parameters
    max_position_size: float = 0.1  # Maximum 10% of capital per position
    max_leverage: float = 2.0
    min_sharpe_ratio: float = 1.0

    def validate_action(self, action: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate if an action respects the axiomatic constraints"""
        # Check position size
        if action.get("position_size", 0) > self.max_position_size:
            return False, f"Position size exceeds maximum: {self.max_position_size}"

        # Check leverage
        if action.get("leverage", 1.0) > self.max_leverage:
            return False, f"Leverage exceeds maximum: {self.max_leverage}"

        # Check ethical constraints
        if action.get("manipulative", False) and self.no_market_manipulation:
            return False, "Action flagged as market manipulation"

        return True, "Action validated"


@dataclass
class AdaptivePerceptionMembrane:
    """
    Fluid sensory interface that adapts to market conditions.
    Performs high-dimensional pattern detection and feature extraction.
    """
    dimensions: int = 128
    sensitivity: float = 0.5
    adaptation_rate: float = 0.01

    # Perception weights (learned over time)
    weights: Optional[NDArray[np.float64]] = None

    def __post_init__(self) -> None:
        if self.weights is None:
            # Initialize with random weights
            self.weights = np.random.randn(self.dimensions) * 0.1

    def perceive(self, raw_data: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Transform raw market data into agent's native representational manifold

        This is the projection from high-dimensional market data to the agent's
        internal representation space.
        """
        # Ensure raw_data matches expected dimensions
        if raw_data.shape[0] != self.dimensions:
            # Pad or truncate as needed
            if raw_data.shape[0] < self.dimensions:
                raw_data = np.pad(raw_data, (0, self.dimensions - raw_data.shape[0]))
            else:
                raw_data = raw_data[:self.dimensions]

        # Apply perception transformation
        perceived = np.tanh(raw_data * self.weights * self.sensitivity)

        return perceived

    def adapt(self, error_signal: NDArray[np.float64]) -> None:
        """Adapt perception weights based on prediction error"""
        if error_signal.shape[0] == self.dimensions:
            self.weights += self.adaptation_rate * error_signal


@dataclass
class AgentState:
    """Complete state vector of an agent"""
    agent_id: str
    position: NDArray[np.float64]  # Position in strategy space
    velocity: NDArray[np.float64]  # Velocity in strategy space
    beliefs: Dict[str, float]  # Probabilistic beliefs about market state
    capital: float  # Available capital
    positions: Dict[str, float]  # Current market positions
    performance: Dict[str, float]  # Performance metrics
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary"""
        return {
            "agent_id": self.agent_id,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "beliefs": self.beliefs,
            "capital": self.capital,
            "positions": self.positions,
            "performance": self.performance,
            "timestamp": self.timestamp.isoformat(),
        }


class Agent:
    """
    Autonomous agent with crystalline core and adaptive perception.

    Each agent:
    - Maintains rigid internal values (Axiomatic Trust Matrix)
    - Adapts flexibly to external environment (Adaptive Perception Membrane)
    - Coordinates with other agents through pheromone trails
    - Evolves its decision-making schemas over time
    """

    def __init__(
        self,
        role: AgentRole,
        initial_capital: float = 100000.0,
        state_dimensions: int = 128,
        perception_dimensions: int = 128,
        agent_id: Optional[str] = None,
    ):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.role = role

        # Crystalline core (immutable values)
        self.axioms = AxiomaticTrustMatrix()

        # Adaptive perception
        self.perception = AdaptivePerceptionMembrane(dimensions=perception_dimensions)

        # Internal state
        self.state = AgentState(
            agent_id=self.agent_id,
            position=np.zeros(state_dimensions),
            velocity=np.zeros(state_dimensions),
            beliefs={},
            capital=initial_capital,
            positions={},
            performance={
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "win_rate": 0.0,
                "max_drawdown": 0.0,
            },
        )

        # Decision-making components
        self.schemas: List[DecisionSchema] = []
        self.active_schema: Optional[DecisionSchema] = None

        # Coordination state
        self.neighbors: Set[str] = set()
        self.pheromone_trails: Dict[str, PheromoneTrail] = {}

        # Internal clock for temporal coherence
        self.internal_time: float = 0.0
        self.phase: float = np.random.uniform(0, 2 * np.pi)  # Phase angle for coherence

    async def perceive_environment(
        self,
        market_data: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Process raw market data through adaptive perception membranes

        Returns the agent's internal representation of the environment
        """
        return self.perception.perceive(market_data)

    async def make_decision(
        self,
        perceived_state: NDArray[np.float64],
        neighbor_signals: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Make a decision based on perceived state and neighbor coordination

        Returns an action dictionary or None if no action should be taken
        """
        # Select or activate decision schema
        if self.active_schema is None and self.schemas:
            self.active_schema = self._select_schema(perceived_state)

        if self.active_schema is None:
            return None

        # Generate candidate action
        action = await self.active_schema.generate_action(
            perceived_state,
            self.state,
            neighbor_signals,
        )

        # Validate against axiomatic constraints
        is_valid, reason = self.axioms.validate_action(action)

        if not is_valid:
            # Log constraint violation
            print(f"Agent {self.agent_id}: Action rejected - {reason}")
            return None

        return action

    def update_state(
        self,
        dt: float,
        external_forces: Optional[NDArray[np.float64]] = None,
        neighbor_coupling: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """
        Update agent state according to stochastic differential equation:
        ds = f(s, E, N) dt + σ(s) dW

        Args:
            dt: Time step
            external_forces: Environmental influences
            neighbor_coupling: Coordination forces from neighbors
        """
        # Deterministic drift
        drift = self._compute_drift(external_forces, neighbor_coupling)

        # Stochastic term
        volatility = self._compute_volatility()
        noise = np.random.randn(len(self.state.position))

        # Update position and velocity
        self.state.velocity += drift * dt + volatility * noise * np.sqrt(dt)
        self.state.position += self.state.velocity * dt

        # Update phase for coherence
        self.phase = (self.phase + 2 * np.pi * dt) % (2 * np.pi)

        # Update internal clock
        self.internal_time += dt

        # Update timestamp
        self.state.timestamp = datetime.utcnow()

    def _compute_drift(
        self,
        external_forces: Optional[NDArray[np.float64]],
        neighbor_coupling: Optional[NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Compute deterministic drift term"""
        drift = np.zeros_like(self.state.position)

        # Self-optimization (gradient ascent on objective)
        drift -= 0.1 * self.state.position  # Regularization

        # Environmental response
        if external_forces is not None:
            drift += 0.5 * external_forces

        # Neighbor coupling
        if neighbor_coupling is not None:
            drift += 0.3 * neighbor_coupling

        return drift

    def _compute_volatility(self) -> float:
        """Compute exploration volatility based on confidence"""
        base_volatility = 0.1
        confidence = self.state.beliefs.get("confidence", 0.5)

        # Higher confidence -> lower exploration
        return base_volatility * (1.0 - confidence)

    def _select_schema(self, perceived_state: NDArray[np.float64]) -> Optional['DecisionSchema']:
        """Select the most appropriate decision schema for current state"""
        if not self.schemas:
            return None

        # Compute fitness of each schema
        fitnesses = [
            schema.compute_fitness(perceived_state, self.state)
            for schema in self.schemas
        ]

        # Softmax selection
        fitnesses_array = np.array(fitnesses)
        probs = np.exp(fitnesses_array) / np.sum(np.exp(fitnesses_array))

        selected_idx = np.random.choice(len(self.schemas), p=probs)
        return self.schemas[selected_idx]

    def emit_pheromone(
        self,
        trail_type: str,
        intensity: float,
        information: Dict[str, Any],
    ) -> 'PheromoneTrail':
        """
        Emit a pheromone trail to communicate with other agents

        Pheromones encode decision context, outcomes, and reasoning
        """
        trail = PheromoneTrail(
            agent_id=self.agent_id,
            trail_type=trail_type,
            intensity=intensity,
            information=information,
            position=self.state.position.copy(),
            timestamp=datetime.utcnow(),
        )

        self.pheromone_trails[trail.trail_id] = trail
        return trail

    def get_phase_vector(self) -> complex:
        """Get the agent's phase in complex form for coherence calculation"""
        return np.exp(1j * self.phase)


@dataclass
class PheromoneTrail:
    """Information trail left by agents for swarm coordination"""
    trail_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    trail_type: str = ""
    intensity: float = 1.0
    information: Dict[str, Any] = field(default_factory=dict)
    position: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    decay_rate: float = 0.1

    def current_intensity(self, current_time: datetime) -> float:
        """Calculate current intensity with exponential decay"""
        age = (current_time - self.timestamp).total_seconds()
        return self.intensity * np.exp(-self.decay_rate * age)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "trail_id": self.trail_id,
            "agent_id": self.agent_id,
            "trail_type": self.trail_type,
            "intensity": self.intensity,
            "information": self.information,
            "position": self.position.tolist(),
            "timestamp": self.timestamp.isoformat(),
        }


class DecisionSchema:
    """
    Abstract decision-making schema that can be evolved and recombined

    Schemas encapsulate strategies, heuristics, and reasoning patterns
    """

    def __init__(self, schema_id: Optional[str] = None):
        self.schema_id = schema_id or str(uuid.uuid4())
        self.fitness_history: List[float] = []
        self.usage_count: int = 0

    async def generate_action(
        self,
        perceived_state: NDArray[np.float64],
        agent_state: AgentState,
        neighbor_signals: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate an action based on the schema's logic"""
        raise NotImplementedError("Subclasses must implement generate_action")

    def compute_fitness(
        self,
        perceived_state: NDArray[np.float64],
        agent_state: AgentState,
    ) -> float:
        """Compute fitness/suitability of this schema for current state"""
        # Default: use historical average fitness
        if self.fitness_history:
            return np.mean(self.fitness_history)
        return 0.5

    def update_fitness(self, outcome: float) -> None:
        """Update fitness based on action outcome"""
        self.fitness_history.append(outcome)
        self.usage_count += 1

        # Keep only recent history
        if len(self.fitness_history) > 100:
            self.fitness_history = self.fitness_history[-100:]
