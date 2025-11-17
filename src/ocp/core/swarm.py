"""
Swarm Coordination System - Distributed Temporal Coherence

Implements:
- Swarm coordination without centralized control
- Self-organized criticality maintenance
- Pheromone-based communication
- Emergent collective behavior
"""

import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from ocp.core.agent import Agent, AgentRole, PheromoneTrail


class SwarmCoordinator:
    """
    Coordinator for managing swarm dynamics without centralized control

    Maintains:
    - Coordination graph (who coordinates with whom)
    - Pheromone repository (shared information trails)
    - Criticality metrics (ensuring edge-of-chaos operation)
    - Temporal synchronization signals
    """

    def __init__(
        self,
        target_order_parameter: Tuple[float, float] = (0.3, 0.7),
        criticality_threshold: float = 2.0,
    ):
        self.agents: Dict[str, Agent] = {}
        self.coordination_graph = nx.Graph()

        # Pheromone repository
        self.pheromone_trails: Dict[str, PheromoneTrail] = {}

        # Criticality parameters
        self.target_order_parameter = target_order_parameter
        self.criticality_threshold = criticality_threshold
        self.current_order_parameter: float = 0.5

        # Performance tracking
        self.swarm_metrics: Dict[str, float] = {
            "coherence": 0.0,
            "diversity": 0.0,
            "total_capital": 0.0,
            "total_return": 0.0,
        }

    def add_agent(self, agent: Agent) -> None:
        """Add an agent to the swarm"""
        self.agents[agent.agent_id] = agent
        self.coordination_graph.add_node(agent.agent_id, role=agent.role)

        # Connect to neighbors based on role compatibility
        self._update_coordination_graph(agent)

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the swarm"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            self.coordination_graph.remove_node(agent_id)

    def _update_coordination_graph(self, new_agent: Agent) -> None:
        """
        Update coordination graph with new connections

        Agents form connections based on:
        - Role complementarity
        - Spatial proximity in strategy space
        - Historical collaboration success
        """
        for agent_id, agent in self.agents.items():
            if agent_id == new_agent.agent_id:
                continue

            # Connect if roles are complementary
            if self._are_roles_complementary(new_agent.role, agent.role):
                self.coordination_graph.add_edge(new_agent.agent_id, agent_id)
                new_agent.neighbors.add(agent_id)
                agent.neighbors.add(new_agent.agent_id)

    def _are_roles_complementary(
        self,
        role1: AgentRole,
        role2: AgentRole,
    ) -> bool:
        """Check if two roles are complementary for coordination"""
        complementary_pairs = {
            (AgentRole.PATTERN_DETECTOR, AgentRole.SENTIMENT_ANALYZER),
            (AgentRole.PATTERN_DETECTOR, AgentRole.RISK_ASSESSOR),
            (AgentRole.EXPLORER, AgentRole.EXPLOITER),
            (AgentRole.COORDINATOR, AgentRole.PATTERN_DETECTOR),
            (AgentRole.COORDINATOR, AgentRole.SENTIMENT_ANALYZER),
            (AgentRole.COORDINATOR, AgentRole.RISK_ASSESSOR),
        }

        return (role1, role2) in complementary_pairs or (role2, role1) in complementary_pairs

    async def step(self, dt: float, market_data: NDArray[np.float64]) -> None:
        """
        Execute one coordination step for the entire swarm

        This is where the magic happens:
        - Agents perceive environment
        - Agents make local decisions
        - Coordination signals propagate
        - Swarm coherence is maintained
        """
        # 1. Broadcast market data to all agents for perception
        perception_tasks = [
            agent.perceive_environment(market_data)
            for agent in self.agents.values()
        ]
        perceived_states = await asyncio.gather(*perception_tasks)

        # 2. Compute neighbor coupling forces
        coupling_forces = self._compute_coupling_forces()

        # 3. Update agent states in parallel
        for agent, perceived, coupling in zip(
            self.agents.values(),
            perceived_states,
            coupling_forces.values(),
        ):
            agent.update_state(dt, perceived, coupling)

        # 4. Collect coordination signals from neighbors
        neighbor_signals = self._collect_neighbor_signals()

        # 5. Agents make decisions
        decision_tasks = [
            agent.make_decision(perceived, neighbor_signals.get(agent.agent_id, []))
            for agent, perceived in zip(self.agents.values(), perceived_states)
        ]
        actions = await asyncio.gather(*decision_tasks)

        # 6. Process pheromone trails
        self._update_pheromones()

        # 7. Maintain criticality
        await self._maintain_criticality()

        # 8. Update swarm metrics
        self._update_swarm_metrics()

    def _compute_coupling_forces(self) -> Dict[str, NDArray[np.float64]]:
        """
        Compute coupling forces between connected agents

        Like starling murmurations: each agent responds to local neighbors
        """
        coupling_forces = {}

        for agent_id, agent in self.agents.items():
            force = np.zeros_like(agent.state.position)
            neighbor_count = 0

            for neighbor_id in agent.neighbors:
                if neighbor_id not in self.agents:
                    continue

                neighbor = self.agents[neighbor_id]

                # Alignment: match neighbor velocity
                alignment = neighbor.state.velocity - agent.state.velocity

                # Cohesion: move toward neighbor center of mass
                cohesion = neighbor.state.position - agent.state.position

                # Separation: maintain minimum distance
                distance = np.linalg.norm(cohesion)
                if distance < 0.1:  # Too close
                    separation = -cohesion / (distance + 1e-6)
                else:
                    separation = np.zeros_like(cohesion)

                # Combine forces with weights
                force += 0.4 * alignment + 0.3 * cohesion + 0.3 * separation
                neighbor_count += 1

            if neighbor_count > 0:
                force /= neighbor_count

            coupling_forces[agent_id] = force

        return coupling_forces

    def _collect_neighbor_signals(self) -> Dict[str, List[Dict]]:
        """Collect coordination signals from neighbors for each agent"""
        signals = defaultdict(list)

        for agent_id, agent in self.agents.items():
            for neighbor_id in agent.neighbors:
                if neighbor_id not in self.agents:
                    continue

                neighbor = self.agents[neighbor_id]

                # Package neighbor state as signal
                signal = {
                    "agent_id": neighbor_id,
                    "role": neighbor.role.value,
                    "position": neighbor.state.position.tolist(),
                    "beliefs": neighbor.state.beliefs,
                    "phase": neighbor.phase,
                }

                signals[agent_id].append(signal)

        return signals

    def _update_pheromones(self) -> None:
        """Update pheromone trail repository and handle decay"""
        current_time = datetime.utcnow()

        # Collect new pheromones from agents
        for agent in self.agents.values():
            for trail_id, trail in agent.pheromone_trails.items():
                self.pheromone_trails[trail_id] = trail

        # Remove expired pheromones
        expired = [
            trail_id
            for trail_id, trail in self.pheromone_trails.items()
            if trail.current_intensity(current_time) < 0.01
        ]

        for trail_id in expired:
            del self.pheromone_trails[trail_id]

    async def _maintain_criticality(self) -> None:
        """
        Maintain swarm at edge of chaos through criticality gradient descent

        If swarm is too ordered -> inject randomness
        If swarm is too chaotic -> strengthen coupling
        """
        # Calculate current order parameter
        self.current_order_parameter = self._calculate_order_parameter()

        min_target, max_target = self.target_order_parameter

        if self.current_order_parameter > max_target:
            # Too ordered - inject exploratory behavior
            await self._inject_exploration()
        elif self.current_order_parameter < min_target:
            # Too chaotic - strengthen coordination
            await self._strengthen_coordination()

    def _calculate_order_parameter(self) -> float:
        """
        Calculate order parameter Ψ = |1/N Σ exp(iθ)|

        Measures coherence of agent phases
        """
        if not self.agents:
            return 0.0

        phase_sum = sum(agent.get_phase_vector() for agent in self.agents.values())
        order_param = abs(phase_sum) / len(self.agents)

        return order_param

    async def _inject_exploration(self) -> None:
        """Inject controlled randomness to prevent over-coherence"""
        # Select random subset of agents to perturb
        num_to_perturb = max(1, len(self.agents) // 10)
        agents_to_perturb = np.random.choice(
            list(self.agents.keys()),
            size=num_to_perturb,
            replace=False,
        )

        for agent_id in agents_to_perturb:
            agent = self.agents[agent_id]
            # Add random perturbation to velocity
            perturbation = np.random.randn(*agent.state.velocity.shape) * 0.5
            agent.state.velocity += perturbation

    async def _strengthen_coordination(self) -> None:
        """Strengthen coupling to reduce chaos"""
        # Increase neighbor connectivity
        for agent_id, agent in self.agents.items():
            if len(agent.neighbors) < 3:
                # Add more neighbors
                potential_neighbors = set(self.agents.keys()) - {agent_id} - agent.neighbors

                if potential_neighbors:
                    new_neighbor = np.random.choice(list(potential_neighbors))
                    agent.neighbors.add(new_neighbor)
                    self.agents[new_neighbor].neighbors.add(agent_id)
                    self.coordination_graph.add_edge(agent_id, new_neighbor)

    def _update_swarm_metrics(self) -> None:
        """Update aggregate swarm performance metrics"""
        if not self.agents:
            return

        # Coherence (order parameter)
        self.swarm_metrics["coherence"] = self.current_order_parameter

        # Diversity (variance in agent positions)
        positions = np.array([agent.state.position for agent in self.agents.values()])
        self.swarm_metrics["diversity"] = float(np.std(positions))

        # Total capital
        self.swarm_metrics["total_capital"] = sum(
            agent.state.capital for agent in self.agents.values()
        )

        # Total return
        total_return = sum(
            agent.state.performance.get("total_return", 0.0)
            for agent in self.agents.values()
        )
        self.swarm_metrics["total_return"] = total_return

    def get_metrics(self) -> Dict[str, float]:
        """Get current swarm metrics"""
        return self.swarm_metrics.copy()

    def get_topology_stats(self) -> Dict[str, float]:
        """Get coordination graph topology statistics"""
        if not self.coordination_graph.nodes():
            return {}

        return {
            "num_agents": self.coordination_graph.number_of_nodes(),
            "num_connections": self.coordination_graph.number_of_edges(),
            "avg_degree": np.mean([d for _, d in self.coordination_graph.degree()]),
            "clustering_coefficient": nx.average_clustering(self.coordination_graph),
            "is_connected": nx.is_connected(self.coordination_graph),
        }


class Swarm:
    """
    High-level swarm interface for creating and managing agent collectives

    This is the main entry point for users of the OCP system
    """

    def __init__(
        self,
        num_pattern_detectors: int = 5,
        num_sentiment_analyzers: int = 3,
        num_risk_assessors: int = 2,
        initial_capital_per_agent: float = 100000.0,
    ):
        self.coordinator = SwarmCoordinator()
        self.agents: List[Agent] = []

        # Create agents with different roles
        self._create_agents(
            AgentRole.PATTERN_DETECTOR,
            num_pattern_detectors,
            initial_capital_per_agent,
        )
        self._create_agents(
            AgentRole.SENTIMENT_ANALYZER,
            num_sentiment_analyzers,
            initial_capital_per_agent,
        )
        self._create_agents(
            AgentRole.RISK_ASSESSOR,
            num_risk_assessors,
            initial_capital_per_agent,
        )

        # Add one coordinator agent
        self._create_agents(AgentRole.COORDINATOR, 1, initial_capital_per_agent)

    def _create_agents(
        self,
        role: AgentRole,
        count: int,
        initial_capital: float,
    ) -> None:
        """Create agents with specified role"""
        for _ in range(count):
            agent = Agent(role=role, initial_capital=initial_capital)
            self.agents.append(agent)
            self.coordinator.add_agent(agent)

    async def run(
        self,
        market_data_stream: asyncio.Queue,
        duration: Optional[float] = None,
    ) -> None:
        """
        Run the swarm on a market data stream

        Args:
            market_data_stream: Async queue of market data
            duration: Optional duration in seconds (None = run forever)
        """
        start_time = asyncio.get_event_loop().time()
        dt = 0.1  # Time step in seconds

        while True:
            # Check duration
            if duration is not None:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= duration:
                    break

            try:
                # Get market data (with timeout)
                market_data = await asyncio.wait_for(
                    market_data_stream.get(),
                    timeout=1.0,
                )

                # Execute swarm coordination step
                await self.coordinator.step(dt, market_data)

            except asyncio.TimeoutError:
                # No data available, continue
                continue
            except Exception as e:
                print(f"Error in swarm execution: {e}")
                break

    def get_state(self) -> Dict:
        """Get complete swarm state snapshot"""
        return {
            "agents": [agent.state.to_dict() for agent in self.agents],
            "metrics": self.coordinator.get_metrics(),
            "topology": self.coordinator.get_topology_stats(),
            "num_pheromones": len(self.coordinator.pheromone_trails),
        }
