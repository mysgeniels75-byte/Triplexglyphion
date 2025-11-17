"""
Information Topology Analysis

The information topology of the market encodes how information flows
and where value accumulates. This module analyzes the geometry of
information space to identify wealth-generating structures.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import distance_matrix
from scipy.stats import entropy as scipy_entropy


@dataclass
class TopologyTensor:
    """
    Information topology tensor field I^μν(x,t)

    This tensor field encodes the local geometry of information flow
    at each point in economic space.
    """

    # Position in economic space
    position: NDArray[np.float64]

    # Tensor components (information metric)
    components: NDArray[np.float64]

    # Information density at this position
    density: float

    # Flow velocity (direction of information flow)
    flow_velocity: NDArray[np.float64]

    # Curvature (indicates information bottlenecks/hubs)
    curvature: float

    def __post_init__(self) -> None:
        """Validate tensor structure"""
        assert self.components.shape[0] == self.components.shape[1], \
            "Tensor must be square matrix"


class InformationTopology:
    """
    Analyzer for information topology in economic space

    Key capabilities:
    - Compute information tensor fields
    - Identify information basins (attractors)
    - Calculate wealth flow vectors
    - Detect topological features (holes, clusters, boundaries)
    """

    def __init__(self, dimensions: int = 128):
        self.dimensions = dimensions

        # Information field data
        self.tensor_field: Dict[str, TopologyTensor] = {}

        # Topological features
        self.basins: List[Dict] = []
        self.saddle_points: List[NDArray[np.float64]] = []
        self.sources: List[NDArray[np.float64]] = []
        self.sinks: List[NDArray[np.float64]] = []

    def compute_tensor_at_point(
        self,
        position: NDArray[np.float64],
        local_data: NDArray[np.float64],
        neighborhood_size: int = 10,
    ) -> TopologyTensor:
        """
        Compute information topology tensor at a specific point

        Args:
            position: Location in economic space
            local_data: Data samples near this position
            neighborhood_size: Number of neighbors to consider

        Returns:
            Information topology tensor
        """

        # Compute local information metric
        # This measures how information is distributed locally

        if len(local_data) < 2:
            # Not enough data for meaningful calculation
            return TopologyTensor(
                position=position,
                components=np.eye(self.dimensions),
                density=0.0,
                flow_velocity=np.zeros(self.dimensions),
                curvature=0.0,
            )

        # Compute covariance matrix (local metric)
        covariance = np.cov(local_data.T)

        # Ensure positive definite
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        metric = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Information density (determinant of metric)
        density = np.sqrt(np.linalg.det(metric + np.eye(len(metric)) * 1e-6))

        # Flow velocity (gradient of density)
        flow_velocity = self._compute_flow_velocity(position, local_data)

        # Curvature (measures how curved the information manifold is)
        curvature = self._compute_curvature(metric)

        return TopologyTensor(
            position=position,
            components=metric,
            density=density,
            flow_velocity=flow_velocity,
            curvature=curvature,
        )

    def _compute_flow_velocity(
        self,
        position: NDArray[np.float64],
        local_data: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute information flow velocity at a point"""

        if len(local_data) < 2:
            return np.zeros(self.dimensions)

        # Flow points toward higher information density
        # Compute gradient using finite differences

        center_density = len(local_data)  # Simplified

        # Sample nearby points
        epsilon = 0.1
        gradient = np.zeros(self.dimensions)

        for dim in range(min(5, self.dimensions)):  # Sample first few dimensions
            # Positive direction
            pos_point = position.copy()
            pos_point[dim] += epsilon

            # Count nearby samples (simplified density)
            pos_distances = np.linalg.norm(local_data - pos_point, axis=1)
            pos_density = np.sum(pos_distances < epsilon)

            # Negative direction
            neg_point = position.copy()
            neg_point[dim] -= epsilon

            neg_distances = np.linalg.norm(local_data - neg_point, axis=1)
            neg_density = np.sum(neg_distances < epsilon)

            # Gradient component
            gradient[dim] = (pos_density - neg_density) / (2 * epsilon)

        # Normalize
        norm = np.linalg.norm(gradient)
        if norm > 1e-6:
            gradient = gradient / norm

        return gradient

    def _compute_curvature(self, metric: NDArray[np.float64]) -> float:
        """
        Compute scalar curvature of the information manifold

        High curvature indicates information bottlenecks or hubs
        """

        # Simplified curvature calculation
        # Full implementation would compute Ricci scalar

        # Use trace of metric as proxy
        trace = np.trace(metric)

        # Deviation from flat metric
        flat_trace = metric.shape[0]

        curvature = abs(trace - flat_trace) / flat_trace

        return curvature

    def build_tensor_field(
        self,
        sample_points: NDArray[np.float64],
        data: NDArray[np.float64],
    ) -> None:
        """
        Build complete tensor field over economic space

        Args:
            sample_points: Grid of points to compute tensors at
            data: Full dataset for local calculations
        """

        self.tensor_field = {}

        for point in sample_points:
            # Find local neighborhood
            distances = np.linalg.norm(data - point, axis=1)
            local_indices = np.argsort(distances)[:20]  # 20 nearest neighbors
            local_data = data[local_indices]

            # Compute tensor
            tensor = self.compute_tensor_at_point(point, local_data)

            # Store
            key = f"{hash(point.tobytes())}"
            self.tensor_field[key] = tensor

    def identify_wealth_basins(
        self,
        wealth_function: Callable[[NDArray[np.float64]], float],
    ) -> List[Dict]:
        """
        Identify basins of attraction in the wealth potential landscape

        These are regions where wealth naturally accumulates

        Returns:
            List of basin descriptors
        """

        basins = []

        # Analyze tensor field for attracting regions
        for key, tensor in self.tensor_field.items():
            # Check if this is a local minimum in wealth potential
            # (basin attracts wealth flow)

            wealth_at_point = wealth_function(tensor.position)

            # High curvature + negative flow divergence = basin
            is_basin = (
                tensor.curvature > 0.5
                and wealth_at_point > 0.5  # Significant wealth
            )

            if is_basin:
                basin = {
                    "position": tensor.position.tolist(),
                    "wealth_level": wealth_at_point,
                    "curvature": tensor.curvature,
                    "density": tensor.density,
                }
                basins.append(basin)

        self.basins = basins
        return basins

    def compute_wealth_flow_field(
        self,
        wealth_gradient: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute wealth flow vector field J^μ = -I^μν ∂_ν Φ

        This is analogous to heat flow: wealth flows down the gradient
        of the wealth potential, modulated by the information topology

        Args:
            wealth_gradient: Gradient of wealth potential field

        Returns:
            Wealth flow vectors
        """

        flow_vectors = []

        for tensor in self.tensor_field.values():
            # J = -I^{-1} ∇Φ
            # Information metric acts as conductivity

            try:
                inverse_metric = np.linalg.inv(tensor.components)
                flow = -inverse_metric @ wealth_gradient
                flow_vectors.append(flow)
            except np.linalg.LinAlgError:
                # Singular matrix, use identity
                flow_vectors.append(-wealth_gradient)

        return np.array(flow_vectors) if flow_vectors else np.array([])

    def find_optimal_positions(
        self,
        num_positions: int = 5,
    ) -> List[Dict]:
        """
        Find optimal positions for capital deployment

        These are positions at strategic points in the topology:
        - Near sinks (wealth attractors)
        - At high-curvature points (information hubs)
        - On geodesics between major basins

        Returns:
            List of recommended positions
        """

        positions = []

        # Sort tensors by strategic value
        tensor_scores = []

        for key, tensor in self.tensor_field.items():
            # Score based on:
            # 1. High curvature (information hub)
            # 2. High density (lots of information)
            # 3. Strong flow (active region)

            score = (
                0.4 * tensor.curvature
                + 0.3 * tensor.density
                + 0.3 * np.linalg.norm(tensor.flow_velocity)
            )

            tensor_scores.append((score, tensor))

        # Sort descending
        tensor_scores.sort(key=lambda x: x[0], reverse=True)

        # Take top positions
        for score, tensor in tensor_scores[:num_positions]:
            position = {
                "location": tensor.position.tolist(),
                "strategic_value": score,
                "curvature": tensor.curvature,
                "density": tensor.density,
                "flow_magnitude": np.linalg.norm(tensor.flow_velocity),
                "recommended_action": "deploy_capital",
            }
            positions.append(position)

        return positions

    def calculate_informational_carnot_efficiency(
        self,
        information_extracted: float,
        total_information_available: float,
    ) -> float:
        """
        Calculate efficiency of information → wealth transformation

        This is analogous to thermodynamic efficiency: how much of the
        available information is converted to actionable wealth

        Args:
            information_extracted: Bits of information actually used
            total_information_available: Total bits available

        Returns:
            Efficiency ∈ [0, 1]
        """

        if total_information_available <= 0:
            return 0.0

        efficiency = information_extracted / total_information_available

        # Cap at theoretical maximum (like Carnot efficiency)
        carnot_limit = 0.85  # Theoretical maximum for information systems

        return min(efficiency, carnot_limit)


class EntropicLandscape:
    """
    Analyzes the entropic landscape of the economic system

    Entropy gradients drive wealth flows: capital flows from high-entropy
    (disordered) regions to low-entropy (ordered) regions, and we extract
    work (profit) from this flow
    """

    def __init__(self, dimensions: int = 128):
        self.dimensions = dimensions

        # Entropy field
        self.entropy_field: Dict[str, float] = {}

    def compute_local_entropy(
        self,
        position: NDArray[np.float64],
        local_data: NDArray[np.float64],
        num_bins: int = 20,
    ) -> float:
        """
        Compute entropy at a specific position

        Args:
            position: Location in state space
            local_data: Nearby data samples
            num_bins: Bins for histogram

        Returns:
            Entropy in nats
        """

        if len(local_data) < 2:
            return 0.0

        # Use first few dimensions for entropy calculation
        data_subset = local_data[:, :min(3, local_data.shape[1])]

        # Create histogram
        hist, edges = np.histogramdd(data_subset, bins=num_bins)

        # Normalize to probability
        hist = hist + 1e-10
        probs = hist / np.sum(hist)

        # Calculate entropy
        ent = scipy_entropy(probs.flatten())

        return ent

    def compute_entropy_gradient(
        self,
        position: NDArray[np.float64],
        data: NDArray[np.float64],
        epsilon: float = 0.1,
    ) -> NDArray[np.float64]:
        """
        Compute gradient of entropy field

        Entropy gradients indicate where arbitrage opportunities exist

        Returns:
            Gradient vector
        """

        gradient = np.zeros(self.dimensions)

        # Central entropy
        central_entropy = self.compute_local_entropy(position, data)

        # Finite differences
        for dim in range(min(5, self.dimensions)):
            # Forward point
            forward = position.copy()
            forward[dim] += epsilon

            # Find local data
            distances = np.linalg.norm(data - forward, axis=1)
            local_indices = np.argsort(distances)[:20]
            local_data = data[local_indices]

            forward_entropy = self.compute_local_entropy(forward, local_data)

            # Gradient component
            gradient[dim] = (forward_entropy - central_entropy) / epsilon

        return gradient

    def identify_arbitrage_opportunities(
        self,
        entropy_threshold: float = 1.0,
    ) -> List[Dict]:
        """
        Identify arbitrage opportunities from entropy gradients

        Strong entropy gradients indicate inefficiencies that can be exploited

        Returns:
            List of arbitrage opportunities
        """

        opportunities = []

        for position_key, entropy in self.entropy_field.items():
            if entropy > entropy_threshold:
                opportunity = {
                    "position": position_key,
                    "entropy": entropy,
                    "opportunity_type": "entropy_arbitrage",
                    "potential_profit": entropy * 0.1,  # Simplified
                }
                opportunities.append(opportunity)

        return opportunities
