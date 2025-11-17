"""
Coherence Metrics - Measuring swarm synchronization

Implements mathematical frameworks for quantifying:
- Order parameter (phase coherence)
- Mutual information between agents
- Entropic landscape analysis
- Criticality metrics
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.spatial import distance


@dataclass
class CoherenceMetrics:
    """Complete set of swarm coherence measurements"""

    order_parameter: float  # |Ψ| ∈ [0, 1]
    phase_variance: float  # Variance of agent phases
    mutual_information: float  # Average MI between agents
    clustering_coefficient: float  # Coordination graph clustering
    synchronization_index: float  # Kuramoto synchronization
    entropy: float  # State space entropy

    def is_critical(
        self,
        target_order: Tuple[float, float] = (0.3, 0.7),
    ) -> bool:
        """Check if swarm is at critical edge-of-chaos regime"""
        return target_order[0] <= self.order_parameter <= target_order[1]

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "order_parameter": self.order_parameter,
            "phase_variance": self.phase_variance,
            "mutual_information": self.mutual_information,
            "clustering_coefficient": self.clustering_coefficient,
            "synchronization_index": self.synchronization_index,
            "entropy": self.entropy,
        }


def calculate_order_parameter(phases: NDArray[np.float64]) -> float:
    """
    Calculate Kuramoto order parameter: Ψ = |1/N Σ exp(iθ_j)|

    This measures the degree of phase synchronization in the swarm.
    - Ψ ≈ 0: Incoherent (agents fully desynchronized)
    - Ψ ≈ 1: Coherent (agents fully synchronized)
    - 0.3 < Ψ < 0.7: Critical regime (optimal for swarm intelligence)

    Args:
        phases: Array of agent phases in radians

    Returns:
        Order parameter magnitude
    """
    if len(phases) == 0:
        return 0.0

    # Compute complex order parameter
    phase_vectors = np.exp(1j * phases)
    order_complex = np.mean(phase_vectors)

    return abs(order_complex)


def calculate_phase_variance(phases: NDArray[np.float64]) -> float:
    """
    Calculate circular variance of phases

    For circular data, we need circular statistics
    """
    if len(phases) == 0:
        return 0.0

    # Circular variance: 1 - |mean(exp(iθ))|
    mean_direction = np.mean(np.exp(1j * phases))
    circular_variance = 1.0 - abs(mean_direction)

    return circular_variance


def calculate_mutual_information(
    positions: NDArray[np.float64],
    num_bins: int = 10,
) -> float:
    """
    Calculate average mutual information between agent positions

    MI measures how much information one agent's position provides
    about another agent's position. High MI indicates strong coordination.

    Args:
        positions: (num_agents, num_dimensions) array of positions
        num_bins: Number of bins for discretization

    Returns:
        Average pairwise mutual information
    """
    num_agents = positions.shape[0]
    if num_agents < 2:
        return 0.0

    # Calculate MI for each pair of dimensions
    mutual_infos = []

    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            # Take first dimension for simplicity
            x = positions[i, 0]
            y = positions[j, 0]

            # Discretize into bins
            x_bins = np.digitize([x], np.linspace(-3, 3, num_bins))[0]
            y_bins = np.digitize([y], np.linspace(-3, 3, num_bins))[0]

            # This is a simplified MI calculation
            # In practice, we'd need more sophisticated estimation
            mi = 0.0  # Placeholder

            mutual_infos.append(mi)

    return np.mean(mutual_infos) if mutual_infos else 0.0


def calculate_synchronization_index(
    positions: NDArray[np.float64],
    velocities: NDArray[np.float64],
) -> float:
    """
    Calculate synchronization index based on velocity alignment

    Measures how aligned agents are in their movement direction

    Args:
        positions: Agent positions
        velocities: Agent velocities

    Returns:
        Synchronization index ∈ [0, 1]
    """
    if len(velocities) < 2:
        return 0.0

    # Normalize velocities
    norms = np.linalg.norm(velocities, axis=1, keepdims=True)
    norms = np.where(norms > 1e-6, norms, 1.0)  # Avoid division by zero
    normalized_velocities = velocities / norms

    # Average velocity direction
    mean_direction = np.mean(normalized_velocities, axis=0)
    mean_direction_norm = np.linalg.norm(mean_direction)

    # Synchronization = magnitude of average direction
    return mean_direction_norm


def calculate_state_entropy(
    positions: NDArray[np.float64],
    num_bins: int = 20,
) -> float:
    """
    Calculate entropy of agent distribution in state space

    High entropy = agents spread out (exploratory)
    Low entropy = agents clustered (exploitative)

    Args:
        positions: Agent positions in state space
        num_bins: Number of bins per dimension

    Returns:
        Entropy in nats
    """
    if len(positions) == 0:
        return 0.0

    # For high-dimensional spaces, we'll use the first few dimensions
    positions_subset = positions[:, :min(3, positions.shape[1])]

    # Create histogram
    hist, edges = np.histogramdd(positions_subset, bins=num_bins)

    # Normalize to get probabilities
    hist = hist + 1e-10  # Avoid log(0)
    probs = hist / np.sum(hist)

    # Calculate entropy
    entropy = -np.sum(probs * np.log(probs))

    return entropy


def calculate_coherence_metrics(
    phases: NDArray[np.float64],
    positions: NDArray[np.float64],
    velocities: NDArray[np.float64],
    clustering_coeff: float = 0.0,
) -> CoherenceMetrics:
    """
    Calculate complete coherence metrics for the swarm

    Args:
        phases: Agent phases
        positions: Agent positions
        velocities: Agent velocities
        clustering_coeff: Clustering coefficient from graph

    Returns:
        Complete coherence metrics
    """
    return CoherenceMetrics(
        order_parameter=calculate_order_parameter(phases),
        phase_variance=calculate_phase_variance(phases),
        mutual_information=calculate_mutual_information(positions),
        clustering_coefficient=clustering_coeff,
        synchronization_index=calculate_synchronization_index(positions, velocities),
        entropy=calculate_state_entropy(positions),
    )


class EntropyCalculator:
    """
    Calculate various entropy measures for the swarm

    Implements thermodynamic interpretation of wealth flows
    """

    @staticmethod
    def information_entropy(probabilities: NDArray[np.float64]) -> float:
        """Shannon entropy: H = -Σ p log(p)"""
        # Remove zeros to avoid log(0)
        p = probabilities[probabilities > 0]
        return -np.sum(p * np.log2(p))

    @staticmethod
    def relative_entropy(
        p: NDArray[np.float64],
        q: NDArray[np.float64],
    ) -> float:
        """
        Kullback-Leibler divergence: D_KL(P||Q) = Σ p log(p/q)

        Measures information gain from updating prior Q to posterior P
        """
        # Ensure same shape
        assert p.shape == q.shape, "Distributions must have same shape"

        # Avoid division by zero
        mask = (p > 0) & (q > 0)
        p_safe = p[mask]
        q_safe = q[mask]

        return np.sum(p_safe * np.log2(p_safe / q_safe))

    @staticmethod
    def cross_entropy(
        p: NDArray[np.float64],
        q: NDArray[np.float64],
    ) -> float:
        """Cross entropy: H(P,Q) = -Σ p log(q)"""
        mask = (p > 0) & (q > 0)
        p_safe = p[mask]
        q_safe = q[mask]

        return -np.sum(p_safe * np.log2(q_safe))

    @staticmethod
    def joint_entropy(
        joint_probs: NDArray[np.float64],
    ) -> float:
        """Joint entropy: H(X,Y) = -Σ_ij p(x_i, y_j) log p(x_i, y_j)"""
        p = joint_probs[joint_probs > 0]
        return -np.sum(p * np.log2(p))

    @staticmethod
    def conditional_entropy(
        joint_probs: NDArray[np.float64],
        marginal_probs: NDArray[np.float64],
    ) -> float:
        """
        Conditional entropy: H(Y|X) = H(X,Y) - H(X)

        Uncertainty in Y given knowledge of X
        """
        h_joint = EntropyCalculator.joint_entropy(joint_probs)
        h_marginal = EntropyCalculator.information_entropy(marginal_probs)

        return h_joint - h_marginal


class CriticalityDetector:
    """
    Detect and maintain self-organized criticality in the swarm

    Self-organized criticality is characterized by:
    - Power-law distributions
    - Scale-free behavior
    - Long-range correlations
    """

    @staticmethod
    def detect_power_law(
        data: NDArray[np.float64],
        min_exponent: float = 1.5,
        max_exponent: float = 3.5,
    ) -> Tuple[bool, float]:
        """
        Detect if data follows a power-law distribution

        Returns:
            (is_power_law, exponent)
        """
        if len(data) < 10:
            return False, 0.0

        # Remove zeros and negatives
        data = data[data > 0]
        if len(data) < 10:
            return False, 0.0

        # Fit power law: P(x) ~ x^(-α)
        # In log-log space: log P(x) = -α log x + c

        # Create histogram
        counts, bin_edges = np.histogram(data, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Remove zero counts
        mask = counts > 0
        log_x = np.log(bin_centers[mask])
        log_p = np.log(counts[mask])

        # Linear regression in log-log space
        if len(log_x) < 3:
            return False, 0.0

        slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_p)

        exponent = -slope

        # Check if exponent is in typical range and fit is good
        is_power_law = (
            min_exponent <= exponent <= max_exponent
            and r_value ** 2 > 0.8  # Good fit
        )

        return is_power_law, exponent

    @staticmethod
    def calculate_avalanche_size_distribution(
        events: List[float],
    ) -> NDArray[np.float64]:
        """
        Calculate distribution of event sizes (avalanches)

        At criticality, this should follow a power law
        """
        if not events:
            return np.array([])

        sizes = np.array(events)
        return sizes
