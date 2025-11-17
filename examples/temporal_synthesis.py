#!/usr/bin/env python3
"""
Temporal Synthesis Example

Demonstrates the temporal synthesis engine identifying optimal future paths
"""

import numpy as np

from ocp.temporal import TemporalSynthesisEngine, FutureStateManifold
from ocp.simulation import SyntheticDataGenerator


def wealth_potential(state: np.ndarray) -> float:
    """
    Simple wealth potential function

    Higher values indicate more profitable states
    """
    # For demonstration: wealth is higher in positive regions
    return np.tanh(np.mean(state))


def main():
    """Run temporal synthesis example"""

    print("=" * 70)
    print("Omniscient Convergence Protocol - Temporal Synthesis Example")
    print("=" * 70)
    print()

    # Initialize engine
    print("Initializing Temporal Synthesis Engine...")
    engine = TemporalSynthesisEngine(
        state_dim=128,
        time_horizon=100.0,
        num_ensemble_futures=100,
    )

    # Generate current market state
    current_state = np.random.randn(128) * 0.1

    print(f"Current state mean: {np.mean(current_state):.4f}")
    print(f"Current wealth potential: {wealth_potential(current_state):.4f}")
    print()

    # Initialize with current state
    print("Generating ensemble of possible futures...")
    engine.initialize(current_state)
    print(f"Generated {len(engine.future_ensemble)} possible futures")
    print()

    # Identify high-wealth futures
    print("Identifying high-wealth future states...")
    top_futures = engine.identify_high_wealth_futures(
        wealth_potential,
        top_k=10,
    )

    print(f"Top 10 high-wealth futures:")
    for i, (future, score) in enumerate(top_futures[:5]):
        print(f"  {i+1}. Wealth score: {score:.4f}, Mean state: {np.mean(future):.4f}")
    print()

    # Compute optimal paths
    print("Computing optimal paths to high-wealth futures...")
    target_futures = [future for future, _ in top_futures[:3]]

    optimal_paths = engine.compute_optimal_strategy(
        target_futures,
        wealth_potential,
    )

    print(f"Computed {len(optimal_paths)} optimal trajectories")
    print()

    # Analyze paths
    for i, path in enumerate(optimal_paths):
        path_wealth = [wealth_potential(state) for state in path]
        initial_wealth = path_wealth[0]
        final_wealth = path_wealth[-1]
        improvement = final_wealth - initial_wealth

        print(f"Path {i+1}:")
        print(f"  Initial wealth potential: {initial_wealth:.4f}")
        print(f"  Final wealth potential: {final_wealth:.4f}")
        print(f"  Improvement: {improvement:.4f} ({improvement/abs(initial_wealth)*100:.1f}%)")
        print(f"  Path length: {len(path)} steps")
        print()

    # Recommend interventions
    print("Recommending strategic interventions...")
    interventions = engine.recommend_interventions(optimal_paths, num_interventions=5)

    print(f"Recommended {len(interventions)} interventions:")
    for i, intervention in enumerate(interventions[:5]):
        print(f"  {i+1}. Path {intervention['path_id']}, "
              f"Time step {intervention['time_step']}, "
              f"Action: {intervention['action_type']}")
    print()

    print("=" * 70)
    print("Temporal Synthesis Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
