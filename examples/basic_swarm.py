#!/usr/bin/env python3
"""
Basic Swarm Example

Demonstrates basic OCP swarm coordination with simulated market data
"""

import asyncio
import numpy as np

from ocp import Swarm
from ocp.simulation import MarketSimulator, MarketRegime


async def main():
    """Run basic swarm example"""

    print("=" * 70)
    print("Omniscient Convergence Protocol - Basic Swarm Example")
    print("=" * 70)
    print()

    # Create swarm with multiple agent types
    print("Initializing swarm...")
    swarm = Swarm(
        num_pattern_detectors=5,
        num_sentiment_analyzers=3,
        num_risk_assessors=2,
        initial_capital_per_agent=100000.0,
    )

    print(f"Created swarm with {len(swarm.agents)} agents")
    print()

    # Create market simulator
    print("Initializing market simulator...")
    market = MarketSimulator(
        dimensions=128,
        base_volatility=0.02,
    )

    # Set market regime
    market.set_regime(MarketRegime.TRENDING_UP)
    print(f"Market regime: {market.current_regime.value}")
    print()

    # Create data queue
    market_queue = asyncio.Queue()

    # Run simulation
    print("Starting simulation...")
    print("Running for 30 seconds with 10 Hz data rate...")
    print()

    # Start market data stream
    market_task = asyncio.create_task(
        market.stream(market_queue, rate_hz=10.0, duration=30.0)
    )

    # Start swarm
    swarm_task = asyncio.create_task(
        swarm.run(market_queue, duration=30.0)
    )

    # Monitor progress
    for i in range(6):
        await asyncio.sleep(5.0)

        # Get swarm state
        state = swarm.get_state()

        print(f"[{i*5}s] Swarm Metrics:")
        print(f"  Coherence: {state['metrics']['coherence']:.3f}")
        print(f"  Diversity: {state['metrics']['diversity']:.3f}")
        print(f"  Total Capital: ${state['metrics']['total_capital']:,.2f}")
        print(f"  Active Pheromones: {state['num_pheromones']}")
        print(f"  Topology Stats: {state['topology']}")
        print()

    # Wait for completion
    await market_task
    await swarm_task

    # Final statistics
    print()
    print("=" * 70)
    print("Simulation Complete")
    print("=" * 70)

    final_state = swarm.get_state()
    print()
    print("Final Swarm Metrics:")
    print(f"  Coherence: {final_state['metrics']['coherence']:.3f}")
    print(f"  Diversity: {final_state['metrics']['diversity']:.3f}")
    print(f"  Total Capital: ${final_state['metrics']['total_capital']:,.2f}")
    print(f"  Total Return: ${final_state['metrics']['total_return']:,.2f}")
    print()

    market_stats = market.get_statistics()
    print("Market Statistics:")
    for key, value in market_stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
