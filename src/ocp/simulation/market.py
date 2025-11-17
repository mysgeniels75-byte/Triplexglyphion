"""
Market Simulation Environment

Provides realistic market dynamics for testing the OCP system
"""

import asyncio
from enum import Enum
from typing import Dict, List, Optional
import numpy as np
from numpy.typing import NDArray
from datetime import datetime, timedelta


class MarketRegime(Enum):
    """Different market regimes for testing"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"


class MarketSimulator:
    """
    Simulates market data streams with realistic characteristics

    Can simulate different market regimes to test agent adaptability
    """

    def __init__(
        self,
        dimensions: int = 128,
        base_volatility: float = 0.02,
        drift: float = 0.0001,
    ):
        self.dimensions = dimensions
        self.base_volatility = base_volatility
        self.drift = drift

        # Current state
        self.current_state = np.zeros(dimensions)
        self.time_step = 0

        # Regime
        self.current_regime = MarketRegime.MEAN_REVERTING

        # Historical data
        self.history: List[NDArray[np.float64]] = []

    def set_regime(self, regime: MarketRegime) -> None:
        """Change market regime"""
        self.current_regime = regime

        # Adjust parameters based on regime
        if regime == MarketRegime.TRENDING_UP:
            self.drift = 0.001
            self.base_volatility = 0.015
        elif regime == MarketRegime.TRENDING_DOWN:
            self.drift = -0.001
            self.base_volatility = 0.015
        elif regime == MarketRegime.MEAN_REVERTING:
            self.drift = 0.0
            self.base_volatility = 0.02
        elif regime == MarketRegime.HIGH_VOLATILITY:
            self.drift = 0.0
            self.base_volatility = 0.05
        elif regime == MarketRegime.LOW_VOLATILITY:
            self.drift = 0.0001
            self.base_volatility = 0.005
        elif regime == MarketRegime.CRISIS:
            self.drift = -0.003
            self.base_volatility = 0.1

    def generate_step(self) -> NDArray[np.float64]:
        """
        Generate one time step of market data

        Uses geometric Brownian motion with mean reversion
        """

        # Mean reversion term
        mean_reversion_strength = 0.1
        mean_reversion = -mean_reversion_strength * self.current_state

        # Drift term
        drift_term = self.drift

        # Volatility (stochastic)
        volatility = self.base_volatility * (1.0 + 0.5 * np.random.randn())
        noise = np.random.randn(self.dimensions)

        # Update state
        self.current_state += (
            mean_reversion +
            drift_term +
            volatility * noise
        )

        # Add to history
        self.history.append(self.current_state.copy())

        self.time_step += 1

        return self.current_state.copy()

    async def stream(
        self,
        output_queue: asyncio.Queue,
        rate_hz: float = 10.0,
        duration: Optional[float] = None,
    ) -> None:
        """
        Stream market data to a queue

        Args:
            output_queue: Queue to push data to
            rate_hz: Data rate in Hz
            duration: Optional duration in seconds
        """

        dt = 1.0 / rate_hz
        start_time = asyncio.get_event_loop().time()

        while True:
            # Check duration
            if duration is not None:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= duration:
                    break

            # Generate data
            data = self.generate_step()

            # Push to queue
            await output_queue.put(data)

            # Wait
            await asyncio.sleep(dt)

    def inject_event(
        self,
        event_type: str,
        magnitude: float = 1.0,
    ) -> None:
        """
        Inject a market event (e.g., news shock)

        Args:
            event_type: Type of event
            magnitude: Size of the shock
        """

        if event_type == "positive_shock":
            self.current_state += magnitude * np.abs(np.random.randn(self.dimensions))
        elif event_type == "negative_shock":
            self.current_state -= magnitude * np.abs(np.random.randn(self.dimensions))
        elif event_type == "volatility_spike":
            self.base_volatility *= (1.0 + magnitude)
        elif event_type == "regime_change":
            # Random regime change
            regimes = list(MarketRegime)
            new_regime = np.random.choice(regimes)
            self.set_regime(new_regime)

    def get_statistics(self) -> Dict:
        """Get current market statistics"""

        if len(self.history) < 2:
            return {}

        recent_data = np.array(self.history[-100:])

        return {
            "current_level": float(np.mean(self.current_state)),
            "volatility": float(np.std(recent_data)),
            "regime": self.current_regime.value,
            "time_step": self.time_step,
            "history_length": len(self.history),
        }
