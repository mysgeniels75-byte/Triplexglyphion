"""
Synthetic Data Generator

Creates test data with known patterns for validation
"""

import numpy as np
from numpy.typing import NDArray


class SyntheticDataGenerator:
    """Generates synthetic market data with embedded patterns"""

    @staticmethod
    def generate_trend(
        length: int,
        slope: float = 0.01,
        noise_level: float = 0.1,
    ) -> NDArray[np.float64]:
        """Generate trending data"""

        t = np.arange(length)
        trend = slope * t
        noise = noise_level * np.random.randn(length)

        return trend + noise

    @staticmethod
    def generate_mean_reverting(
        length: int,
        mean: float = 0.0,
        reversion_speed: float = 0.1,
        volatility: float = 0.1,
    ) -> NDArray[np.float64]:
        """Generate mean-reverting series"""

        data = np.zeros(length)
        data[0] = mean

        for t in range(1, length):
            # Ornstein-Uhlenbeck process
            data[t] = (
                data[t-1]
                - reversion_speed * (data[t-1] - mean)
                + volatility * np.random.randn()
            )

        return data

    @staticmethod
    def generate_cyclic(
        length: int,
        period: int = 50,
        amplitude: float = 1.0,
        noise_level: float = 0.1,
    ) -> NDArray[np.float64]:
        """Generate cyclic data"""

        t = np.arange(length)
        cycle = amplitude * np.sin(2 * np.pi * t / period)
        noise = noise_level * np.random.randn(length)

        return cycle + noise

    @staticmethod
    def generate_jumps(
        length: int,
        jump_probability: float = 0.05,
        jump_size: float = 1.0,
    ) -> NDArray[np.float64]:
        """Generate data with random jumps"""

        data = np.cumsum(np.random.randn(length) * 0.1)

        # Add jumps
        jump_times = np.random.random(length) < jump_probability
        jumps = np.random.choice([-1, 1], size=length) * jump_size
        data += jump_times * jumps

        return data

    @staticmethod
    def generate_multidimensional(
        length: int,
        dimensions: int,
        correlations: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        """
        Generate correlated multidimensional data

        Args:
            length: Number of time steps
            dimensions: Number of dimensions
            correlations: Optional correlation matrix

        Returns:
            (length, dimensions) array
        """

        if correlations is None:
            # Default: random correlation matrix
            A = np.random.randn(dimensions, dimensions)
            correlations = A @ A.T

        # Cholesky decomposition
        L = np.linalg.cholesky(correlations)

        # Generate independent random data
        independent = np.random.randn(length, dimensions)

        # Correlate
        correlated = independent @ L.T

        return correlated
