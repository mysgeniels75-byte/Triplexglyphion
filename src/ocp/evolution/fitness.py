"""
Fitness Evaluation for Schemas
"""

from typing import Dict, List
import numpy as np


class FitnessEvaluator:
    """Evaluates fitness of decision schemas"""

    @staticmethod
    def evaluate(
        schema_performance: Dict[str, float],
        weights: Dict[str, float] = None,
    ) -> float:
        """
        Evaluate schema fitness based on performance metrics

        Args:
            schema_performance: Dict of metric_name -> value
            weights: Optional weights for each metric

        Returns:
            Fitness score
        """

        if weights is None:
            weights = {
                "return": 0.4,
                "sharpe_ratio": 0.3,
                "win_rate": 0.2,
                "consistency": 0.1,
            }

        fitness = 0.0

        for metric, weight in weights.items():
            value = schema_performance.get(metric, 0.0)
            fitness += weight * value

        return fitness
