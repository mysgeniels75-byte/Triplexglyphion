"""
Pattern Detection System

High-dimensional pattern detection that agents use to identify
opportunities invisible to human perception
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks, correlate
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN


class PatternType(Enum):
    """Types of patterns the system can detect"""
    TREND = "trend"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    CORRELATION = "correlation"
    ANOMALY = "anomaly"
    CYCLE = "cycle"
    INFORMATION_CASCADE = "information_cascade"


class PatternDetector:
    """Detects high-dimensional patterns in market data"""

    def __init__(self, dimensions: int = 128):
        self.dimensions = dimensions
        self.detected_patterns: List[Dict] = []

    def detect_all_patterns(
        self,
        data: NDArray[np.float64],
    ) -> List[Dict]:
        """Run all pattern detection algorithms"""

        patterns = []

        patterns.extend(self.detect_trends(data))
        patterns.extend(self.detect_anomalies(data))
        patterns.extend(self.detect_correlations(data))
        patterns.extend(self.detect_cycles(data))

        self.detected_patterns = patterns
        return patterns

    def detect_trends(
        self,
        data: NDArray[np.float64],
        window: int = 20,
    ) -> List[Dict]:
        """Detect trending patterns"""

        patterns = []

        if len(data) < window:
            return patterns

        # Compute rolling statistics
        for dim in range(min(5, data.shape[1] if data.ndim > 1 else 1)):
            series = data[:, dim] if data.ndim > 1 else data

            # Simple trend detection via linear regression
            for i in range(window, len(series)):
                window_data = series[i - window:i]
                x = np.arange(len(window_data))

                # Fit line
                coeffs = np.polyfit(x, window_data, 1)
                slope = coeffs[0]

                # Strong trend if slope significant
                if abs(slope) > np.std(window_data) * 0.1:
                    pattern = {
                        "type": PatternType.TREND.value,
                        "dimension": dim,
                        "position": i,
                        "strength": abs(slope),
                        "direction": "up" if slope > 0 else "down",
                        "confidence": 0.7,
                    }
                    patterns.append(pattern)

        return patterns

    def detect_anomalies(
        self,
        data: NDArray[np.float64],
        threshold: float = 3.0,
    ) -> List[Dict]:
        """Detect anomalous points (outliers)"""

        patterns = []

        # Z-score method
        if data.ndim == 1:
            z_scores = zscore(data)
            anomaly_indices = np.where(np.abs(z_scores) > threshold)[0]

            for idx in anomaly_indices:
                pattern = {
                    "type": PatternType.ANOMALY.value,
                    "position": int(idx),
                    "magnitude": float(abs(z_scores[idx])),
                    "confidence": 0.8,
                }
                patterns.append(pattern)

        return patterns

    def detect_correlations(
        self,
        data: NDArray[np.float64],
        min_correlation: float = 0.7,
    ) -> List[Dict]:
        """Detect correlated dimensions"""

        patterns = []

        if data.ndim < 2 or data.shape[1] < 2:
            return patterns

        # Compute correlation matrix
        corr_matrix = np.corrcoef(data.T)

        # Find strong correlations
        for i in range(corr_matrix.shape[0]):
            for j in range(i + 1, corr_matrix.shape[1]):
                if abs(corr_matrix[i, j]) >= min_correlation:
                    pattern = {
                        "type": PatternType.CORRELATION.value,
                        "dim1": i,
                        "dim2": j,
                        "correlation": float(corr_matrix[i, j]),
                        "confidence": 0.75,
                    }
                    patterns.append(pattern)

        return patterns

    def detect_cycles(
        self,
        data: NDArray[np.float64],
        min_period: int = 5,
        max_period: int = 50,
    ) -> List[Dict]:
        """Detect cyclical patterns"""

        patterns = []

        if data.ndim > 1:
            data = data[:, 0]  # Use first dimension

        # Autocorrelation
        autocorr = correlate(data, data, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]

        # Find peaks in autocorrelation
        peaks, properties = find_peaks(autocorr, distance=min_period)

        for peak in peaks:
            if min_period <= peak <= max_period:
                pattern = {
                    "type": PatternType.CYCLE.value,
                    "period": int(peak),
                    "strength": float(autocorr[peak]),
                    "confidence": 0.65,
                }
                patterns.append(pattern)

        return patterns
