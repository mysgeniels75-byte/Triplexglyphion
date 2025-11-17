"""
Pheromone Management System

Implements information trails that agents use to coordinate without
centralized communication. Like ants leaving chemical trails to food sources.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray


class PheromoneType(Enum):
    """Types of pheromone trails agents can leave"""

    # Information discovery pheromones
    PATTERN_DETECTED = "pattern_detected"
    OPPORTUNITY_FOUND = "opportunity_found"
    RISK_IDENTIFIED = "risk_identified"

    # Coordination pheromones
    POSITION_TAKEN = "position_taken"
    POSITION_CLOSED = "position_closed"
    STRATEGY_CHANGE = "strategy_change"

    # Performance pheromones
    SUCCESS = "success"
    FAILURE = "failure"
    LEARNING = "learning"

    # Exploration pheromones
    TERRITORY_EXPLORED = "territory_explored"
    DEAD_END = "dead_end"


class PheromoneManager:
    """
    Manages the global pheromone trail repository

    Responsibilities:
    - Store and retrieve pheromone trails
    - Handle decay over time
    - Compute pheromone gradients for agent navigation
    - Detect pheromone patterns (trail convergence, etc.)
    """

    def __init__(
        self,
        default_decay_rate: float = 0.1,
        max_trail_age: float = 3600.0,  # 1 hour
    ):
        self.trails: Dict[str, 'PheromoneTrailData'] = {}
        self.default_decay_rate = default_decay_rate
        self.max_trail_age = max_trail_age

        # Statistics
        self.stats = {
            "total_trails_created": 0,
            "total_trails_expired": 0,
            "active_trails_by_type": {},
        }

    def add_trail(
        self,
        trail_id: str,
        agent_id: str,
        trail_type: PheromoneType,
        position: NDArray[np.float64],
        intensity: float,
        information: Dict[str, Any],
        decay_rate: Optional[float] = None,
    ) -> None:
        """Add a new pheromone trail"""

        trail = PheromoneTrailData(
            trail_id=trail_id,
            agent_id=agent_id,
            trail_type=trail_type,
            position=position,
            initial_intensity=intensity,
            current_intensity=intensity,
            information=information,
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow(),
            decay_rate=decay_rate or self.default_decay_rate,
        )

        self.trails[trail_id] = trail
        self.stats["total_trails_created"] += 1

    def get_trails_in_region(
        self,
        center: NDArray[np.float64],
        radius: float,
        trail_type: Optional[PheromoneType] = None,
    ) -> List['PheromoneTrailData']:
        """Get all trails within a spatial region"""

        trails_in_region = []

        for trail in self.trails.values():
            # Calculate distance
            distance = np.linalg.norm(trail.position - center)

            if distance <= radius:
                # Check type filter
                if trail_type is None or trail.trail_type == trail_type:
                    trails_in_region.append(trail)

        return trails_in_region

    def compute_pheromone_gradient(
        self,
        position: NDArray[np.float64],
        trail_type: Optional[PheromoneType] = None,
        sensing_radius: float = 1.0,
    ) -> NDArray[np.float64]:
        """
        Compute pheromone gradient at a given position

        The gradient points in the direction of increasing pheromone concentration,
        guiding agents toward areas of interest.

        Returns:
            Gradient vector in state space
        """

        # Get trails in sensing radius
        nearby_trails = self.get_trails_in_region(
            position,
            sensing_radius,
            trail_type,
        )

        if not nearby_trails:
            return np.zeros_like(position)

        # Compute gradient as weighted sum of direction vectors
        gradient = np.zeros_like(position)

        for trail in nearby_trails:
            # Direction from current position to trail
            direction = trail.position - position
            distance = np.linalg.norm(direction)

            if distance < 1e-6:
                continue

            # Normalize direction
            direction = direction / distance

            # Weight by intensity and inverse distance
            weight = trail.current_intensity / (distance + 0.1)

            gradient += weight * direction

        return gradient

    def update(self, dt: float) -> None:
        """
        Update all pheromone trails (decay over time)

        Args:
            dt: Time elapsed in seconds
        """

        current_time = datetime.utcnow()
        expired_trails = []

        for trail_id, trail in self.trails.items():
            # Calculate age
            age = (current_time - trail.created_at).total_seconds()

            # Check if expired
            if age > self.max_trail_age:
                expired_trails.append(trail_id)
                continue

            # Apply exponential decay: I(t) = I_0 * exp(-λt)
            trail.current_intensity *= np.exp(-trail.decay_rate * dt)

            # Mark as expired if intensity too low
            if trail.current_intensity < 0.01:
                expired_trails.append(trail_id)

            trail.last_updated = current_time

        # Remove expired trails
        for trail_id in expired_trails:
            del self.trails[trail_id]
            self.stats["total_trails_expired"] += 1

        # Update statistics
        self._update_statistics()

    def _update_statistics(self) -> None:
        """Update trail statistics"""

        self.stats["active_trails_by_type"] = {}

        for trail in self.trails.values():
            trail_type = trail.trail_type.value
            self.stats["active_trails_by_type"][trail_type] = (
                self.stats["active_trails_by_type"].get(trail_type, 0) + 1
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get pheromone system statistics"""
        return {
            **self.stats,
            "active_trails": len(self.trails),
        }

    def detect_convergence_points(
        self,
        trail_type: Optional[PheromoneType] = None,
        min_trails: int = 3,
        max_radius: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Detect points where multiple pheromone trails converge

        These are areas of high collective interest - likely opportunities
        """

        convergence_points = []

        trails = [
            t for t in self.trails.values()
            if trail_type is None or t.trail_type == trail_type
        ]

        if len(trails) < min_trails:
            return []

        # Cluster trails by position
        positions = np.array([t.position for t in trails])

        # Simple clustering: find points with many neighbors
        for i, pos in enumerate(positions):
            neighbors = []

            for j, other_pos in enumerate(positions):
                if i == j:
                    continue

                distance = np.linalg.norm(pos - other_pos)
                if distance <= max_radius:
                    neighbors.append(j)

            if len(neighbors) >= min_trails - 1:
                # Found convergence point
                neighbor_trails = [trails[j] for j in neighbors] + [trails[i]]

                total_intensity = sum(t.current_intensity for t in neighbor_trails)

                convergence_points.append({
                    "center": pos.tolist(),
                    "num_trails": len(neighbor_trails),
                    "total_intensity": total_intensity,
                    "trail_ids": [t.trail_id for t in neighbor_trails],
                })

        return convergence_points


class PheromoneTrailData:
    """Data structure for a single pheromone trail"""

    def __init__(
        self,
        trail_id: str,
        agent_id: str,
        trail_type: PheromoneType,
        position: NDArray[np.float64],
        initial_intensity: float,
        current_intensity: float,
        information: Dict[str, Any],
        created_at: datetime,
        last_updated: datetime,
        decay_rate: float,
    ):
        self.trail_id = trail_id
        self.agent_id = agent_id
        self.trail_type = trail_type
        self.position = position
        self.initial_intensity = initial_intensity
        self.current_intensity = current_intensity
        self.information = information
        self.created_at = created_at
        self.last_updated = last_updated
        self.decay_rate = decay_rate

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "trail_id": self.trail_id,
            "agent_id": self.agent_id,
            "trail_type": self.trail_type.value,
            "position": self.position.tolist(),
            "initial_intensity": self.initial_intensity,
            "current_intensity": self.current_intensity,
            "information": self.information,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "decay_rate": self.decay_rate,
            "age_seconds": (datetime.utcnow() - self.created_at).total_seconds(),
        }
