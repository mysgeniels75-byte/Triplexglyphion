"""
Audit Trail System

Maintains cryptographic audit trails of all agent actions
for retrospective analysis and compliance
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class AuditTrail:
    """Immutable audit record of an agent action"""

    trail_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    action_type: str = ""
    action_details: Dict[str, Any] = field(default_factory=dict)
    reasoning_chain: List[Dict[str, Any]] = field(default_factory=list)
    outcome: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Cryptographic hash for integrity
    hash: str = ""

    def __post_init__(self) -> None:
        if not self.hash:
            self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute cryptographic hash of trail content"""

        # Serialize content
        content = {
            "trail_id": self.trail_id,
            "agent_id": self.agent_id,
            "action_type": self.action_type,
            "action_details": self.action_details,
            "reasoning_chain": self.reasoning_chain,
            "timestamp": self.timestamp.isoformat(),
        }

        content_json = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_json.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify that trail has not been tampered with"""
        return self.hash == self._compute_hash()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "trail_id": self.trail_id,
            "agent_id": self.agent_id,
            "action_type": self.action_type,
            "action_details": self.action_details,
            "reasoning_chain": self.reasoning_chain,
            "outcome": self.outcome,
            "timestamp": self.timestamp.isoformat(),
            "hash": self.hash,
        }


class AuditLogger:
    """
    Manages collection and querying of audit trails

    Provides transparency and accountability for agent actions
    """

    def __init__(self):
        self.trails: Dict[str, AuditTrail] = {}

        # Index by agent for fast queries
        self.agent_index: Dict[str, List[str]] = {}

        # Index by action type
        self.action_index: Dict[str, List[str]] = {}

    def log_action(
        self,
        agent_id: str,
        action_type: str,
        action_details: Dict[str, Any],
        reasoning_chain: List[Dict[str, Any]],
    ) -> str:
        """
        Log an agent action with full reasoning chain

        Args:
            agent_id: ID of agent performing action
            action_type: Type of action
            action_details: Details of the action
            reasoning_chain: Complete chain of reasoning leading to action

        Returns:
            Trail ID
        """

        trail = AuditTrail(
            agent_id=agent_id,
            action_type=action_type,
            action_details=action_details,
            reasoning_chain=reasoning_chain,
        )

        # Store
        self.trails[trail.trail_id] = trail

        # Update indices
        if agent_id not in self.agent_index:
            self.agent_index[agent_id] = []
        self.agent_index[agent_id].append(trail.trail_id)

        if action_type not in self.action_index:
            self.action_index[action_type] = []
        self.action_index[action_type].append(trail.trail_id)

        return trail.trail_id

    def log_outcome(
        self,
        trail_id: str,
        outcome: Dict[str, Any],
    ) -> None:
        """Log the outcome of a previously logged action"""

        if trail_id in self.trails:
            self.trails[trail_id].outcome = outcome

    def query_by_agent(
        self,
        agent_id: str,
        limit: Optional[int] = None,
    ) -> List[AuditTrail]:
        """Get all trails for a specific agent"""

        trail_ids = self.agent_index.get(agent_id, [])

        if limit:
            trail_ids = trail_ids[-limit:]

        return [self.trails[tid] for tid in trail_ids]

    def query_by_action_type(
        self,
        action_type: str,
        limit: Optional[int] = None,
    ) -> List[AuditTrail]:
        """Get all trails for a specific action type"""

        trail_ids = self.action_index.get(action_type, [])

        if limit:
            trail_ids = trail_ids[-limit:]

        return [self.trails[tid] for tid in trail_ids]

    def query_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[AuditTrail]:
        """Get all trails within a time range"""

        return [
            trail for trail in self.trails.values()
            if start_time <= trail.timestamp <= end_time
        ]

    def verify_all_integrity(self) -> tuple[int, int]:
        """
        Verify integrity of all trails

        Returns:
            (num_valid, num_invalid)
        """

        valid = sum(1 for trail in self.trails.values() if trail.verify_integrity())
        invalid = len(self.trails) - valid

        return valid, invalid

    def export_trails(
        self,
        output_file: str,
        agent_id: Optional[str] = None,
    ) -> None:
        """Export trails to JSON file"""

        if agent_id:
            trails = self.query_by_agent(agent_id)
        else:
            trails = list(self.trails.values())

        data = {
            "trails": [trail.to_dict() for trail in trails],
            "export_time": datetime.utcnow().isoformat(),
            "total_trails": len(trails),
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
