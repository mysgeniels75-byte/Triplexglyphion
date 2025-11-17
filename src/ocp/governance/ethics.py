"""
Ethical Constraints and Consensus Protocols

Ensures the system operates within ethical and legal boundaries
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum


class EthicalDecision(Enum):
    """Outcome of ethical evaluation"""
    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_REVIEW = "requires_review"


@dataclass
class EthicalConstraints:
    """Hard constraints that cannot be violated"""

    # Market integrity
    no_market_manipulation: bool = True
    no_insider_trading: bool = True
    no_front_running: bool = True

    # Regulatory compliance
    respect_trading_limits: bool = True
    respect_position_limits: bool = True
    comply_with_reporting: bool = True

    # Social responsibility
    protect_retail_investors: bool = True
    avoid_systemic_risk: bool = True
    transparent_operations: bool = True

    # Risk management
    max_leverage: float = 2.0
    max_position_concentration: float = 0.25
    min_liquidity_buffer: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "market_integrity": {
                "no_market_manipulation": self.no_market_manipulation,
                "no_insider_trading": self.no_insider_trading,
                "no_front_running": self.no_front_running,
            },
            "regulatory_compliance": {
                "respect_trading_limits": self.respect_trading_limits,
                "respect_position_limits": self.respect_position_limits,
                "comply_with_reporting": self.comply_with_reporting,
            },
            "social_responsibility": {
                "protect_retail_investors": self.protect_retail_investors,
                "avoid_systemic_risk": self.avoid_systemic_risk,
                "transparent_operations": self.transparent_operations,
            },
            "risk_management": {
                "max_leverage": self.max_leverage,
                "max_position_concentration": self.max_position_concentration,
                "min_liquidity_buffer": self.min_liquidity_buffer,
            },
        }


class EthicsEvaluator:
    """
    Evaluates proposed actions against ethical constraints

    Uses distributed consensus for gray-area decisions
    """

    def __init__(self, constraints: Optional[EthicalConstraints] = None):
        self.constraints = constraints or EthicalConstraints()

        # Voting history
        self.voting_history: List[Dict] = []

    def evaluate_action(
        self,
        action: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple[EthicalDecision, str]:
        """
        Evaluate if an action is ethically acceptable

        Returns:
            (decision, reason)
        """

        # Check hard constraints
        if action.get("manipulative", False):
            if self.constraints.no_market_manipulation:
                return EthicalDecision.REJECTED, "Action flagged as market manipulation"

        if action.get("uses_insider_info", False):
            if self.constraints.no_insider_trading:
                return EthicalDecision.REJECTED, "Action uses insider information"

        # Check leverage
        leverage = action.get("leverage", 1.0)
        if leverage > self.constraints.max_leverage:
            return EthicalDecision.REJECTED, f"Leverage {leverage} exceeds limit {self.constraints.max_leverage}"

        # Check position concentration
        concentration = action.get("position_concentration", 0.0)
        if concentration > self.constraints.max_position_concentration:
            return EthicalDecision.REJECTED, f"Position concentration {concentration} exceeds limit"

        # Check if action affects retail investors
        if action.get("impacts_retail", False):
            if self.constraints.protect_retail_investors:
                # Requires review
                return EthicalDecision.REQUIRES_REVIEW, "Action may impact retail investors"

        # Check systemic risk
        systemic_risk = action.get("systemic_risk_score", 0.0)
        if systemic_risk > 0.7:
            if self.constraints.avoid_systemic_risk:
                return EthicalDecision.REJECTED, f"Systemic risk score {systemic_risk} too high"

        # Action passes all checks
        return EthicalDecision.APPROVED, "Action approved"

    def distributed_vote(
        self,
        action: Dict[str, Any],
        agent_votes: List[Dict[str, Any]],
    ) -> tuple[EthicalDecision, str]:
        """
        Conduct distributed consensus vote on ethically ambiguous action

        Each agent votes based on its own ethical reasoning

        Args:
            action: Proposed action
            agent_votes: List of {agent_id, vote, reasoning}

        Returns:
            (consensus_decision, explanation)
        """

        if not agent_votes:
            return EthicalDecision.REQUIRES_REVIEW, "No votes received"

        # Tally votes
        approve_votes = sum(
            1 for vote in agent_votes
            if vote.get("vote") == EthicalDecision.APPROVED.value
        )
        reject_votes = sum(
            1 for vote in agent_votes
            if vote.get("vote") == EthicalDecision.REJECTED.value
        )

        total_votes = len(agent_votes)

        # Require 2/3 majority to approve
        if approve_votes >= (2 * total_votes) / 3:
            decision = EthicalDecision.APPROVED
            reason = f"Consensus approval: {approve_votes}/{total_votes} votes"
        elif reject_votes >= (2 * total_votes) / 3:
            decision = EthicalDecision.REJECTED
            reason = f"Consensus rejection: {reject_votes}/{total_votes} votes"
        else:
            decision = EthicalDecision.REQUIRES_REVIEW
            reason = f"No consensus: {approve_votes} approve, {reject_votes} reject"

        # Record vote
        self.voting_history.append({
            "action": action,
            "votes": agent_votes,
            "decision": decision.value,
            "reason": reason,
        })

        return decision, reason
