"""
Knowledge Graph Construction

Builds structured knowledge representations from detected patterns
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
import uuid

import networkx as nx


@dataclass
class KnowledgeNode:
    """Node in the knowledge graph"""
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class KnowledgeGraph:
    """Graph structure for organizing detected knowledge"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, KnowledgeNode] = {}

    def add_node(
        self,
        node_type: str,
        content: Dict[str, Any],
        confidence: float = 0.5,
    ) -> str:
        """Add a knowledge node"""

        node = KnowledgeNode(
            node_type=node_type,
            content=content,
            confidence=confidence,
        )

        self.nodes[node.node_id] = node
        self.graph.add_node(node.node_id, **node.content)

        return node.node_id

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship: str,
        weight: float = 1.0,
    ) -> None:
        """Add relationship between nodes"""

        self.graph.add_edge(
            source_id,
            target_id,
            relationship=relationship,
            weight=weight,
        )

    def query(self, node_type: Optional[str] = None) -> List[KnowledgeNode]:
        """Query nodes by type"""

        if node_type is None:
            return list(self.nodes.values())

        return [
            node for node in self.nodes.values()
            if node.node_type == node_type
        ]

    def get_connected(self, node_id: str, max_depth: int = 2) -> Set[str]:
        """Get all nodes connected to given node within max depth"""

        if node_id not in self.graph:
            return set()

        connected = set()
        current_level = {node_id}

        for _ in range(max_depth):
            next_level = set()
            for node in current_level:
                neighbors = set(self.graph.successors(node)) | set(self.graph.predecessors(node))
                next_level.update(neighbors)

            connected.update(next_level)
            current_level = next_level

        return connected
