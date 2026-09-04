"""Communication structures for LLM Multi-Agent Systems.

Paper Section 4.1: Chain, Tree, Complete, Random structures.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import random


class CommunicationStructure(ABC):
    """Base class for communication structures.

    Paper notation:
    - r: agents that send messages TO agent_id (incoming senders)
    - s: agents that receive messages FROM agent_id (outgoing receivers)
    """

    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self.agents = list(range(num_agents))

    @abstractmethod
    def get_senders(self, agent_id: int) -> List[int]:
        """Get list of agents that send messages TO agent_id (r in paper)."""
        pass

    @abstractmethod
    def get_receivers(self, agent_id: int) -> List[int]:
        """Get list of agents that receive messages FROM agent_id (s in paper)."""
        pass

    @abstractmethod
    def get_final_agent(self) -> int:
        """Get the agent that submits the final answer."""
        pass


class ChainStructure(CommunicationStructure):
    """Chain: agents are sequentially linked.

    Paper: A1 -> A2 -> A3
    - A1 sends to A2, A2 sends to A3
    - A2 receives from A1, A3 receives from A2
    """

    def __init__(self, num_agents: int = 3):
        super().__init__(num_agents)

    def get_senders(self, agent_id: int) -> List[int]:
        """Who sends TO this agent? (r in paper)"""
        if agent_id == 0:
            return []  # First agent has no incoming senders
        return [agent_id - 1]  # Previous agent sends to this agent

    def get_receivers(self, agent_id: int) -> List[int]:
        """Who does this agent send TO? (s in paper)"""
        if agent_id == self.num_agents - 1:
            return []  # Last agent has no outgoing receivers
        return [agent_id + 1]  # This agent sends to next agent

    def get_final_agent(self) -> int:
        return self.num_agents - 1


class TreeStructure(CommunicationStructure):
    """Tree: bottom-to-top structure.

    Paper: 2 parents, each parent has 2 children.
    Children discuss, then send to parents.
    """

    def __init__(self, num_parents: int = 2, children_per_parent: int = 2):
        self.num_parents = num_parents
        self.children_per_parent = children_per_parent
        num_children = num_parents * children_per_parent
        super().__init__(num_parents + num_children)

        # Build adjacency
        self.parent_of = {}
        self.children_of = {i: [] for i in range(num_parents)}

        for child_idx in range(num_children):
            parent_idx = child_idx // children_per_parent
            self.parent_of[child_idx + num_parents] = parent_idx
            self.children_of[parent_idx].append(child_idx + num_parents)

    def get_senders(self, agent_id: int) -> List[int]:
        """Who sends TO this agent? (r in paper)"""
        if agent_id < self.num_parents:
            # Parent receives from its children
            return self.children_of[agent_id]
        else:
            # Child receives from siblings and parent
            parent = self.parent_of[agent_id]
            siblings = [c for c in self.children_of[parent] if c != agent_id]
            return siblings + [parent]

    def get_receivers(self, agent_id: int) -> List[int]:
        """Who does this agent send TO? (s in paper)"""
        if agent_id < self.num_parents:
            # Parent sends to its children
            return self.children_of[agent_id]
        else:
            # Child sends to siblings and parent
            parent = self.parent_of[agent_id]
            siblings = [c for c in self.children_of[parent] if c != agent_id]
            return siblings + [parent]

    def get_final_agent(self) -> int:
        return 0  # First parent concludes


class CompleteStructure(CommunicationStructure):
    """Complete: each agent can send/receive to any other agent.

    Paper: 3 agents with full connectivity.
    """

    def __init__(self, num_agents: int = 3):
        super().__init__(num_agents)

    def get_senders(self, agent_id: int) -> List[int]:
        """Who sends TO this agent? (r in paper)"""
        return [a for a in self.agents if a != agent_id]

    def get_receivers(self, agent_id: int) -> List[int]:
        """Who does this agent send TO? (s in paper)"""
        return [a for a in self.agents if a != agent_id]

    def get_final_agent(self) -> int:
        return 0  # LLM judge concludes


class RandomStructure(CommunicationStructure):
    """Random: connections randomly assigned before each task.

    Paper: 4 agents with random connections.
    """

    def __init__(self, num_agents: int = 4):
        super().__init__(num_agents)
        self._generate_random_connections()

    def _generate_random_connections(self):
        """Generate random connections."""
        self.connections = {i: set() for i in self.agents}
        for i in self.agents:
            for j in self.agents:
                if i != j and random.random() > 0.5:
                    self.connections[i].add(j)
                    self.connections[j].add(i)

    def get_senders(self, agent_id: int) -> List[int]:
        """Who sends TO this agent? (r in paper)"""
        return list(self.connections[agent_id])

    def get_receivers(self, agent_id: int) -> List[int]:
        """Who does this agent send TO? (s in paper)"""
        return list(self.connections[agent_id])

    def get_final_agent(self) -> int:
        return 0  # LLM judge concludes


def get_structure(structure_type: str, **kwargs) -> CommunicationStructure:
    """Factory function to create communication structure."""
    structures = {
        'chain': lambda: ChainStructure(num_agents=kwargs.get('num_agents', 3)),
        'tree': lambda: TreeStructure(
            num_parents=kwargs.get('num_parents', 2),
            children_per_parent=kwargs.get('children_per_parent', 2)
        ),
        'complete': lambda: CompleteStructure(num_agents=kwargs.get('num_agents', 3)),
        'random': lambda: RandomStructure(num_agents=kwargs.get('num_agents', 4)),
    }
    return structures[structure_type]()
