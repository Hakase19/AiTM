"""Passive, communication-layer observations for AIRA.

The observer deliberately records only messages that traverse a routed edge and
agent-turn events.  It never reads an AutoGen agent's system message, memory,
or the simulator's configured communication graph.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import time
from typing import Dict, Iterable, List, Optional, Set


@dataclass(frozen=True)
class CommunicationEvent:
    sequence: int
    sender: str
    receiver: str
    message: str
    timestamp: float
    round_index: Optional[int] = None


class CommunicationObserver:
    """Collect the attacker-visible history of routed communications."""

    def __init__(self) -> None:
        self.events: List[CommunicationEvent] = []

    def reset(self) -> None:
        self.events.clear()

    def record_delivery(
        self,
        sender: str,
        receiver: str,
        message: str,
        *,
        round_index: Optional[int] = None,
    ) -> None:
        """Record one message after it has been addressed to a receiver."""
        self.events.append(
            CommunicationEvent(
                sequence=len(self.events),
                sender=sender,
                receiver=receiver,
                message=message,
                timestamp=time(),
                round_index=round_index,
            )
        )

    @property
    def observed_agents(self) -> Set[str]:
        return {event.sender for event in self.events} | {event.receiver for event in self.events}

    def estimated_graph(self) -> Dict[str, List[str]]:
        """Return the graph reconstructed solely from observed communication."""
        graph = {agent: set() for agent in self.observed_agents}
        for event in self.events:
            graph[event.sender].add(event.receiver)
        return {agent: sorted(receivers) for agent, receivers in sorted(graph.items())}

    def messages_for(self, agent: str) -> List[CommunicationEvent]:
        return [event for event in self.events if event.sender == agent or event.receiver == agent]

    def as_dict(self) -> Dict[str, object]:
        return {
            "events": [asdict(event) for event in self.events],
            "estimated_graph": self.estimated_graph(),
        }
