"""Communication influence estimates from the observation history only."""

from __future__ import annotations

import re
from typing import Dict, Iterable, Mapping, Sequence

from observer.communication_observer import CommunicationObserver


class CommunicationInfluenceEstimator:
    """Estimate propagation and downstream-decision influence at time T."""

    def __init__(self, weights: Sequence[float] = (0.5, 0.5)) -> None:
        if len(weights) != 2 or any(weight < 0 for weight in weights) or sum(weights) == 0:
            raise ValueError("influence weights must contain two non-negative values with positive sum")
        total = float(sum(weights))
        self.weights = tuple(weight / total for weight in weights)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z]{3,}", text.lower()))

    @staticmethod
    def _reachable_count(source: str, graph: Mapping[str, Iterable[str]]) -> int:
        visited = {source}
        frontier = [source]
        while frontier:
            node = frontier.pop()
            for neighbor in graph.get(node, ()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append(neighbor)
        return len(visited) - 1

    @classmethod
    def _reaches_any(cls, source: str, graph: Mapping[str, Iterable[str]], terminals: set[str]) -> bool:
        if not terminals:
            return False
        visited = {source}
        frontier = [source]
        while frontier:
            node = frontier.pop()
            for neighbor in graph.get(node, ()):
                if neighbor in terminals:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append(neighbor)
        return False

    def estimate(
        self,
        observer: CommunicationObserver,
        agents: Iterable[str],
        terminal_agents: Iterable[str] = (),
    ) -> Dict[str, Dict[str, float]]:
        """Return online estimates available at the current observation time.

        ``final_flow_estimate`` is deliberately prospective.  It is never
        presented as evidence that a message changed a completed final answer.
        """
        graph = observer.estimated_graph()
        all_agents = sorted(set(agents) | observer.observed_agents)
        observed_terminals = set(terminal_agents) & observer.observed_agents
        count = max(1, len(all_agents) - 1)
        sent_by = {agent: [event for event in observer.events if event.sender == agent] for agent in all_agents}
        all_messages = list(observer.events)
        result = {}
        for agent in all_agents:
            sent = sent_by[agent]
            direct_range = len(set(graph.get(agent, ()))) / count
            references = 0
            comparisons = 0
            for event in sent:
                source_tokens = self._tokens(event.message)
                if not source_tokens:
                    continue
                for later in all_messages[event.sequence + 1 :]:
                    if later.sender == agent:
                        continue
                    overlap = len(source_tokens & self._tokens(later.message)) / len(source_tokens)
                    comparisons += 1
                    if overlap >= 0.20:
                        references += 1
            reference_score = references / comparisons if comparisons else 0.0
            propagation = 0.5 * direct_range + 0.5 * reference_score
            downstream_range = self._reachable_count(agent, graph) / count
            final_flow_estimate = 1.0 if self._reaches_any(agent, graph, observed_terminals) else downstream_range
            w_prop, w_final = self.weights
            result[agent] = {
                "propagation": propagation,
                "final_flow_estimate": final_flow_estimate,
                "score": w_prop * propagation + w_final * final_flow_estimate,
            }
        return result

    def posthoc_final_influence(
        self,
        observer: CommunicationObserver,
        agents: Iterable[str],
        terminal_agents: Iterable[str],
    ) -> Dict[str, Dict[str, float]]:
        """Measure whether observed messages actually entered a terminal input.

        This is an evaluation-only quantity calculated after the episode.  A
        direct terminal delivery is an observed fact; an indirect value records
        a routed path to a terminal, not an unsupported causal claim about the
        terminal LLM's internal reasoning.
        """
        graph = observer.estimated_graph()
        all_agents = sorted(set(agents) | observer.observed_agents)
        terminals = set(terminal_agents) & observer.observed_agents
        direct = {
            agent: any(event.sender == agent and event.receiver in terminals for event in observer.events)
            for agent in all_agents
        }
        result = {}
        for agent in all_agents:
            reaches_terminal = self._reaches_any(agent, graph, terminals)
            # Binary provenance: the observed output either has a routed path
            # into a terminal input or it does not.  Keep directness separate
            # rather than assigning an arbitrary partial-credit constant.
            final_entry = float(reaches_terminal)
            result[agent] = {
                "direct_terminal_delivery": float(direct[agent]),
                "indirect_terminal_path": float(reaches_terminal and not direct[agent]),
                "final_entry": final_entry,
            }
        return result
