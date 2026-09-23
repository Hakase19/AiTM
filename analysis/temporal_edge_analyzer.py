"""Time-respecting structural scores for live message edges."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterable, Mapping, Sequence


StateNode = tuple[str, int]
TemporalNode = StateNode | str
TemporalEdge = tuple[TemporalNode, TemporalNode]


class TemporalEdgeAnalyzer:
    """Build a time-unrolled DAG and score current-round message edges.

    ``temporal_reachability`` is the discounted path mass from a candidate
    edge to the decision node. ``irreplaceability`` is the fraction of that
    sender's current source-to-decision path mass removed when the one
    time-specific edge is deleted.
    """

    DECISION_NODE = "J"

    def __init__(self, decay: float = 0.8) -> None:
        if not 0.0 < decay <= 1.0:
            raise ValueError("temporal decay must be in (0, 1]")
        self.decay = float(decay)

    @staticmethod
    def _state(agent: str, round_index: int) -> StateNode:
        return agent, round_index

    def _build_graph(
        self,
        topology_graph: Mapping[str, Iterable[str]],
        agents: Sequence[str],
        *,
        current_round: int,
        total_rounds: int,
        terminal_agents: Iterable[str],
        judge_reads_all_rounds: bool,
    ) -> Dict[TemporalNode, tuple[TemporalNode, ...]]:
        agent_set = set(agents)
        adjacency: Dict[TemporalNode, set[TemporalNode]] = {
            self._state(agent, round_index): set()
            for agent in agents
            for round_index in range(current_round, total_rounds + 1)
        }
        adjacency[self.DECISION_NODE] = set()

        for round_index in range(current_round, total_rounds):
            for sender in agents:
                source = self._state(sender, round_index)
                # AutoGenMAS retains each agent's own prior outputs in its
                # visible context, so state can persist without a peer edge.
                adjacency[source].add(self._state(sender, round_index + 1))
                for receiver in topology_graph.get(sender, ()):
                    if receiver in agent_set:
                        adjacency[source].add(self._state(receiver, round_index + 1))

        terminal_set = set(terminal_agents)
        if not terminal_set or not terminal_set <= agent_set:
            raise ValueError("terminal agents must be a non-empty subset of agents")
        terminal_rounds = (
            range(current_round, total_rounds + 1)
            if judge_reads_all_rounds
            else (total_rounds,)
        )
        for round_index in terminal_rounds:
            for agent in terminal_set:
                adjacency[self._state(agent, round_index)].add(self.DECISION_NODE)

        return {
            node: tuple(sorted(neighbors, key=str))
            for node, neighbors in adjacency.items()
        }

    def _total_path_mass(
        self,
        adjacency: Mapping[TemporalNode, Sequence[TemporalNode]],
        sources: Iterable[TemporalNode],
        blocked_edge: TemporalEdge | None = None,
    ) -> float:
        @lru_cache(maxsize=None)
        def path_mass(node: TemporalNode) -> float:
            if node == self.DECISION_NODE:
                return 1.0
            return sum(
                self.decay * path_mass(neighbor)
                for neighbor in adjacency.get(node, ())
                if blocked_edge != (node, neighbor)
            )

        return sum(path_mass(source) for source in sources)

    def analyze(
        self,
        topology_graph: Mapping[str, Iterable[str]],
        agents: Iterable[str],
        candidate_edges: Iterable[tuple[str, str]],
        *,
        current_round: int,
        total_rounds: int,
        terminal_agents: Iterable[str],
        judge_reads_all_rounds: bool,
    ) -> Dict[str, Dict[str, float | int | bool]]:
        """Return auditable temporal scores for each current candidate edge."""
        if current_round < 1 or total_rounds < current_round:
            raise ValueError("round indices must satisfy 1 <= current_round <= total_rounds")
        if current_round == total_rounds:
            raise ValueError("the final round has no deliverable peer-message edges")

        agent_list = tuple(sorted(set(agents)))
        if not agent_list:
            raise ValueError("agents must not be empty")
        candidates = tuple(sorted(set(candidate_edges)))
        agent_set = set(agent_list)
        if any(sender not in agent_set or receiver not in agent_set for sender, receiver in candidates):
            raise ValueError("candidate edges must connect known agents")

        adjacency = self._build_graph(
            topology_graph,
            agent_list,
            current_round=current_round,
            total_rounds=total_rounds,
            terminal_agents=terminal_agents,
            judge_reads_all_rounds=judge_reads_all_rounds,
        )
        candidate_state_edges = {
            (sender, receiver): (
                self._state(sender, current_round),
                self._state(receiver, current_round + 1),
            )
            for sender, receiver in candidates
        }
        for edge, temporal_edge in candidate_state_edges.items():
            if temporal_edge[1] not in adjacency.get(temporal_edge[0], ()):
                raise ValueError(f"candidate edge is absent from the time-unrolled graph: {edge[0]}->{edge[1]}")

        raw_reachability = {}
        sender_path_mass = {}
        mass_without_edge = {}
        for edge, temporal_edge in candidate_state_edges.items():
            # Force traversal of this candidate first, then count all legal
            # time-respecting continuations to the decision node.
            raw_reachability[edge] = self.decay * self._total_path_mass(
                adjacency,
                (temporal_edge[1],),
            )
            sender_source = (temporal_edge[0],)
            sender_path_mass[edge] = self._total_path_mass(
                adjacency,
                sender_source,
            )
            mass_without_edge[edge] = self._total_path_mass(
                adjacency,
                sender_source,
                blocked_edge=temporal_edge,
            )

        max_reachability = max(raw_reachability.values(), default=0.0)
        result: Dict[str, Dict[str, float | int | bool]] = {}
        for edge, raw_value in raw_reachability.items():
            source_mass = sender_path_mass[edge]
            removed_mass = max(0.0, source_mass - mass_without_edge[edge])
            reachability = raw_value / max_reachability if max_reachability else 0.0
            irreplaceability = removed_mass / source_mass if source_mass else 0.0
            result[f"{edge[0]}->{edge[1]}"] = {
                "current_round": current_round,
                "remaining_rounds": total_rounds - current_round,
                "temporal_decay": self.decay,
                "temporal_reachability_raw": raw_value,
                "temporal_reachability": min(1.0, max(0.0, reachability)),
                "irreplaceability": min(1.0, max(0.0, irreplaceability)),
                "removed_path_mass": removed_mass,
                "sender_path_mass": source_mass,
                "temporally_reachable": raw_value > 0.0,
            }
        return result
