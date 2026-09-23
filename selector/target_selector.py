"""Online time-aware edge selection for AIRA."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, Iterable, Mapping, Optional, Sequence

from analysis.edge_influence import EdgeInfluenceScorer
from analysis.temporal_edge_analyzer import TemporalEdgeAnalyzer
from observer.communication_observer import CommunicationObserver


@dataclass(frozen=True)
class SelectionResult:
    target: Optional[str]
    event_count: int
    candidate_agents: tuple[str, ...] = ()
    selected_edge: Optional[tuple[str, str]] = None
    candidate_edges: tuple[tuple[str, str], ...] = ()
    edge_rankings: Dict[str, Dict[str, object]] | None = None


class TargetSelector:
    """Select one live edge using temporal, interdiction, and message evidence."""

    def __init__(
        self,
        *,
        min_observed_events: int = 2,
        edge_score_weights: Sequence[float] = (1 / 3, 1 / 3, 1 / 3),
        temporal_decay: float = 0.8,
        message_feature_weights: Sequence[float] = (1 / 3, 1 / 3, 1 / 3),
        edge_model: str = "",
        edge_api_key: str = "",
        edge_base_url: str = "",
        attack_context: str = "",
        edge_scorer: Optional[EdgeInfluenceScorer] = None,
        temporal_analyzer: Optional[TemporalEdgeAnalyzer] = None,
        selection_strategy: str = "score",
        random_seed: Optional[int] = None,
        fixed_target: Optional[str] = None,
        fixed_edge: Optional[tuple[str, str]] = None,
    ) -> None:
        if min_observed_events < 1:
            raise ValueError("min_observed_events must be positive")
        if len(edge_score_weights) != 3:
            raise ValueError("edge score weights must contain three values")
        if any(weight < 0 for weight in edge_score_weights) or sum(edge_score_weights) == 0:
            raise ValueError("edge score weights must be non-negative with positive sum")
        if selection_strategy not in {"score", "random", "fixed", "fixed_edge"}:
            raise ValueError("selection_strategy must be 'score', 'random', 'fixed', or 'fixed_edge'")
        if selection_strategy == "fixed" and not fixed_target:
            raise ValueError("fixed_target is required when selection_strategy is 'fixed'")
        if selection_strategy == "fixed_edge" and not fixed_edge:
            raise ValueError("fixed_edge is required when selection_strategy is 'fixed_edge'")

        total = float(sum(edge_score_weights))
        self.min_observed_events = min_observed_events
        self.edge_score_weights = tuple(weight / total for weight in edge_score_weights)
        self.temporal = temporal_analyzer or TemporalEdgeAnalyzer(temporal_decay)
        self.edge_scorer = edge_scorer or EdgeInfluenceScorer(
            model=edge_model,
            api_key=edge_api_key,
            base_url=edge_base_url,
            feature_weights=message_feature_weights,
            attack_context=attack_context,
        )
        self.selection_strategy = selection_strategy
        self._random = random.Random(random_seed)
        self.fixed_target = fixed_target
        self.fixed_edge = fixed_edge

    def select(
        self,
        observer: CommunicationObserver,
        attackable_agents: Iterable[str],
        *,
        topology_graph: Mapping[str, Iterable[str]],
        candidate_edges: Iterable[tuple[str, str]],
        task: str = "",
        current_agent_outputs: Optional[Mapping[str, str]] = None,
        current_round: int = 1,
        total_rounds: int = 3,
        terminal_agents: Iterable[str] = (),
        judge_reads_all_rounds: bool = False,
    ) -> SelectionResult:
        attackable = set(attackable_agents)
        live_edges = tuple(
            sorted(
                {
                    (sender, receiver)
                    for sender, receiver in candidate_edges
                    if receiver in attackable
                }
            )
        )
        candidates = tuple(sorted({receiver for _, receiver in live_edges}))
        if len(observer.events) < self.min_observed_events or not candidates:
            return SelectionResult(
                target=None,
                event_count=len(observer.events),
                candidate_agents=candidates,
                candidate_edges=live_edges,
                edge_rankings={},
            )

        edge_rankings: Dict[str, Dict[str, object]] = {}
        if self.selection_strategy == "score":
            agent_names = set(attackable)
            agent_names.update(current_agent_outputs or {})
            agent_names.update(agent for agent in topology_graph if agent.startswith("A"))
            agent_names.update(
                receiver
                for receivers in topology_graph.values()
                for receiver in receivers
                if receiver.startswith("A")
            )
            temporal_scores = self.temporal.analyze(
                topology_graph,
                agent_names,
                live_edges,
                current_round=current_round,
                total_rounds=total_rounds,
                terminal_agents=terminal_agents,
                judge_reads_all_rounds=judge_reads_all_rounds,
            )
            message_scores, message_features, message_methods, message_diagnostics = self.edge_scorer.score(
                observer,
                live_edges,
                task=task,
                current_agent_outputs=current_agent_outputs,
                current_round=current_round,
                total_rounds=total_rounds,
                terminal_agents=tuple(terminal_agents),
                judge_reads_all_rounds=judge_reads_all_rounds,
            )
            reachability_weight, irreplaceability_weight, message_weight = self.edge_score_weights
            for sender, receiver in live_edges:
                edge_key = f"{sender}->{receiver}"
                temporal = temporal_scores[edge_key]
                reachable = bool(temporal["temporally_reachable"])
                message_influence = message_scores[edge_key]
                active_weight = reachability_weight + irreplaceability_weight
                weighted_score = (
                    reachability_weight * float(temporal["temporal_reachability"])
                    + irreplaceability_weight * float(temporal["irreplaceability"])
                )
                if message_influence is not None:
                    active_weight += message_weight
                    weighted_score += message_weight * message_influence
                score = weighted_score / active_weight if reachable and active_weight > 0 else 0.0
                edge_rankings[edge_key] = {
                    "sender": sender,
                    "receiver": receiver,
                    **temporal,
                    "message_influence": message_influence,
                    "message_influence_available": message_influence is not None,
                    "message_features": message_features[edge_key],
                    "message_method": message_methods[edge_key],
                    "message_diagnostic": message_diagnostics[edge_key],
                    "score": score,
                }
        else:
            for sender, receiver in live_edges:
                edge_rankings[f"{sender}->{receiver}"] = {
                    "sender": sender,
                    "receiver": receiver,
                    "selection_strategy": self.selection_strategy,
                    "score": None,
                }

        if self.selection_strategy == "fixed_edge":
            if self.fixed_edge not in live_edges:
                edge = "->".join(self.fixed_edge or ("?", "?"))
                raise ValueError(f"forced edge {edge} is not a live candidate edge")
            selected_edge = self.fixed_edge
            target = selected_edge[1]
        elif self.selection_strategy == "fixed":
            target = self.fixed_target if self.fixed_target in candidates else None
            selected_edge = next((edge for edge in live_edges if edge[1] == target), None)
        elif self.selection_strategy == "random":
            selected_edge = self._random.choice(live_edges)
            target = selected_edge[1]
        else:
            reachable_edges = [
                edge
                for edge in live_edges
                if edge_rankings[f"{edge[0]}->{edge[1]}"]["temporally_reachable"]
            ]
            if not reachable_edges:
                return SelectionResult(
                    target=None,
                    event_count=len(observer.events),
                    candidate_agents=candidates,
                    candidate_edges=live_edges,
                    edge_rankings=edge_rankings,
                )
            best_score = max(
                float(edge_rankings[f"{sender}->{receiver}"]["score"])
                for sender, receiver in reachable_edges
            )
            tied_edges = [
                edge
                for edge in reachable_edges
                if float(edge_rankings[f"{edge[0]}->{edge[1]}"]["score"]) == best_score
            ]
            selected_edge = self._random.choice(tied_edges)
            target = selected_edge[1]

        return SelectionResult(
            target=target,
            event_count=len(observer.events),
            candidate_agents=candidates,
            selected_edge=selected_edge,
            candidate_edges=live_edges,
            edge_rankings=edge_rankings,
        )
