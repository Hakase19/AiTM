"""Online adaptive target selection for AIRA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence

from analysis.influence_estimator import CommunicationInfluenceEstimator
from analysis.role_inference import RoleInference
from analysis.topology_analyzer import TopologyAnalyzer
from observer.communication_observer import CommunicationObserver


@dataclass(frozen=True)
class SelectionResult:
    target: Optional[str]
    rankings: Dict[str, Dict[str, object]]
    all_scores: Dict[str, Dict[str, object]]
    event_count: int


class TargetSelector:
    """Rank pending, attackable agents using only observer state."""

    def __init__(
        self,
        *,
        min_observed_events: int = 2,
        topology_weights: Sequence[float] = (1 / 3, 1 / 3, 1 / 3),
        influence_weights: Sequence[float] = (0.5, 0.5),
        score_weights: Sequence[float] = (1 / 3, 1 / 3, 1 / 3),
        role_inference_mode: str = "llm",
        role_model: str = "",
        role_api_key: str = "",
        role_base_url: str = "",
        role_importance: Optional[Mapping[str, float]] = None,
        require_observed_forwarding: bool = False,
    ) -> None:
        if min_observed_events < 1:
            raise ValueError("min_observed_events must be positive")
        if len(score_weights) != 3 or any(weight < 0 for weight in score_weights) or sum(score_weights) == 0:
            raise ValueError("score weights must contain three non-negative values with positive sum")
        total = float(sum(score_weights))
        self.min_observed_events = min_observed_events
        self.score_weights = tuple(weight / total for weight in score_weights)
        self.topology = TopologyAnalyzer(topology_weights)
        self.roles = RoleInference(
            role_importance,
            mode=role_inference_mode,
            model=role_model,
            api_key=role_api_key,
            base_url=role_base_url,
        )
        self.influence = CommunicationInfluenceEstimator(influence_weights)
        self.require_observed_forwarding = require_observed_forwarding

    def select(self, observer: CommunicationObserver, attackable_agents: Iterable[str]) -> SelectionResult:
        attackable = set(attackable_agents)
        graph = observer.estimated_graph()
        pending = observer.pending_receivers & attackable
        candidates = sorted(
            agent for agent in pending
            if not self.require_observed_forwarding or bool(graph.get(agent, ()))
        )
        if len(observer.events) < self.min_observed_events or not candidates:
            return SelectionResult(target=None, rankings={}, all_scores={}, event_count=len(observer.events))

        # Analysis inputs are deliberately restricted to the observer view.
        # ``attackable`` is used only to exclude non-agent endpoints such as a
        # terminal judge from the target candidate set.
        observed_agents = observer.observed_agents
        topology = self.topology.analyze(graph, observed_agents)
        roles = self.roles.infer(observer, observed_agents)
        influence = self.influence.estimate(observer, observed_agents)
        alpha, beta, gamma = self.score_weights
        all_scores = {
            agent: {
                "topology": topology.get(agent, {}).get("score", 0.0),
                "role": float(roles.get(agent, {}).get("score", 0.0)),
                "communication": influence.get(agent, {}).get("score", 0.0),
                "role_probabilities": roles.get(agent, {}).get("probabilities", {}),
                "role_method": roles.get(agent, {}).get("method"),
                "role_fallback_reason": roles.get(agent, {}).get("fallback_reason"),
                "topology_metrics": topology.get(agent, {}),
                "communication_metrics": influence.get(agent, {}),
            }
            for agent in sorted(observed_agents & attackable)
        }
        for values in all_scores.values():
            values["score"] = alpha * values["topology"] + beta * values["role"] + gamma * values["communication"]
        rankings = {
            agent: {
                **all_scores[agent],
                "candidate_evidence": (
                    "pending_delivery_and_observed_forwarding"
                    if graph.get(agent, ())
                    else "pending_delivery"
                ),
            }
            for agent in candidates
        }
        # Stable tie-breaking makes the experimental policy reproducible.
        target = max(candidates, key=lambda agent: (rankings[agent]["score"], agent))
        return SelectionResult(target=target, rankings=rankings, all_scores=all_scores, event_count=len(observer.events))
