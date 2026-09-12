"""Topology metrics calculated from AIRA's reconstructed directed graph."""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, Mapping, Sequence


class TopologyAnalyzer:
    """Compute normalized degree, betweenness, and closeness centralities."""

    def __init__(self, weights: Sequence[float] = (1 / 3, 1 / 3, 1 / 3)) -> None:
        if len(weights) != 3 or any(weight < 0 for weight in weights) or sum(weights) == 0:
            raise ValueError("topology weights must contain three non-negative values with positive sum")
        total = float(sum(weights))
        self.weights = tuple(weight / total for weight in weights)

    @staticmethod
    def _nodes(graph: Mapping[str, Iterable[str]], nodes: Iterable[str] = ()) -> list[str]:
        return sorted(set(nodes) | set(graph) | {receiver for receivers in graph.values() for receiver in receivers})

    @staticmethod
    def _shortest_paths(source: str, graph: Mapping[str, Iterable[str]]) -> Dict[str, int]:
        distances = {source: 0}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for neighbor in graph.get(node, ()):
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        return distances

    def analyze(self, graph: Mapping[str, Iterable[str]], nodes: Iterable[str] = ()) -> Dict[str, Dict[str, float]]:
        node_list = self._nodes(graph, nodes)
        count = len(node_list)
        adjacency = {node: list(graph.get(node, ())) for node in node_list}
        if count <= 1:
            return {node: {"degree": 0.0, "betweenness": 0.0, "closeness": 0.0, "score": 0.0} for node in node_list}

        incoming = {node: 0 for node in node_list}
        for receivers in adjacency.values():
            for receiver in receivers:
                incoming[receiver] += 1
        degree = {
            node: (len(adjacency[node]) + incoming[node]) / (2 * (count - 1))
            for node in node_list
        }

        # Brandes' algorithm for unweighted directed betweenness centrality.
        betweenness = {node: 0.0 for node in node_list}
        for source in node_list:
            stack = []
            predecessors = {node: [] for node in node_list}
            paths = {node: 0.0 for node in node_list}
            paths[source] = 1.0
            distance = {source: 0}
            queue = deque([source])
            while queue:
                vertex = queue.popleft()
                stack.append(vertex)
                for neighbor in adjacency[vertex]:
                    if neighbor not in distance:
                        distance[neighbor] = distance[vertex] + 1
                        queue.append(neighbor)
                    if distance[neighbor] == distance[vertex] + 1:
                        paths[neighbor] += paths[vertex]
                        predecessors[neighbor].append(vertex)
            dependency = {node: 0.0 for node in node_list}
            while stack:
                vertex = stack.pop()
                for predecessor in predecessors[vertex]:
                    dependency[predecessor] += (paths[predecessor] / paths[vertex]) * (1 + dependency[vertex])
                if vertex != source:
                    betweenness[vertex] += dependency[vertex]
        scale = (count - 1) * (count - 2)
        if scale:
            betweenness = {node: value / scale for node, value in betweenness.items()}

        closeness = {}
        for node in node_list:
            distances = self._shortest_paths(node, adjacency)
            reachable = len(distances) - 1
            distance_sum = sum(distance for target, distance in distances.items() if target != node)
            closeness[node] = (reachable * reachable / ((count - 1) * distance_sum)) if distance_sum else 0.0

        w_degree, w_between, w_close = self.weights
        return {
            node: {
                "degree": degree[node],
                "betweenness": betweenness[node],
                "closeness": closeness[node],
                "score": w_degree * degree[node] + w_between * betweenness[node] + w_close * closeness[node],
            }
            for node in node_list
        }
