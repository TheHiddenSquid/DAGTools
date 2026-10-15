"""Cached Gaussian scores matching ColoredNodes.utils.score_DAG.

S = -.5 sum_B |B| [log(mean residual variance in B) + 1]
    - log(n)/(2n) (number of edges + number of nonempty blocks).
The constant -p log(2*pi)/2 and the constant mean-parameter penalty are
omitted, as in the research code. Higher is better on the SAME dataset.
"""
import math
from collections import OrderedDict

import numpy as np

from .models import Graph


def graph_arrays(graph: Graph, columns: list[dict]):
    ids = [column["id"] for column in columns]
    if {node.id for node in graph.nodes} != set(ids):
        raise ValueError("The candidate must contain exactly the dataset's variables.")
    lookup = {node.id: node for node in graph.nodes}
    if any(lookup[c["id"]].label != c["label"] for c in columns):
        raise ValueError("Candidate variable names must match the dataset columns.")
    index = {node_id: i for i, node_id in enumerate(ids)}
    adjacency = np.zeros((len(ids), len(ids)), dtype=np.int64)
    for edge in graph.edges:
        adjacency[index[edge.source], index[edge.target]] = 1
    blocks = {}
    for node_id in ids:
        blocks.setdefault(lookup[node_id].group, []).append(index[node_id])
    return adjacency, list(blocks.values())


class Scorer:
    def __init__(self, data: np.ndarray, columns: list[dict], center=True):
        self.columns = columns
        self.n, self.p = data.shape
        self.center = center
        data = data - data.mean(axis=0) if center else data
        self.gram = data.T @ data / self.n
        self.penalty = math.log(self.n) / (2 * self.n)
        self.cache = OrderedDict()

    def residual(self, node, parents):
        key = node, tuple(parents)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        value = self.gram[node, node]
        if parents:
            if len(parents) >= self.n - int(self.center):
                raise ValueError("Too few observations to fit this parent set.")
            sub = self.gram[np.ix_(parents, parents)]
            if np.linalg.matrix_rank(sub) < len(parents):
                raise ValueError("Some parent variables are collinear. Remove redundant parents.")
            cross = self.gram[parents, node]
            value -= cross @ np.linalg.solve(sub, cross)
        if not np.isfinite(value) or value <= max(1e-14, abs(self.gram[node, node]) * 1e-12):
            raise ValueError("A variable has zero or near-zero residual variance for this graph.")
        self.cache[key] = float(value)
        if len(self.cache) > 20000:
            self.cache.popitem(last=False)
        return float(value)

    def score(self, graph: Graph, optimize=False):
        adjacency, blocks = graph_arrays(graph, self.columns)
        residuals = [self.residual(i, np.flatnonzero(adjacency[:, i]).tolist()) for i in range(self.p)]
        if optimize:
            from ColoredNodes.utils import get_optimal_partition
            blocks, _ = get_optimal_partition(residuals, self.penalty)
        terms = []
        for block in blocks:
            indices = sorted(block)
            variance = sum(residuals[i] for i in indices) / len(indices)
            terms.append({"nodes": [self.columns[i]["id"] for i in indices], "variance": variance,
                          "contribution": -0.5 * len(indices) * (math.log(variance) + 1) - self.penalty})
        edge_penalty = self.penalty * int(adjacency.sum())
        score = sum(term["contribution"] for term in terms) - edge_penalty
        return {"score": score, "higher_is_better": True, "edges": int(adjacency.sum()),
                "groups": len(blocks), "parameters": int(adjacency.sum()) + len(blocks),
                "edge_penalty": edge_penalty, "blocks": terms, "center": self.center,
                "sample_size": self.n, "convention": "DAGdiscovery normalized Gaussian BIC"}


def simulate(graph: Graph, n: int, seed: int):
    """Topological SEM simulation; coefficients follow source -> target."""
    import networkx as nx
    columns = [{"id": node.id, "label": node.label} for node in graph.nodes]
    adjacency, _ = graph_arrays(graph, columns)
    index = {node.id: i for i, node in enumerate(graph.nodes)}
    weights = np.zeros_like(adjacency, dtype=float)
    for edge in graph.edges:
        weights[index[edge.source], index[edge.target]] = edge.weight
    rng = np.random.default_rng(seed)
    variances = [graph.variances.get(node.group, 1.0) for node in graph.nodes]
    data = rng.normal(size=(n, len(columns))) * np.sqrt(variances)
    for node in nx.topological_sort(nx.DiGraph(adjacency)):
        parents = np.flatnonzero(adjacency[:, node])
        if len(parents):
            data[:, node] += data[:, parents] @ weights[parents, node]
    if not np.isfinite(data).all():
        raise ValueError("These coefficients produce non-finite values; reduce their magnitudes.")
    return data, columns
