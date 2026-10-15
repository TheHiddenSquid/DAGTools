"""Seeded model generators using the research code's distributions and defaults."""
import math
import numpy as np

from .models import Edge, Graph, Node, RandomGraphRequest, RandomPartitionRequest


def circular_layout(graph):
    radius = max(150, len(graph.nodes) * 120 / (2 * math.pi))
    nodes = [node.model_copy(update={
        "x": radius + radius * math.cos(-math.pi / 2 + 2 * math.pi * i / len(graph.nodes)),
        "y": radius + radius * math.sin(-math.pi / 2 + 2 * math.pi * i / len(graph.nodes)),
    }) for i, node in enumerate(graph.nodes)]
    return graph.model_copy(update={"nodes": nodes})


def _partition(graph, num_colors, omega_min, omega_max, rng):
    # Same independent uniform group assignment as generate_random_partition;
    # unused colors are dropped and the occupied groups are numbered in order.
    assignments = rng.integers(num_colors, size=len(graph.nodes))
    mapping = {color: str(i + 1) for i, color in enumerate(sorted(set(assignments)))}
    nodes = [node.model_copy(update={"group": mapping[color]})
             for node, color in zip(graph.nodes, assignments)]
    variances = {group: float(rng.uniform(omega_min, omega_max)) for group in mapping.values()}
    return graph.model_copy(update={"nodes": nodes, "variances": variances})


def random_partition(request: RandomPartitionRequest):
    return _partition(request.graph, request.num_colors, request.omega_min,
                      request.omega_max, np.random.default_rng(request.seed))


def random_graph(request: RandomGraphRequest):
    rng = np.random.default_rng(request.seed)
    graph = request.graph or Graph(nodes=[Node(id=f"x{i+1}", label=f"X{i+1}") for i in range(request.p)])
    graph = _partition(graph, request.num_colors, request.omega_min, request.omega_max, rng)
    order = rng.permutation(request.p)
    edges = []
    for i, a in enumerate(order):
        for b in order[i+1:]:
            if rng.random() < request.density:
                source, target = graph.nodes[a].id, graph.nodes[b].id
                weight = float(rng.uniform(request.lambda_min, request.lambda_max) * rng.choice([-1, 1]))
                edges.append(Edge(id=f"{source}-{target}", source=source, target=target, weight=weight))
    return circular_layout(graph.model_copy(update={"edges": edges}))
