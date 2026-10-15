"""Versioned, framework-independent inputs for the workbench."""
from typing import Literal
import math

from pydantic import BaseModel, ConfigDict, Field, model_validator
import networkx as nx


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Node(Model):
    id: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=80)
    group: str = Field(default="1", min_length=1, max_length=40)
    x: float = 0
    y: float = 0


class Column(Model):
    id: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=80)


class Edge(Model):
    id: str = Field(min_length=1, max_length=180)
    source: str
    target: str
    weight: float = 0.7


class Graph(Model):
    nodes: list[Node] = Field(min_length=1, max_length=100)
    edges: list[Edge] = Field(default_factory=list, max_length=4950)
    variances: dict[str, float] = Field(default_factory=lambda: {"1": 1.0})

    @model_validator(mode="after")
    def valid_graph(self):
        ids = [node.id for node in self.nodes]
        labels = [node.label for node in self.nodes]
        if len(set(ids)) != len(ids) or len(set(labels)) != len(labels):
            raise ValueError("Variable IDs and names must be unique.")
        pairs = [(edge.source, edge.target) for edge in self.edges]
        if len(set(pairs)) != len(pairs) or len({e.id for e in self.edges}) != len(self.edges):
            raise ValueError("Duplicate edges are not allowed.")
        if any(a not in ids or b not in ids or a == b for a, b in pairs):
            raise ValueError("Edges must connect two distinct existing variables.")
        graph = nx.DiGraph()
        graph.add_nodes_from(ids)
        graph.add_edges_from(pairs)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("This connection would create a cycle.")
        for node in self.nodes:
            variance = self.variances.get(node.group, 1.0)
            if not math.isfinite(variance) or variance <= 0:
                raise ValueError("Noise variances must be finite and greater than zero.")
        if any(not math.isfinite(v) or v <= 0 for v in self.variances.values()):
            raise ValueError("Noise variances must be finite and greater than zero.")
        return self


class GenerateRequest(Model):
    graph: Graph
    sample_size: int = Field(default=1000, ge=2, le=1_000_000)
    seed: int = Field(default=42, ge=0, le=2**32-1)
    name: str = Field(default="Generated sample", min_length=1, max_length=120)


class RandomParameters(Model):
    num_colors: int = Field(default=3, ge=1, le=100)
    omega_min: float = Field(default=0.2, gt=0)
    omega_max: float = Field(default=2.0, gt=0)
    seed: int = Field(default=42, ge=0, le=2**32-1)

    @model_validator(mode="after")
    def ordered_variances(self):
        if self.omega_max < self.omega_min:
            raise ValueError("omega_max must be at least omega_min.")
        return self


class RandomGraphRequest(RandomParameters):
    p: int = Field(default=5, ge=1, le=100)
    density: float = Field(default=0.5, ge=0, le=1)
    lambda_min: float = Field(default=0.25, ge=0)
    lambda_max: float = Field(default=1.0, ge=0)
    graph: Graph | None = None

    @model_validator(mode="after")
    def valid_parameters(self):
        if self.lambda_max < self.lambda_min:
            raise ValueError("lambda_max must be at least lambda_min.")
        if self.num_colors > self.p:
            raise ValueError("The number of available groups cannot exceed p.")
        if self.graph is not None and len(self.graph.nodes) != self.p:
            raise ValueError("p must match the existing variables.")
        return self


class RandomPartitionRequest(RandomParameters):
    graph: Graph

    @model_validator(mode="after")
    def valid_groups(self):
        if self.num_colors > len(self.graph.nodes):
            raise ValueError("The number of available groups cannot exceed p.")
        return self


class ScoreRequest(Model):
    graph: Graph
    dataset_id: str
    center: bool = True
    optimize_partition: bool = False


class RunSettings(Model):
    seed: int = Field(default=42, ge=0, le=2**32-1)
    center: bool = True
    start_from_candidate: bool = True
    fixed_partition: bool = False
    num_iters: int = Field(default=20000, ge=1, le=100_000_000)
    num_starts: int = Field(default=5, ge=1, le=100)
    mode: Literal["bic", "map"] = "bic"
    color_probability: float = Field(default=0.4, ge=0, le=1)
    moves: list[Literal["add_edge", "remove_edge", "flip_edge", "change_color"]] = Field(
        default_factory=lambda: ["add_edge", "remove_edge", "flip_edge", "change_color"])
    refine_partition: bool = True
    target_bic: float | None = None
    record_history: bool = False
    chunk_size: int = Field(default=10000, ge=1, le=1_000_000)
    ges_phases: list[Literal["forward", "backward", "turning"]] = Field(default_factory=lambda: ["forward", "backward"], min_length=1, max_length=3)
    ges_iterate: bool = False


class RunRequest(Model):
    dataset_id: str
    graph: Graph
    algorithm: str
    settings: RunSettings = Field(default_factory=RunSettings)


class Workspace(Model):
    schema_version: Literal[1] = 1
    name: str = Field(default="Untitled experiment", min_length=1, max_length=120)
    dataset_id: str | None = None
    source_graph: Graph
    candidate_graph: Graph
    algorithm: str = "mcmc"
    settings: RunSettings = Field(default_factory=RunSettings)
    center: bool = True


def demo_graph() -> Graph:
    graph = Graph.model_validate({
        "nodes": [{"id": f"x{i+1}", "label": f"X{i+1}", "group": group, "x": x, "y": y}
                  for i, (group, x, y) in enumerate([
                      ("1", 50, 160), ("1", 260, 40), ("2", 260, 280), ("2", 480, 160), ("3", 700, 160)])],
        "edges": [{"id": f"x{a}-x{b}", "source": f"x{a}", "target": f"x{b}", "weight": w}
                  for a, b, w in [(1, 2, 0.8), (1, 3, -0.6), (2, 4, 0.7), (3, 4, 0.9), (4, 5, -0.8)]],
        "variances": {"1": 0.5, "2": 1.0, "3": 1.6},
    })

    from .generation import circular_layout
    return circular_layout(graph)
