import numpy as np
import pytest
from pydantic import ValidationError

from ColoredNodes import utils
from daglab.models import Graph, demo_graph
from daglab.scoring import Scorer, graph_arrays, simulate


@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("structure", ["full", "empty", "one_group", "singletons"])
def test_live_score_matches_original_research_code(center, structure):
    graph = demo_graph()
    data, columns = simulate(graph, 500, 7)
    data += np.array([1, -2, 3, 0.1, -0.5])
    if structure == "empty":
        graph.edges = []
    elif structure in ("one_group", "singletons"):
        for i, node in enumerate(graph.nodes):
            node.group = "1" if structure == "one_group" else str(i+1)
    adjacency, partition = graph_arrays(graph, columns)
    before = data.copy()
    actual = Scorer(data, columns, center).score(graph)["score"]
    expected = utils.score_DAG(data, adjacency, partition, center=center)
    assert actual == pytest.approx(expected, abs=1e-10)
    np.testing.assert_array_equal(data, before)


def test_profile_partition_and_cached_edge_changes():
    graph = demo_graph()
    data, columns = simulate(graph, 700, 3)
    scorer = Scorer(data, columns)
    initial = scorer.score(graph)
    assert scorer.score(graph, optimize=True)["score"] >= initial["score"] - 1e-10
    assert len(scorer.cache) == 5
    graph.edges.pop()
    scorer.score(graph)
    assert len(scorer.cache) == 6


def test_sampling_coefficients_and_shared_variance():
    graph = demo_graph()
    sample, columns = simulate(graph, 50000, 92)
    again, _ = simulate(graph, 50000, 92)
    np.testing.assert_array_equal(sample, again)
    index = {c["id"]: i for i, c in enumerate(columns)}
    residual = sample.copy()
    for edge in graph.edges:
        residual[:, index[edge.target]] -= edge.weight * sample[:, index[edge.source]]
    expected = [graph.variances[n.group] for n in graph.nodes]
    np.testing.assert_allclose(np.var(residual, axis=0), expected, rtol=0.025)
    assert np.max(np.abs(np.corrcoef(residual.T)-np.eye(5))) < 0.02


def test_cycles_and_data_mismatch_rejected():
    payload = demo_graph().model_dump()
    payload["edges"].append({"id": "back", "source": "x5", "target": "x1"})
    with pytest.raises(ValidationError, match="cycle"):
        Graph.model_validate(payload)
    graph = demo_graph()
    data, columns = simulate(graph, 100, 2)
    graph.nodes[0].label = "wrong"
    with pytest.raises(ValueError, match="names"):
        Scorer(data, columns).score(graph)


def test_singular_parent_sets_are_reported():
    graph = demo_graph()
    data, columns = simulate(graph, 100, 2)
    data[:, 2] = data[:, 1]
    with pytest.raises(ValueError, match="collinear"):
        Scorer(data, columns).score(graph)
