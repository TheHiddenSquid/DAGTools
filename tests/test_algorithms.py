import random

import numpy as np
import pytest

from ColoredNodes.GreedyColorfuncs import CausalGreedyColor
from ColoredNodes.GreedyDPfuncs import CausalGreedyDP
from ColoredNodes.MCMCfuncs import CausalMCMC
from daglab.algorithms import execute, validate_settings
from daglab.models import Graph, RunSettings, demo_graph
from daglab.scoring import simulate


def grouping(graph):
    groups = {}
    for node in graph["nodes"]:
        groups.setdefault(node["group"], set()).add(node["id"])
    return {frozenset(block) for block in groups.values()}


@pytest.mark.parametrize("algorithm", ["mcmc", "greedy_color", "greedy_dp"])
def test_real_algorithms_return_consistent_scores_and_dags(algorithm):
    graph = demo_graph()
    data, columns = simulate(graph, 250, 1)
    events = []
    result, history = execute(algorithm, data, columns, graph,
        RunSettings(num_iters=100, num_starts=2, record_history=True), events.append)
    assert np.isfinite(result["score"])
    assert events and events[0]["iteration"] == 0
    Graph.model_validate(result["graph"])
    assert all("directed" in edge for edge in result["cpdag_edges"])
    if algorithm == "mcmc":
        assert len(history) == 101


@pytest.mark.parametrize("algorithm", ["mcmc", "greedy_color"])
def test_fixed_partition_survives_search_and_refinement(algorithm):
    graph = demo_graph()
    data, columns = simulate(graph, 300, 8)
    result, _ = execute(algorithm, data, columns, graph,
        RunSettings(num_iters=200, num_starts=2, fixed_partition=True, refine_partition=True), lambda _: None)
    assert grouping(result["graph"]) == grouping(graph.model_dump())


def test_map_visits_are_not_mislabelled_as_bic():
    graph = demo_graph()
    data, columns = simulate(graph, 200, 6)
    result, _ = execute("mcmc", data, columns, graph, RunSettings(mode="map", num_iters=30), lambda _: None)
    assert result["diagnostics"]["visit_count"] >= 1
    assert result["score"] < 0


def test_seed_reproduces_search():
    graph = demo_graph()
    data, columns = simulate(graph, 250, 1)
    settings = RunSettings(num_iters=100, seed=5)
    first, _ = execute("mcmc", data, columns, graph, settings, lambda _: None)
    second, _ = execute("mcmc", data, columns, graph, settings, lambda _: None)
    assert first == second


@pytest.mark.parametrize("solver", [CausalMCMC, CausalGreedyColor, CausalGreedyDP])
def test_legacy_return_contract_remains_three_values(solver):
    graph = demo_graph()
    data, _ = simulate(graph, 200, 4)
    random.seed(1)
    np.random.seed(1)
    kwargs = {"num_iters": 2} if solver is CausalMCMC else {"num_starts": 1}
    assert len(solver(data, **kwargs)) == 3


def test_single_variable_does_not_loop_forever():
    graph = Graph(nodes=[{"id": "x1", "label": "X1"}])
    data, columns = simulate(graph, 100, 1)
    result, _ = execute("mcmc", data, columns, graph, RunSettings(), lambda _: None)
    assert result["diagnostics"]["no_moves_available"]


def test_invalid_algorithm_combinations_rejected():
    with pytest.raises(ValueError, match="fixing"):
        validate_settings("greedy_dp", RunSettings(fixed_partition=True))
    with pytest.raises(ValueError, match="history"):
        validate_settings("mcmc", RunSettings(record_history=True, num_iters=3_000_000))


def test_target_stopping_at_initial_state_keeps_history_consistent():
    graph = demo_graph()
    data, columns = simulate(graph, 200, 1)
    result, history = execute("mcmc", data, columns, graph,
        RunSettings(num_iters=100, target_bic=-100, record_history=True), lambda _: None)
    assert result["diagnostics"]["num_iters"] == 0
    assert result["diagnostics"]["target_reached"]
    assert len(history) == 1


@pytest.mark.parametrize('center', [True, False])
@pytest.mark.parametrize('start', [True, False])
def test_ges_matches_live_bic_and_preserves_equivalence_class(center, start):
    from ges.utils import dag_to_cpdag
    from daglab.scoring import Scorer, graph_arrays
    graph = demo_graph()
    data, columns = simulate(graph, 400, 12)
    data += np.arange(5) * 2
    events = []
    result, history = execute('ges', data, columns, graph,
        RunSettings(center=center, start_from_candidate=start,
                    ges_phases=['forward', 'backward', 'turning'], ges_iterate=True), events.append)
    output = Graph.model_validate(result['graph'])
    raw, partition = graph_arrays(output, columns)
    assert len(partition) == 5
    assert history is None
    assert result['score'] == pytest.approx(Scorer(data, columns, center).score(output)['score'])
    assert result['diagnostics']['native_bic'] / len(data) == pytest.approx(result['score'])
    assert events[0]['iteration'] == 0
    assert events[-1]['score'] == pytest.approx(result['score'])
    assert all(b['score'] >= a['score'] - 1e-8 for a, b in zip(events, events[1:]))
    cpdag = dag_to_cpdag(raw)
    reconstructed = np.zeros_like(cpdag)
    ids = {c['id']: i for i,c in enumerate(columns)}
    for e in result['cpdag_edges']:
        a,b = ids[e['source']],ids[e['target']]
        reconstructed[a,b] = 1
        if not e['directed']:
            reconstructed[b,a] = 1
    np.testing.assert_array_equal(cpdag, reconstructed)


def test_ges_rejects_collinear_columns_and_duplicate_phases():
    graph = demo_graph()
    data, columns = simulate(graph, 100, 3)
    data[:,1] = data[:,0]
    with pytest.raises(ValueError, match='independent'):
        execute('ges', data, columns, graph, RunSettings(), lambda _: None)
    with pytest.raises(ValueError, match='only once'):
        validate_settings('ges', RunSettings(ges_phases=['forward', 'forward']))
