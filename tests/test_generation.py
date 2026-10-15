import numpy as np
import pytest
from pydantic import ValidationError
from daglab.generation import random_graph, random_partition
from daglab.models import Graph, RandomGraphRequest, RandomPartitionRequest, demo_graph


def test_seeded_generation_bounds_and_acyclicity():
    request = RandomGraphRequest(p=20, num_colors=4, density=0.6, lambda_min=0.4,
                                 lambda_max=0.8, omega_min=0.5, omega_max=1.2, seed=15)
    first = random_graph(request)
    assert first == random_graph(request)
    Graph.model_validate(first.model_dump())
    assert all(0.4 <= abs(e.weight) <= 0.8 for e in first.edges)
    assert all(0.5 <= v <= 1.2 for v in first.variances.values())
    assert len(first.variances) <= 4
    assert len({(n.x, n.y) for n in first.nodes}) == 20
    # The API generator must not disturb legacy algorithms' global RNG.
    np.random.seed(5)
    expected = np.random.random()
    np.random.seed(5)
    random_graph(request)
    assert np.random.random() == expected


@pytest.mark.parametrize('density,edges', [(0, 0), (1, 10)])
def test_density_endpoints_and_template_identity(density, edges):
    graph = demo_graph()
    graph.nodes[0].label = 'Custom name'
    result = random_graph(RandomGraphRequest(graph=graph, p=5, density=density))
    assert len(result.edges) == edges
    assert [(n.id, n.label) for n in result.nodes] == [(n.id, n.label) for n in graph.nodes]


def test_partition_preserves_edges_and_positions():
    graph = demo_graph()
    request = RandomPartitionRequest(graph=graph, num_colors=1, omega_min=0.7, omega_max=0.7)
    result = random_partition(request)
    assert result == random_partition(request)
    assert result.edges == graph.edges
    assert [(n.id, n.label, n.x, n.y) for n in result.nodes] == [(n.id, n.label, n.x, n.y) for n in graph.nodes]
    assert {n.group for n in result.nodes} == {'1'}
    assert result.variances == {'1': 0.7}


@pytest.mark.parametrize('values', [dict(p=2, num_colors=3), dict(density=1.1),
    dict(omega_min=2, omega_max=1), dict(lambda_min=2, lambda_max=1), dict(omega_min=0),
    dict(graph=demo_graph(), p=6), dict(seed=-1)])
def test_invalid_generator_parameters(values):
    with pytest.raises(ValidationError):
        RandomGraphRequest(**values)
