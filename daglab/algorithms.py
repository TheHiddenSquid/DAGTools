"""Algorithm registry: add an adapter and its field descriptions here.

Adapters run in a fresh worker process, never in the API process. UI field
definitions are served by /api/algorithms rather than duplicated in React.
"""
import random

import numpy as np

from .models import Graph, RunSettings
from .scoring import Scorer, graph_arrays


def field(name, label, kind, **kwargs):
    return {"name": name, "label": label, "type": kind, **kwargs}


COMMON = [
    field("seed", "Random seed", "number", min=0, max=2**32-1, step=1),
    field("center", "Center sample columns", "boolean", hint="Subtract column means; do not standardize."),
    field("start_from_candidate", "Use DAG from Explore & score", "boolean",
          hint="Uses the current candidate graph. Greedy searches seed their first restart with it."),
]
MOVES = [{"value": value, "label": label} for value, label in [
    ("add_edge", "Add edge"), ("remove_edge", "Remove edge"),
    ("flip_edge", "Reverse edge"), ("change_color", "Change color")]]
REFINE = field("refine_partition", "Refine final partition", "boolean",
               hint="Optimize colors after the graph search.", visible_if={"fixed_partition": False})


REGISTRY = {
    "ges": {
        "id": "ges", "name": "GES · Gaussian BIC",
        "description": "Greedy equivalence search with a separate noise variance per variable. Candidate colors are ignored. Returns a CPDAG and a consistent DAG for further exploration or search.",
        "source": "ges 1.1.1 · ges.fit", "progress_unit": "accepted moves",
        "fields": [field("ges_phases", "Search phases (in this order)", "multiselect", options=[
            {"value": "forward", "label": "Forward · add"},
            {"value": "backward", "label": "Backward · remove"},
            {"value": "turning", "label": "Turning · reorient"}]),
            field("ges_iterate", "Repeat phases until convergence", "boolean")] + COMMON[1:],
    },
    "mcmc": {
        "id": "mcmc", "name": "MCMC", "description": "Search over DAGs and variance partitions with your MCMC implementation.",
        "source": "ColoredNodes/MCMCfuncs.py · CausalMCMC", "progress_unit": "iterations",
        "fields": [
            field("num_iters", "Iterations", "number", min=1, max=100_000_000, step=1000),
            field("mode", "Return estimate", "select", options=[{"value": "bic", "label": "Highest BIC"}, {"value": "map", "label": "Most visited model (MAP mode)"}]),
            field("fixed_partition", "Fix candidate partition", "boolean", hint="Disable color proposals and final partition refinement."),
            field("color_probability", "Probability of a color move", "number", min=0, max=1, step=0.05, visible_if={"fixed_partition": False}),
            REFINE,
            field("target_bic", "Stop at BIC score (optional)", "optional_number", step=0.01, visible_if={"mode": "bic"}),
            field("record_history", "Save every iteration's score", "boolean", hint="Optional exact history; limited to 2 million iterations."),
            field("chunk_size", "Random draw batch size", "number", min=1, max=1_000_000, step=1000,
                  hint="Changing this changes the seeded trajectory."),
        ] + COMMON,
    },
    "greedy_color": {
        "id": "greedy_color", "name": "Greedy · graph + colors", "description": "Joint local search over edge moves and color assignments.",
        "source": "ColoredNodes/GreedyColorfuncs.py · CausalGreedyColor", "progress_unit": "restarts",
        "fields": [field("num_starts", "Random restarts", "number", min=1, max=100, step=1),
                   field("fixed_partition", "Fix candidate partition", "boolean"),
                   field("moves", "Allowed moves", "multiselect", options=MOVES), REFINE] + COMMON,
    },
    "greedy_dp": {
        "id": "greedy_dp", "name": "Greedy · optimal partition", "description": "Edge search with dynamic programming to optimize the variance partition at each step.",
        "source": "ColoredNodes/GreedyDPfuncs.py · CausalGreedyDP", "progress_unit": "restarts",
        "fields": [field("num_starts", "Random restarts", "number", min=1, max=100, step=1),
                   field("moves", "Allowed moves", "multiselect", options=MOVES[:3])] + COMMON,
    },
}


def validate_settings(algorithm, settings: RunSettings):
    if algorithm not in REGISTRY:
        raise ValueError("Unknown algorithm.")
    if algorithm == "ges":
        if len(set(settings.ges_phases)) != len(settings.ges_phases):
            raise ValueError("Choose each GES phase only once.")
        return []
    if algorithm == "greedy_dp" and settings.fixed_partition:
        raise ValueError("Greedy DP optimizes the partition and does not support fixing it.")
    if algorithm == "mcmc" and settings.mode == "map" and settings.target_bic is not None:
        raise ValueError("Target BIC stopping is available only in highest-BIC mode.")
    if algorithm == "mcmc" and settings.record_history and settings.num_iters > 2_000_000:
        raise ValueError("Exact history is limited to 2 million iterations. Disable it to run a longer chain; the live trace remains available.")
    moves = list(settings.moves)
    if algorithm == "greedy_dp" or settings.fixed_partition:
        moves = [move for move in moves if move != "change_color"]
    if algorithm != "mcmc" and not moves:
        raise ValueError("Choose at least one allowed move.")
    return moves


def as_graph(adjacency, partition, template: Graph, columns):
    by_id = {node.id: node for node in template.nodes}
    groups = {int(node): str(i+1) for i, block in enumerate(partition) for node in block}
    nodes = [by_id[column["id"]].model_copy(update={"group": groups[i]}) for i, column in enumerate(columns)]
    edges = [{"id": f"{columns[i]['id']}-{columns[j]['id']}", "source": columns[i]["id"],
              "target": columns[j]["id"], "weight": 0.7} for i, j in zip(*np.nonzero(adjacency))]
    return Graph(nodes=nodes, edges=edges, variances={group: 1.0 for group in groups.values()})


def execute(algorithm, data, columns, template, settings, progress):
    moves = validate_settings(algorithm, settings)
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    adjacency, partition = graph_arrays(template, columns)
    initial = adjacency if settings.start_from_candidate else None
    initial_partition = partition if settings.fixed_partition or settings.start_from_candidate else None
    refine = settings.refine_partition and not settings.fixed_partition
    history = None
    diagnostics = {}
    if len(columns) == 1:
        scored = Scorer(data, columns, settings.center).score(template)
        graph = template.model_copy(deep=True)
        graph.variances[graph.nodes[0].group] = scored["blocks"][0]["variance"]
        progress({"iteration": 0, "total": 0, "score": scored["score"], "best_score": scored["score"]})
        return {"graph": graph.model_dump(), "cpdag_edges": [], "score": scored["score"],
                "score_details": scored, "diagnostics": {"num_iters": 0, "no_moves_available": True}}, None
    if algorithm == "ges":
        from .ges_adapter import search
        cpdag, partition, native_score, raw, diagnostics = search(data, initial, settings, progress)
    elif algorithm == "mcmc":
        from ColoredNodes.MCMCfuncs import CausalMCMC
        probability = 0.0 if settings.fixed_partition else settings.color_probability
        output = CausalMCMC(data, num_iters=settings.num_iters, mode=settings.mode,
            move_weights=[probability, 1-probability], A0=initial, P0=initial_partition,
            center=settings.center, refine_partition=refine, target_bic=settings.target_bic,
            return_history=settings.record_history, chunk_size=settings.chunk_size,
            return_dag=True, on_progress=progress)
        cpdag, partition, native_score, raw = output[:4]
        if len(output) > 4:
            diagnostics = output[4]
            history = diagnostics.pop("bic_history")
        if settings.mode == "map":
            diagnostics["visit_count"] = int(native_score)
    elif algorithm == "greedy_color":
        from ColoredNodes.GreedyColorfuncs import CausalGreedyColor
        cpdag, partition, native_score, raw = CausalGreedyColor(data,
            num_starts=settings.num_starts, moves=moves, center=settings.center,
            refine_partition=refine, A0=initial, P0=initial_partition,
            return_dag=True, on_progress=progress)
    elif algorithm == "greedy_dp":
        from ColoredNodes.GreedyDPfuncs import CausalGreedyDP
        cpdag, partition, native_score, raw = CausalGreedyDP(data,
            num_starts=settings.num_starts, moves=moves, center=settings.center,
            A0=initial, return_dag=True, on_progress=progress)
    else:
        raise ValueError("No adapter is registered for this algorithm.")
    result_graph = as_graph(raw, partition, template, columns)
    scored = Scorer(data, columns, settings.center).score(result_graph)
    if algorithm != "mcmc" or settings.mode == "bic":
        if not np.isclose(scored["score"], native_score, rtol=1e-7, atol=1e-8):
            raise ValueError("Algorithm and live scorer disagree; this result was not accepted.")
    cpdag_edges = []
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            if cpdag[i, j] and cpdag[j, i]:
                cpdag_edges.append({"source": columns[i]["id"], "target": columns[j]["id"], "directed": False})
            elif cpdag[i, j] or cpdag[j, i]:
                a, b = (i, j) if cpdag[i, j] else (j, i)
                cpdag_edges.append({"source": columns[a]["id"], "target": columns[b]["id"], "directed": True})
    # Store fitted parameters so an estimated DAG can also generate new data.
    centered = data - data.mean(axis=0) if settings.center else data
    for edge in result_graph.edges:
        j = next(i for i, c in enumerate(columns) if c["id"] == edge.target)
        parents = np.flatnonzero(raw[:, j])
        coefficients = np.linalg.lstsq(centered[:, parents], centered[:, j], rcond=None)[0]
        i = next(i for i, c in enumerate(columns) if c["id"] == edge.source)
        edge.weight = float(coefficients[list(parents).index(i)])
    for block in scored["blocks"]:
        group = next(node.group for node in result_graph.nodes if node.id == block["nodes"][0])
        result_graph.variances[group] = block["variance"]
    return {"graph": result_graph.model_dump(), "cpdag_edges": cpdag_edges,
            "score": scored["score"], "score_details": scored,
            "diagnostics": diagnostics}, history
