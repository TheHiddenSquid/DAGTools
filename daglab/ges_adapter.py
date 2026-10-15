"""Standard Gaussian GES, with normalized scores and a consistent DAG output."""
import numpy as np
import ges
from ges import utils
from ges.scores import GaussObsL0Pen


def search(data, initial, settings, progress):
    n, p = data.shape

    class GaussianScore(GaussObsL0Pen):
        def full_score(self, adjacency):
            # GES accepts a CPDAG initial state, whereas the library's Gaussian
            # full_score expects a DAG. Score a consistent extension instead.
            dag = utils.pdag_to_dag(adjacency)
            return sum(self.local_score(j, set(np.flatnonzero(dag[:, j]))) for j in range(self.p))

    score = GaussianScore(data)
    if not settings.center:
        score._scatter = data.T @ data / n
    if np.linalg.matrix_rank(score._scatter) < p:
        raise ValueError("GES needs linearly independent sample columns. Remove constant or redundant variables.")
    cpdag0 = utils.dag_to_cpdag(initial) if initial is not None else np.zeros((p, p), dtype=int)
    steps = 0
    current = score.full_score(cpdag0) / n
    progress({"iteration": 0, "score": current, "best_score": current})

    def completion(adjacency):
        nonlocal steps
        completed = utils.pdag_to_cpdag(adjacency)
        steps += 1
        value = score.full_score(completed) / n
        progress({"iteration": steps, "score": value, "best_score": value})
        return completed

    cpdag, native_score = ges.fit(score, completion_algorithm=completion, A0=cpdag0,
                                 phases=settings.ges_phases, iterate=settings.ges_iterate)
    raw = utils.pdag_to_dag(cpdag).astype(int)
    # Every variable has its own error variance in ordinary Gaussian GES.
    return cpdag, [[i] for i in range(p)], native_score / n, raw, {
        "accepted_moves": steps, "standard_gaussian_bic": True,
        "native_bic": float(native_score), "consistent_dag_representative": True,
    }
