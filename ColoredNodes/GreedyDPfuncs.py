import functools
import math
import random
import time
from collections.abc import Iterable

import networkx as nx
import numpy as np
if __package__:
    from . import utils
    from .utils import get_optimal_partition, optimal_partition_core
else:
    import utils
    from utils import get_optimal_partition, optimal_partition_core
from numba import njit

"""
Greedy solver that uses the optimal partition for every graph and thus only needs to search through the space of 
DAGs during the optimization. The drawback is O(p^2) extra computation each iteration.
"""

# Main functions

def CausalGreedyDP(samples, num_starts = 5, moves = None, *, center=True,
                   A0=None, return_dag=False, on_progress=None):
    """Find a colored graph, centering each column by default.

    Set center=False for already-centered samples or an intentionally fixed
    zero mean. The input array is never modified or standardized.
    """

    #Clear cache for new run of algorithm
    calc_lstsq_S.cache_clear()
    
    # Setup global variables
    global S
    global num_edges
    global num_nodes
    global BIC_constant
    global used_moves
    global size_term

    if moves == None:
        used_moves = ["add_edge", "remove_edge", "flip_edge"]
    else:
        used_moves = moves

    num_nodes = samples.shape[1]
    num_samples = samples.shape[0]
    if center:
        samples = samples - samples.mean(axis=0)
    S = (1/num_samples) * samples.T @ samples
    BIC_constant = math.log(num_samples)/(num_samples*2)
    size_term = np.array([0.0] + [m * (math.log(m) - 1) for m in range(1, num_nodes + 1)])


    # Setup iterations
    best_A = np.zeros((num_nodes, num_nodes)) if A0 is None else np.array(A0, dtype=np.int64, copy=True)
    num_edges = np.count_nonzero(best_A)
    best_bic, best_ss_res, _ = score_DAG_full(best_A)

    edge_probs = list(np.linspace(0,1,num_starts))
    num_colors = [int(x) for x in np.linspace(1,num_nodes,num_starts)]

    visited = set()
    total_steps = 0
    if on_progress:
        on_progress({"iteration": 0, "restart": 0, "total": num_starts, "score": float(best_bic), "best_score": float(best_bic)})

    
    # Perform iterations
    for i in range(num_starts):
        _, lambda_matrix, _ = utils.generate_colored_DAG(num_nodes, num_colors[i], edge_probs[i])
        A = np.array(lambda_matrix != 0, dtype=np.int64)
        if A0 is not None and i == 0:
            A = np.array(A0, dtype=np.int64, copy=True)

        num_edges = np.count_nonzero(A)
        bic, ss_res, _ = score_DAG_full(A)
        if bic > best_bic:
            best_A, best_ss_res, best_bic = A.copy(), ss_res.copy(), bic
        sorted_edges = get_sorted_edges(A)
        done = False

        while not done:
            key = A.tobytes()
            if key in visited:
                break   # This continuation was already explored by an earlier start.
            visited.add(key)

            A, bic, ss_res, sorted_edges, done = iteration(A, bic, ss_res, sorted_edges)
            if bic > best_bic:
                best_A = A.copy()
                best_ss_res = ss_res.copy()
                best_bic = bic
            total_steps += 1
            if on_progress:
                on_progress({"iteration": total_steps, "restart": i + 1, "total": num_starts, "score": float(bic), "best_score": float(best_bic)})

    # Ectract optimal partition
    best_P, _ = get_optimal_partition(best_ss_res, BIC_constant, size_term)
    CPDAG_A = utils.getCPDAG(best_A, best_P)
    if on_progress:
        on_progress({"iteration": total_steps, "restart": num_starts, "total": num_starts, "score": float(bic), "best_score": float(best_bic)})
    if return_dag:
        return CPDAG_A, best_P, best_bic, best_A.copy()
    return CPDAG_A, best_P, best_bic
    
def iteration(A, bic, ss_res, sorted_edges):
    global num_edges

    best_move = None
    best_A = None
    best_bic = bic
    best_updates = None
    edges_in_DAG, edges_giving_DAGs, _ = sorted_edges
    parent_masks = get_parent_masks(A)



    # Check all potential edge adds
    if "add_edge" in used_moves:
        for edge in edges_giving_DAGs:
            A[edge] = 1
            num_edges += 1
            potential_bic, potential_updates = score_DAG_edge_edit(ss_res, [edge], [parent_masks[edge[1]] | (1 << int(edge[0]))])

            if is_improvement(potential_bic, best_bic):
                best_A = A.copy()
                best_bic = potential_bic
                best_updates = potential_updates
                best_move = "add_edge"
                best_saved_edge = edge
        
            A[edge] = 0
            num_edges -= 1


    # Check all potential edge removals
    if "remove_edge" in used_moves:
        for edge in edges_in_DAG:
            A[edge] = 0
            num_edges -= 1
            
            potential_bic, potential_updates = score_DAG_edge_edit(ss_res, [edge], [parent_masks[edge[1]] & ~ (1 << int(edge[0]))])

            if is_improvement(potential_bic, best_bic):
                best_A = A.copy()
                best_bic = potential_bic
                best_updates = potential_updates
                best_move = "remove_edge"
                best_saved_edge = edge
            
            A[edge] = 1
            num_edges += 1


    # Check all potential edge flips
    if "flip_edge" in used_moves:
        for edge in edges_in_DAG:
            rev = (edge[1], edge[0])
            A[edge] = 0
            A[rev] = 1
        
            if utils.is_DAG(A):
                potential_bic, potential_updates = score_DAG_edge_edit(
                    ss_res, [edge, rev],
                    [parent_masks[edge[1]] & ~(1 << int(edge[0])),
                     parent_masks[rev[1]] | (1 << int(rev[0]))])

                if is_improvement(potential_bic, best_bic):
                    best_A = A.copy()
                    best_bic = potential_bic
                    best_updates = potential_updates
                    best_move = "flip_edge"
                    best_saved_edge = [edge, rev]
                        
            A[edge] = 1
            A[rev] = 0


    # Do the best possible jump

    if best_updates is None:
        return A, bic, ss_res, sorted_edges, True
    
    else:
        new_bic = best_bic 
        new_ss_res = ss_res
        for node, residual in best_updates:
            new_ss_res[node] = residual
        new_A = best_A

        if best_move == "add_edge":
            num_edges += 1
            new_sorted_edges = update_sorted_edges_ADD(new_A, *sorted_edges, best_saved_edge)
        elif best_move == "remove_edge":
            num_edges -= 1
            new_sorted_edges = update_sorted_edges_REMOVE(new_A, *sorted_edges, best_saved_edge)
        elif best_move == "flip_edge":
            removed_edge, added_edge = best_saved_edge
            intermediate_A = new_A.copy()
            intermediate_A[added_edge] = 0
            intermediate_edges = update_sorted_edges_REMOVE(intermediate_A, *sorted_edges, removed_edge)
            new_sorted_edges = update_sorted_edges_ADD(new_A, *intermediate_edges, added_edge)
            
        
    return new_A, new_bic, new_ss_res, new_sorted_edges, False



# For edge lookups

def get_sorted_edges(A):
  
    tmp_edge_array = A.copy()
    n = np.shape(tmp_edge_array)[0]

    edges_in_DAG = []
    edges_giving_DAGs = []
    edges_not_giving_DAGs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if tmp_edge_array[i, j] == 1:
                edges_in_DAG.append((i,j))
                continue

            tmp_edge_array[i, j] = 1
            if utils.is_DAG(tmp_edge_array):
                edges_giving_DAGs.append((i,j))
            else:
                edges_not_giving_DAGs.append((i,j))
            tmp_edge_array[i,j] = 0

    return [edges_in_DAG, edges_giving_DAGs, edges_not_giving_DAGs]

def update_sorted_edges_REMOVE(A, edges_in, addable_edges, not_addable_edges, removed_edge):
    
    tmp_edge_array = A.copy()

    edges_in_DAG = edges_in.copy()
    edges_in_DAG.remove(removed_edge)

    edges_giving_DAGs = addable_edges.copy() + [removed_edge]
    edges_not_giving_DAGs = []

    for edge in not_addable_edges:
        tmp_edge_array[edge] = 1
        if utils.is_DAG(tmp_edge_array):
            edges_giving_DAGs.append(edge)
        else:
            edges_not_giving_DAGs.append(edge)
        tmp_edge_array[edge] = 0

    return [edges_in_DAG, edges_giving_DAGs, edges_not_giving_DAGs]

def update_sorted_edges_ADD(A, edges_in, addable_edges, not_addable_edges, added_edge):

    tmp_edge_array = A.copy()

    edges_in_DAG = edges_in.copy() + [added_edge]
    edges_giving_DAGs = []
    edges_not_giving_DAGs = not_addable_edges.copy()

    for edge in addable_edges:
        if edge == added_edge:
            continue

        tmp_edge_array[edge] = 1
        if utils.is_DAG(tmp_edge_array):
            edges_giving_DAGs.append(edge)
        else:
            edges_not_giving_DAGs.append(edge)
        tmp_edge_array[edge] = 0

    return [edges_in_DAG, edges_giving_DAGs, edges_not_giving_DAGs]



# For DAG heuristic
def score_DAG_full(A):
    ss_res = get_ss_res(A, range(num_nodes))
    P, score = get_optimal_partition(ss_res, BIC_constant, size_term)
    bic = -score - BIC_constant * np.count_nonzero(A)
    return bic, ss_res, P


def score_DAG_edge_edit(ss_res, changed_edges, proposed_masks):
    """Return the candidate BIC and residual updates, leaving ss_res unchanged.

    Re-optimize the partition because an edge edit can change every block.
    Only the changed residual entries are temporarily substituted; the winning
    candidate's updates are committed by iteration().
    """
    updates = []
    old_values = []
    try:
        for edge, parent_mask in zip(changed_edges, proposed_masks):
            node = edge[1]
            residual = calc_lstsq_S(node, parent_mask)
            old_values.append((node, ss_res[node]))
            ss_res[node] = residual
            updates.append((node, residual))
        _, score = get_optimal_partition(ss_res, BIC_constant, size_term)
    finally:
        for node, residual in reversed(old_values):
            ss_res[node] = residual
    bic = -score - BIC_constant * num_edges
    return bic, updates


def get_ss_res(A, nodes):
    parent_masks = get_parent_masks(A)
    if not isinstance(nodes, Iterable):
        return calc_lstsq_S(nodes, parent_masks[nodes])
    return [calc_lstsq_S(node, parent_masks[node]) for node in nodes]


def get_parent_masks(A):
    """Build cache keys once for a graph without changing its matrix representation."""
    if len(A) <= 64:
        return _parent_masks_numba(A).tolist()
    return [sum(1 << int(parent) for parent in utils.get_parents(node, A))
            for node in range(len(A))]

@njit(cache=True)
def _parent_masks_numba(A):
    masks = np.zeros(len(A), dtype=np.uint64)
    for node in range(len(A)):
        for parent in range(len(A)):
            if A[parent, node] == 1:
                masks[node] |= np.uint64(1) << np.uint64(parent)
    return masks

@functools.cache
def calc_lstsq_S(node, parent_mask):
    """Cache residuals; decode 64-bit parent masks inside Numba on a miss."""
    if parent_mask <= 0xFFFFFFFFFFFFFFFF:
        return calc_lstsq_mask_numba(node, np.uint64(parent_mask), S)

    # Preserve support for Python masks with parent indices above 63.
    parents = []
    while parent_mask:
        bit = parent_mask & -parent_mask
        parents.append(bit.bit_length() - 1)
        parent_mask ^= bit
    return calc_lstsq_S_numba(node, np.array(parents, dtype=np.int64), S)

@njit(cache=True)
def calc_lstsq_mask_numba(node, parent_mask, G):
    """Decode an unsigned 64-bit mask in ascending order, then score its parents."""
    if parent_mask == 0:
        return G[node, node]

    parents = np.empty(G.shape[0], dtype=np.int64)
    count = 0
    index = 0
    while parent_mask:
        if parent_mask & np.uint64(1):
            parents[count] = index
            count += 1
        parent_mask >>= np.uint64(1)
        index += 1
    return calc_lstsq_S_numba(node, parents[:count], G)

@njit(cache=True)
def calc_lstsq_S_numba(node, parents, G):
    """Compute residual variance using Cholesky factorization.
    """
    k = len(parents)
    if k == 0:
        return G[node, node]

    L = np.empty((k, k))
    y = np.empty(k)
    explained = 0.0
    for i in range(k):
        # Build the next row of L directly from G.
        for j in range(i + 1):
            value = G[parents[i], parents[j]]
            for s in range(j):
                value -= L[i, s] * L[j, s]
            if i == j:
                if not value > 0.0:
                    raise np.linalg.LinAlgError("Parent covariance matrix is not positive definite")
                L[i, j] = math.sqrt(value)
            else:
                L[i, j] = value / L[j, j]

        # Forward substitution needs only the rows built so far.
        value = G[parents[i], node]
        for j in range(i):
            value -= L[i, j] * y[j]
        y[i] = value / L[i, i]
        explained += y[i] * y[i]

    return G[node, node] - explained




def is_improvement(candidate, current):
    BIC_TOL = 1e-12
    return candidate > current + BIC_TOL * max(1.0, abs(current))



def main():
    import matplotlib.pyplot as plt
    #random.seed(2)
    #np.random.seed(2)
    no_nodes = 6
    no_colors = 3
    edge_prob = 0.6
    sample_size = 1000
    num_starts = 8

    real_partition, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(no_nodes, no_colors, edge_prob)
    real_edge_array = np.array(real_lambda_matrix != 0, dtype=np.int64)


    # Create plots
    fig, ((ax1, ax2,)) = plt.subplots(1, 2)
    plt.tight_layout()


    # Plot data generating graph
    plt.axes(ax1)
    G = nx.DiGraph(real_edge_array)
    nx.draw_circular(G, node_color=utils.generate_color_map(real_partition), with_labels=True)
    plt.title("Real DAG")


    # GES estimate of graph
    samples = utils.generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)


    t = time.perf_counter()
    edge_array, partition, bic = CausalGreedyDP(samples, num_starts)
    plt.axes(ax2)
    G = nx.DiGraph(edge_array)
    nx.draw_circular(G, node_color=utils.generate_color_map(partition), with_labels=True)
    plt.title("Greedy")
    

    print(f"Ran Hybrid with {num_starts} starts")
    print(f"It took {time.perf_counter()-t} seconds")
    print("Found DAG with BIC:", bic)
    print("Greedy: SHD to real DAG was:", utils.calc_SHD(edge_array, real_edge_array))
    print("Correct DAG and correct coloring gives BIC:", utils.score_DAG(samples, real_edge_array, real_partition))


   

    plt.show()

        
    



if __name__ == "__main__":
    main()

