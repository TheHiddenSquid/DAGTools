import copy
import functools
import math

import numpy as np
if __package__:
    from . import utils
else:
    import utils
from numba import njit

"""
Greedy solver that searches through the space of all clored DAGs (G, P) - pairs during the optimization. 
This is fast per iteratation, but the search space grows extremely quick with p.
"""

# Main functions

def CausalGreedyColor(samples, num_starts = 5, moves = None, *, center=True, refine_partition=True,
                      A0=None, P0=None, return_dag=False, on_progress=None):
    """Find a colored graph, centering each column by default.

    Set center=False for already-centered samples or an intentionally fixed
    zero mean. The input array is never modified or standardized.
    By default, optimize the partition of each restart's final DAG with DP
    before selecting the overall winner. Set refine_partition=False to retain
    the local-search coloring, including when intentionally fixing colors.
    Refinement does not resume the graph search.
    Optional A0/P0 seed the first restart; P0 is used for every restart when
    supplied. Disable color moves and refinement to hold that partition fixed.
    return_dag appends the representative DAG to the legacy return tuple.
    on_progress receives iteration/restart/score dictionaries.
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
        used_moves = ["change_color", "add_edge", "remove_edge", "flip_edge"]
    else:
        used_moves = moves

    num_samples, num_nodes = samples.shape
    if center:
        samples = samples - samples.mean(axis=0)
    S = (1/num_samples) * samples.T @ samples
    BIC_constant = math.log(num_samples)/(num_samples*2)
    size_term = [0.0] + [m * (math.log(m) - 1) for m in range(1, num_nodes + 1)]


    # Setup iterations
    best_A = np.zeros((num_nodes, num_nodes)) if A0 is None else np.array(A0, dtype=np.int64, copy=True)
    num_edges = np.count_nonzero(best_A)
    best_P = [{i} for i in range(num_nodes)] if P0 is None else [set(block) for block in P0 if block]
    best_P += [set() for _ in range(num_nodes-len(best_P))]
    best_bic, initial_ML_data = score_DAG_full(best_A, best_P)
    if refine_partition:
        refined_P, cost = utils.get_optimal_partition(initial_ML_data[0], BIC_constant)
        refined_bic = -cost - BIC_constant * num_edges
        if refined_bic > best_bic:
            best_P, best_bic = refined_P, refined_bic

    edge_probs = list(np.linspace(0,1,num_starts))
    num_colors = [int(x) for x in np.linspace(1,num_nodes,num_starts)]

    
    total_steps = 0
    if on_progress:
        on_progress({"iteration": 0, "restart": 0, "total": num_starts, "score": float(best_bic), "best_score": float(best_bic)})
    # Perform iterations
    for i in range(num_starts):
        P, lambda_matrix, _ = utils.generate_colored_DAG(num_nodes, num_colors[i], edge_probs[i])
        A = np.array(lambda_matrix != 0, dtype=np.int64)
        if A0 is not None and i == 0:
            A = np.array(A0, dtype=np.int64, copy=True)
        if P0 is not None:
            P = [set(block) for block in P0 if block]
        for _ in range(num_nodes-len(P)):
            P.append(set())

        num_edges = np.count_nonzero(A)
        bic, ML_data = score_DAG_full(A, P)
        if bic > best_bic:
            best_A, best_P, best_bic = A.copy(), utils.sorted_partition(P), bic
        sorted_edges = get_sorted_edges(A)
        done = False

        while not done:
            A, P, bic, ML_data, sorted_edges, done = iteration(A, P, bic, ML_data, sorted_edges)
            if bic > best_bic:
                best_A = A.copy()
                best_P = utils.sorted_partition(P)
                best_bic = bic
            total_steps += 1
            if on_progress:
                on_progress({"iteration": total_steps, "restart": i + 1, "total": num_starts, "score": float(bic), "best_score": float(best_bic)})

        if refine_partition:
            # The terminal graph's residuals are already current in ML_data.
            refined_P, cost = utils.get_optimal_partition(ML_data[0], BIC_constant)
            refined_bic = -cost - BIC_constant * int(np.count_nonzero(A))
            if refined_bic > best_bic:
                best_A = A.copy()
                best_P = refined_P
                best_bic = refined_bic

    CPDAG_A = utils.getCPDAG(best_A, best_P)
    if on_progress:
        on_progress({"iteration": total_steps, "restart": num_starts, "total": num_starts, "score": float(bic), "best_score": float(best_bic)})
    if return_dag:
        return CPDAG_A, best_P, best_bic, best_A.copy()
    return CPDAG_A, best_P, best_bic
    
def iteration(A, P, bic, ML_data, sorted_edges):
    global num_edges

    best_move = None
    best_A = None
    best_P = None
    best_bic = bic
    best_update = None
    edges_in_DAG, edges_giving_DAGs, _ = sorted_edges
    parent_masks = get_parent_masks(A)


    # Check all neighboring colorings
    if "change_color" in used_moves:
        for node in range(num_nodes):
            old_color = None
            other_colors = []

            for i, part in enumerate(P):
                if node in part:
                    old_color = i
                elif len(part) != 0:
                    other_colors.append(i)
                else:
                    empty_color = i

            if len(P[old_color]) != 1:
                other_colors.append(empty_color)

            P[old_color].remove(node)
            for new_color in other_colors:
                P[new_color].add(node)

                delta_bic, potential_update = score_DAG_color_edit(P, ML_data, node, old_color, new_color)
                potential_bic = bic + delta_bic

                if is_improvement(potential_bic, best_bic):
                    best_P = copy.deepcopy(P)
                    best_bic = potential_bic
                    best_update = potential_update
                    best_move = "change_color"
                    best_colors = (old_color, new_color)


                P[new_color].remove(node)
            P[old_color].add(node)


    # Check all potential edge adds
    if "add_edge" in used_moves:
        for edge in edges_giving_DAGs:
            A[edge] = 1
            num_edges += 1
            proposed_mask = parent_masks[edge[1]] | (1 << int(edge[0]))
            delta_bic, potential_update = score_DAG_edge_edit(proposed_mask, P, ML_data, edge, edge_added=True)
            potential_bic = bic + delta_bic

            if is_improvement(potential_bic, best_bic):
                best_A = A.copy()
                best_bic = potential_bic
                best_update = (potential_update,)
                best_move = "add_edge"
                best_saved_edge = edge
        
            A[edge] = 0
            num_edges -= 1


    # Check all potential edge removals
    if "remove_edge" in used_moves:
        for edge in edges_in_DAG:
            A[edge] = 0
            num_edges -= 1
            
            proposed_mask = parent_masks[edge[1]] & ~ (1 << int(edge[0]))
            delta_bic, potential_update = score_DAG_edge_edit(proposed_mask, P, ML_data, edge, edge_added=False)
            potential_bic = bic + delta_bic

            if is_improvement(potential_bic, best_bic):
                best_A = A.copy()
                best_bic = potential_bic
                best_update = (potential_update,)
                best_move = "remove_edge"
                best_saved_edge = edge
            
            A[edge] = 1
            num_edges += 1


    if "flip_edge" in used_moves:
        for edge in edges_in_DAG:
            rev = (edge[1], edge[0])
            A[edge] = 0
            A[rev] = 1
        
            if utils.is_DAG(A):
                removed_mask = parent_masks[edge[1]] & ~(1 << int(edge[0]))
                added_mask = parent_masks[rev[1]] | (1 << int(rev[0]))
                first_delta, first_update = score_DAG_edge_edit(removed_mask, P, ML_data, edge, edge_added=False)
                node, block, _, _, _ = first_update
                undo_update = (node, block, ML_data[0][node], ML_data[1][block], ML_data[2][block])
                # The second endpoint may share the first endpoint's color.
                # Temporarily apply the first update, then restore the base state.
                _apply_edge_update(ML_data, first_update)
                try:
                    second_delta, second_update = score_DAG_edge_edit(added_mask, P, ML_data, rev, edge_added=True)
                finally:
                    _apply_edge_update(ML_data, undo_update)
                potential_bic = bic + first_delta + second_delta
                potential_update = (first_update, second_update)
    
                if is_improvement(potential_bic, best_bic):
                    best_A = A.copy()
                    best_bic = potential_bic
                    best_update = potential_update
                    best_move = "flip_edge"
                    best_saved_edge = [edge, rev]
                        
            A[edge] = 1
            A[rev] = 0



    # Do the best possible jump

    if best_update is None:
        return A, P, bic, ML_data, sorted_edges, True
    
    else:
        new_bic = best_bic 
        new_ML_data = ML_data
        if best_move == "change_color":
            old_color, new_color = best_colors
            old_sum, new_sum, old_decomp, new_decomp = best_update
            ML_data[1][old_color] = old_sum
            ML_data[1][new_color] = new_sum
            ML_data[2][old_color] = old_decomp
            ML_data[2][new_color] = new_decomp
        else:
            for update in best_update:
                _apply_edge_update(ML_data, update)
        
        if best_move == "change_color":
            new_A = A
            new_P = best_P
            new_sorted_edges = sorted_edges
        elif best_move == "add_edge":
            new_A = best_A
            new_P = P
            num_edges += 1
            new_sorted_edges = update_sorted_edges_ADD(new_A, sorted_edges[0], sorted_edges[1], sorted_edges[2], best_saved_edge)
        elif best_move == "remove_edge":
            new_A = best_A
            new_P = P
            num_edges -= 1
            new_sorted_edges = update_sorted_edges_REMOVE(new_A, sorted_edges[0], sorted_edges[1], sorted_edges[2], best_saved_edge)
        elif best_move == "flip_edge":
            new_A = best_A
            new_P = P
            removed_edge, added_edge = best_saved_edge
            tmp = new_A.copy()
            tmp[added_edge] = 0
            tmp_edges = update_sorted_edges_REMOVE(tmp, *sorted_edges, removed_edge)
            new_sorted_edges = update_sorted_edges_ADD(new_A, *tmp_edges, added_edge)

        
    return new_A, new_P, new_bic, new_ML_data, new_sorted_edges, False



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
def score_DAG_full(A, P):
    """Return the absolute BIC and [omegas_ML, block_sums, bic_decomp]."""

    parent_masks = get_parent_masks(A)

    # Calculate ML-eval via least sqares
    omegas_ML = [calc_lstsq_S(node, parent_masks[node]) for node in range(num_nodes)]

    # Calculate decomposed BIC
    bic_decomp = [0] * num_nodes
    block_sums = [0] * num_nodes

    for i, block in enumerate(P):
        if len(block) == 0:
            continue
        tot = 0
        for node in block:
            tot += omegas_ML[node]
        block_sums[i] = tot
        block_size = len(block)
        bic_decomp[i] = size_term[block_size] - block_size * math.log(tot)
    
    # Calculate full BIC
    bic = sum(bic_decomp)/2 - BIC_constant * (np.count_nonzero(A) + sum(bool(block) for block in P))
    
    return bic, [omegas_ML, block_sums, bic_decomp]

def score_DAG_color_edit(P, ML_data, node, old_color, new_color):
    """Score a performed color move without modifying ML_data.
    Return (delta_bic, (old_sum, new_sum, old_decomp, new_decomp)),
    where all four values describe the proposed blocks after the move.
    """
    
    # ML data is the same
    omegas_ML, block_sums, bic_decomp = ML_data

    
    # Update decomposed BIC
    old_decomp = bic_decomp[old_color]
    new_decomp = bic_decomp[new_color]
    old_size = len(P[old_color])
    new_size = len(P[new_color])
    
    if old_size != 0:
        proposed_old_sum = block_sums[old_color] - omegas_ML[node]
        proposed_old_decomp = size_term[old_size] - old_size * math.log(proposed_old_sum)
    else:
        proposed_old_sum = 0
        proposed_old_decomp = 0

    proposed_new_sum = block_sums[new_color] + omegas_ML[node]
    proposed_new_decomp = size_term[new_size] - new_size * math.log(proposed_new_sum)


    # P already reflects the proposed move; no partition scan is needed.
    delta_colors = int(new_size == 1) - int(old_size == 0)
    delta_decomp = (proposed_old_decomp - old_decomp) + (proposed_new_decomp - new_decomp)
    delta_bic = delta_decomp/2 - BIC_constant * delta_colors
    
    return delta_bic, (proposed_old_sum, proposed_new_sum, proposed_old_decomp, proposed_new_decomp)

def score_DAG_edge_edit(parent_mask, P, ML_data, changed_edge, edge_added):
    """Return a BIC delta and an unapplied update for one changed node.
    parent_mask describes the proposed parent set for the changed node.
    """
    # Get old ML-eval
    omegas_ML, block_sums, bic_decomp = ML_data

    
    # Update ML-eval
    _, active_node = changed_edge
    ss_res = calc_lstsq_S(active_node, parent_mask)
    old_omega_ML = omegas_ML[active_node]
    new_omega_ML = ss_res


    # Update decomposed BIC
    active_block_id = next(i for i, block in enumerate(P) if active_node in block)

    old_decomp = bic_decomp[active_block_id]
    block_size = len(P[active_block_id])
    new_sum = block_sums[active_block_id] - old_omega_ML + new_omega_ML
    new_decomp = size_term[block_size] - block_size * math.log(new_sum)
  

    # Choose the signed penalty directly to avoid NumPy-scalar arithmetic.
    edge_penalty = BIC_constant if edge_added else -BIC_constant
    delta_bic = (new_decomp - old_decomp)/2 - edge_penalty

    return delta_bic, (active_node, active_block_id, new_omega_ML, new_sum, new_decomp)


def _apply_edge_update(ML_data, update):
    node, block, omega, block_sum, decomp = update
    ML_data[0][node] = omega
    ML_data[1][block] = block_sum
    ML_data[2][block] = decomp


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
    pass



if __name__ == "__main__":
    main()

