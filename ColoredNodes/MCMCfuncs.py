import functools
import math
import time
from collections import defaultdict

import numpy as np
if __package__:
    from . import utils
else:
    import utils
from numba import njit

# Main MCMC functions

def CausalMCMC(data, num_iters = None, mode = "bic", move_weights = None, A0 = None, P0 = None, debug = False,
               *, return_history=False, target_bic=None, chunk_size=10_000, center=True, refine_partition=True,
               return_dag=False, on_progress=None):
    """
    Run MCMC to find the colored DAG that best fits the data. By default,
    center each column without modifying the input array.

    Parameters
    ----------
    data : numpy.ndarray
        The n x p array containing the observations,
        where columns correspond to variables and rows to observations.
    num_iters : int, optional
        Maximum number of MCMC iterations. Default is 25*5^min(p, 8).
        A target_bic can stop the run earlier.
    mode : [{'bic', 'map'}*], optional
        Option on what DAG the algorithm will return. If 'bic' it returns the DAG with the highest bic-score.
        If 'map' it returns the most visited DAG. Default is 'bic'.
    move_weights : [int, int], optional
        List of probability of changing a color versuis changing an edge. First value is color.
        Default is [0.4, 0.6]
    A0 : numpy.ndarray, optional
        The initial DAG on which the algorithm will run, where where `A0[i,j]
        != 0` implies the edge `i -> j`.
    P0 : [{int}*] optional
        The initial coloring partition on which the algorithm will run.
    debug : bool, optional
        If true, returns extra values, such as number exact iteration best DAG was found and the number of failed jumps.

    return_history : bool, optional
        Record the initial BIC and the chain BIC after every attempted iteration,
        including rejected and unchanged proposals. Default is False.
    target_bic : float, optional
        Stop when best_bic >= target_bic, including at the initial state.
        Supported only in mode='bic'. num_iters remains the iteration limit.

    chunk_size : int, optional
        Maximum iterations per batch of move and acceptance draws (default
        10_000). Node draws refill independently in batches of up to
        3*chunk_size values. Also sets the partition manager's node batch size.
        Buffers are bounded by num_iters for shorter runs. Changing chunk_size
        changes the seeded trajectory because random draws are interleaved
        differently; the proposal and acceptance distributions are unchanged.

    center : bool, optional
        Subtract each column's sample mean before scoring (default True).
        Set False for already-centered data or an intentionally fixed zero mean.
        The caller's data is never modified; variables are not standardized.

    refine_partition : bool, optional
        Optimize the saved best DAG's partition with DP before returning
        (default True). Ignored in mode='map'. Set False to preserve the sampled
        partition, e.g. when intentionally holding colors fixed. Refinement
        does not change bic_history, best_iter, or target-based stopping.

    Returns
    -------
    estimate_edges : numpy.ndarray
        The adjacency matrix of the estimated DAG.
    estimate_partition : [[int]*]
        The estimated node-partition.
    total_score : float
        The score of the estimate (visit count in mode='map').
    diagnostics : dict, optional
        Appended after the existing return values (including debug values) when
        return_history=True or target_bic is supplied. Contains 'num_iters'
        (completed iterations; initial state is iteration zero), 'target_reached'
        (None when no target was requested), and 'bic_history' (a float64 array
        of length num_iters+1, or None when recording is disabled).
        In BIC mode, 'search_best_bic' is the best score before refinement and
        'partition_refinement_gain' is the improvement after sampling; both
        are None in MAP mode. The returned score may exceed max(bic_history).
        Existing debug best_iter retains its legacy zero-based loop index and
        refers to the sampled best state, not a visit to the refined model.

    Raises
    ------
    TypeError:
        If the type of some of the parameters was not expected,
        e.g. if data is not a numpy array.
    ValueError:
        If the value of some of the parameters is not appropriat.
    """

    #Clear cache for new run of algorithm
    calc_lstsq.cache_clear()


    # Setup global constants
    global num_nodes
    global BIC_constant
    global S
    global size_term

    num_nodes = data.shape[1]
    num_samples = data.shape[0]
    if center:
        data = data - data.mean(axis=0)
    S = (1/num_samples) * data.T @ data
    BIC_constant = math.log(num_samples)/(num_samples*2)
    # For a block of size m and residual sum s, its contribution is
    # size_term[m] - m * log(s). Empty blocks contribute zero.
    size_term = [0.0] + [m * (math.log(m) - 1) for m in range(1, num_nodes + 1)]


    # Calculate number of iterations
    if num_iters is None:
        num_iters = 25 * 5**num_nodes
        if num_nodes > 8:
            print("Warning! Default number of iterations on more than 8 variables is too big. Set to 25*5^8")
            num_iters = 25 * 5**8
    elif not isinstance(num_iters, int):
        raise TypeError("num_iters needs to be an int")
    elif num_iters < 0:
        raise ValueError("num_iters needs to be positive")

    
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise TypeError("chunk_size needs to be an int")
    if chunk_size <= 0:
        raise ValueError("chunk_size needs to be positive")

    # Check that mode is legal
    if mode not in ["bic", "map"]:
        raise ValueError("Mode not supported")
    
    
    if target_bic is not None:
        if mode != "bic":
            raise ValueError("target_bic is only supported in mode='bic'")
        target_bic = float(target_bic)
        if not math.isfinite(target_bic):
            raise ValueError("target_bic must be finite")

    # Check that wieghts are legal and prepare moves
    if move_weights is not None:
        if len(move_weights) != 2:
            raise ValueError("Lenght of weights must be 3")
        p_change_color, p_change_edge = move_weights
        if p_change_color<0 or p_change_edge<0 or not np.isclose(sum(move_weights),1):
            raise ValueError("Invalid move probabilities")
    else:
        move_weights = [0.4, 0.6]
    
    # Generate move/acceptance batches only when the loop needs them.
    chunk_start = chunk_end = 0
    node_batch_size = min(chunk_size, max(1, num_iters))
    global random_nodes, random_node_chunk_size
    random_nodes = []
    random_node_chunk_size = 3 * node_batch_size

    # Setup initial guesses
    global num_edges


    P = utils.PartitionManager(n=num_nodes, chunk_size=node_batch_size)
    if P0 is not None:
        blocks = [set(block) for block in P0 if len(block)]
        nodes = [node for block in blocks for node in block]
        if len(nodes) != num_nodes or set(nodes) != set(range(num_nodes)):
            raise ValueError("P0 must partition all node indices exactly once")
        # Merge initial singleton blocks using the manager's own bookkeeping.
        for block in blocks:
            destination = min(block)
            for node in sorted(block - {destination}):
                P.perform_move(node, node, destination)
                P.commit_move(node, node, destination)
 

    # Bit i in parent_masks[j] represents the edge i -> j. Use Python ints
    # so shifts remain valid for node indices of 63 and above.
    if A0 is None:
        parent_masks = [0] * num_nodes
    else:
        parent_masks = [
            sum(1 << int(parent) for parent in utils.get_parents(node, A0))
            for node in range(num_nodes)
        ]
    num_edges = sum(mask.bit_count() for mask in parent_masks)


    # Setup for iters
    bic, ML_data = score_DAG_full(parent_masks, P)
    bic_history = np.empty(num_iters + 1) if return_history else None
    if return_history:
        bic_history[0] = bic
    completed_iters = 0
    target_reached = None if target_bic is None else bic >= target_bic

    search_best_bic = None
    partition_refinement_gain = None

    def finish(result):
        if return_dag:
            result = (*result, best_A.copy())
        if return_history or target_bic is not None:
            # Release unused capacity when a target stops the chain early.
            history = bic_history
            if return_history and completed_iters < num_iters:
                history = bic_history[:completed_iters + 1].copy()
            return (*result, {"num_iters": completed_iters,
                              "target_reached": target_reached,
                              "bic_history": history,
                              "search_best_bic": search_best_bic,
                              "partition_refinement_gain": partition_refinement_gain})
        return result

    report_every = max(1, min(1000, num_iters // 100))
    def report(step, current, best):
        if on_progress:
            on_progress({"iteration": step, "total": num_iters,
                         "score": float(current), "best_score": float(best)})
    report(0, bic, bic)


    if mode == "bic":
        best_masks = parent_masks.copy()
        best_P = utils.sorted_partition(P.P)
        best_bic = bic
        best_iter = 0
        num_fails = 0

        # Run MCMC iters
        for i in range(0 if target_reached else num_iters):
            if i == chunk_end:
                batch_size = min(chunk_size, num_iters - i)
                moves = (np.random.random(batch_size) < move_weights[1]).tolist()
                random_floats = np.log(np.random.random(batch_size)).tolist()
                chunk_start = i
                chunk_end = i + batch_size
            batch_index = i - chunk_start
            move = moves[batch_index]
            accept_float = random_floats[batch_index]
            bic, fail = MCMC_iteration(move, parent_masks, P, bic, ML_data, accept_float)
            if bic > best_bic:
                best_masks = parent_masks.copy()
                best_P = utils.sorted_partition(P.P)
                best_bic = bic
                best_iter = i
            num_fails += fail
            if return_history:
                bic_history[i + 1] = bic
            if (i + 1) % report_every == 0 or i + 1 == num_iters:
                report(i + 1, bic, best_bic)
            if target_bic is not None and best_bic >= target_bic:
                target_reached = True
                completed_iters = i + 1
                break
        else:
            completed_iters = 0 if target_reached else num_iters

        search_best_bic = best_bic
        if refine_partition:
            residuals = [calc_lstsq(node, mask) for node, mask in enumerate(best_masks)]
            refined_P, cost = utils.get_optimal_partition(residuals, BIC_constant)
            refined_bic = -cost - BIC_constant * sum(mask.bit_count() for mask in best_masks)
       
            if refined_bic > best_bic:
                best_P, best_bic = refined_P, refined_bic
        partition_refinement_gain = best_bic - search_best_bic
        report(completed_iters, bic, best_bic)

        best_A = utils.parent_masks_to_adjacency(best_masks)
        CPDAG_A = utils.getCPDAG(best_A, best_P)
        if debug: 
            return finish((CPDAG_A, best_P, best_bic, best_iter, num_fails))
        else:
            return finish((CPDAG_A, best_P, best_bic))
    

    if mode == "map":
        cashe = defaultdict(lambda: 0)
        num_fails = 0

        chain_best = bic

        # Run MCMC iters    
        for i in range(num_iters):
            if i == chunk_end:
                batch_size = min(chunk_size, num_iters - i)
                moves = (np.random.random(batch_size) < move_weights[1]).tolist()
                random_floats = np.log(np.random.random(batch_size)).tolist()
                chunk_start = i
                chunk_end = i + batch_size
            batch_index = i - chunk_start
            move = moves[batch_index]
            accept_float = random_floats[batch_index]
            bic, fail = MCMC_iteration(move, parent_masks, P, bic, ML_data, accept_float)
            graph_key = (tuple(parent_masks), tuple(tuple(block) for block in utils.sorted_partition(P.P)))
            cashe[graph_key] += 1
            num_fails += fail
            if return_history:
                bic_history[i + 1] = bic
            chain_best = max(chain_best, bic)
            if (i + 1) % report_every == 0 or i + 1 == num_iters:
                report(i + 1, bic, chain_best)
        completed_iters = num_iters

        # With no transitions, return the initial graph with zero visits.
        if not cashe:
            graph_key = (tuple(parent_masks), tuple(tuple(block) for block in utils.sorted_partition(P.P)))
            cashe[graph_key] = 0
        most_visited = max(cashe, key=cashe.get)
        num_visits = cashe[most_visited]

        best_A = utils.parent_masks_to_adjacency(most_visited[0])

        best_P = [set(x) for x in most_visited[1]]

        CPDAG_A = utils.getCPDAG(best_A, best_P)
        if debug: 
            return finish((CPDAG_A, best_P, num_visits, num_fails))
        else:
            return finish((CPDAG_A, best_P, num_visits))
    
def MCMC_iteration(move, parent_masks, P, bic, ML_data, accept_float):
    """Update ML_data in place only on acceptance; snapshots need their own copies."""
    global num_edges

    # Create new colored DAG based on move
    if move:
        edge, did_change, edge_added = change_edge(parent_masks)
        if did_change:
            delta_bic, proposed_update = score_DAG_edge_edit(parent_masks, P, ML_data, edge, edge_added)
        else:
            return bic, 0   
    else:
        node, old_color, new_color = P.propose_move()
        P.perform_move(node, old_color, new_color)
        delta_bic, proposed_update = score_DAG_color_edit(P, ML_data, node, old_color, new_color)
     

    # Metropolis algorithm to accept or reject new colored DAG
    if accept_float <= delta_bic:
        omegas_ML, block_sums, bic_decomp = ML_data
        if move:
            new_omega, new_sum, new_decomp = proposed_update
            active_node = edge[1]
            active_block = P.node_to_block[active_node]
            omegas_ML[active_node] = new_omega
            block_sums[active_block] = new_sum
            bic_decomp[active_block] = new_decomp
        else:
            old_sum, new_sum, old_decomp, new_decomp = proposed_update
            block_sums[old_color] = old_sum
            block_sums[new_color] = new_sum
            bic_decomp[old_color] = old_decomp
            bic_decomp[new_color] = new_decomp
            P.commit_move(node, old_color, new_color)

        new_bic = bic + delta_bic
        failed = 0
    else:
        if move:
            parent_masks[edge[1]] ^= 1 << int(edge[0])
            num_edges += -1 if edge_added else 1
        else:
            P.undo_move(node, old_color, new_color)
            
        new_bic = bic
        failed = 1

    return new_bic, failed


# For edge moves
def change_edge(parent_masks):
    """Propose an edge toggle in place; leave cyclic additions unchanged."""
    global num_edges

    did_change = False
    edge_added = None
    # Repeated equal-node draws can consume more than three values per move.
    # Refill independently of the iteration batches, including during retries.
    if len(random_nodes) < 2:
        random_nodes.extend(np.random.randint(0, num_nodes, random_node_chunk_size).tolist())
    n1 = random_nodes.pop()
    n2 = random_nodes.pop()
    while n2 == n1:
        if not random_nodes:
            random_nodes.extend(np.random.randint(0, num_nodes, random_node_chunk_size).tolist())
        n2 = random_nodes.pop()
    edge = (n1, n2)

    parent_bit = 1 << int(n1)
    if parent_masks[n2] & parent_bit:
        parent_masks[n2] ^= parent_bit
        did_change = True
        num_edges -= 1
        edge_added = 0
    elif not utils.is_reachable_masks(parent_masks, n1, n2):
        parent_masks[n2] ^= parent_bit
        did_change = True
        num_edges += 1
        edge_added = 1
    
    return edge, did_change, edge_added


# For DAG heuristic
def score_DAG_full(parent_masks, P):
    """Return the absolute BIC and [omegas_ML, block_sums, bic_decomp]."""

    # Calculate ML-eval via least sqares
    omegas_ML = [calc_lstsq(node, parent_masks[node]) for node in range(num_nodes)]

    # Calculate decomposed BIC
    bic_decomp = [0] * num_nodes
    block_sums = [0] * num_nodes

    for i, block in enumerate(P.P):
        if len(block) == 0:
            continue
        tot = 0
        for node in block:
            tot += omegas_ML[node]
        block_sums[i] = tot
        block_size = len(block)
        bic_decomp[i] = size_term[block_size] - block_size * math.log(tot)
    
    # Calculate full BIC
    bic = sum(bic_decomp)/2 - BIC_constant * (num_edges + len(P.active_blocks))
    
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
    old_size = len(P.P[old_color])
    new_size = len(P.P[new_color])
    
    if old_size != 0:
        proposed_old_sum = block_sums[old_color] - omegas_ML[node]
        proposed_old_decomp = size_term[old_size] - old_size * math.log(proposed_old_sum)
    else:
        proposed_old_sum = 0
        proposed_old_decomp = 0

    proposed_new_sum = block_sums[new_color] + omegas_ML[node]
    proposed_new_decomp = size_term[new_size] - new_size * math.log(proposed_new_sum)


    # Sizes reflect the proposal, but active_blocks is updated only on commit.
    delta_colors = int(new_size == 1) - int(old_size == 0)
    delta_decomp = (proposed_old_decomp - old_decomp) + (proposed_new_decomp - new_decomp)
    delta_bic = delta_decomp/2 - BIC_constant * delta_colors
    
    return delta_bic, (proposed_old_sum, proposed_new_sum, proposed_old_decomp, proposed_new_decomp)

def score_DAG_edge_edit(parent_masks, P, ML_data, changed_edge, edge_added):
    """Score an edge after change_edge() updates parent_masks.
    Does not modify ML_data.
    Return (delta_bic, (new_omega, new_sum, new_decomp)) for the affected node/block.
    """

    # Get old ML-eval
    omegas_ML, block_sums, bic_decomp = ML_data

    
    # Update ML-eval
    _, active_node = changed_edge
    ss_res = calc_lstsq(active_node, parent_masks[active_node])
    old_omega_ML = omegas_ML[active_node]
    new_omega_ML = ss_res


    # Update decomposed BIC
    active_block_id = P.node_to_block[active_node]

    old_decomp = bic_decomp[active_block_id]
    block_size = len(P.P[active_block_id])
    new_sum = block_sums[active_block_id] - old_omega_ML + new_omega_ML
    new_decomp = size_term[block_size] - block_size * math.log(new_sum)
  

    # Choose the signed penalty directly to avoid NumPy-scalar arithmetic.
    edge_penalty = BIC_constant if edge_added else -BIC_constant
    delta_bic = (new_decomp - old_decomp)/2 - edge_penalty

    return delta_bic, (new_omega_ML, new_sum, new_decomp)


@functools.cache
def calc_lstsq(node, parent_mask):
    """Cache residuals; decode 64-bit parent masks inside Numba on a miss."""
    if parent_mask <= 0xFFFFFFFFFFFFFFFF:
        return calc_lstsq_mask_numba(node, np.uint64(parent_mask), S)

    # Preserve support for Python masks with parent indices above 63.
    parents = []
    while parent_mask:
        bit = parent_mask & -parent_mask
        parents.append(bit.bit_length() - 1)
        parent_mask ^= bit
    return calc_lstsq_numba(node, np.array(parents, dtype=np.int64), S)

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
    return calc_lstsq_numba(node, parents[:count], G)

@njit(cache=True)
def calc_lstsq_numba(node, parents, G):
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


def main():
    pass



if __name__ == "__main__":
    main()
