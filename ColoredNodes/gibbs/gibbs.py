import copy
import functools
import random
import sys
import time

import ges
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.append("../")
import utils

# Main MCMC functions

def CausalMCMC_gibbs(data, num_iters = None, move_weights = None, A0 = None, P0 = None, debug = False):

    #Clear cache for new run of algorithm
    calc_lstsq.cache_clear()

    # Setup global variables
    global my_data
    global num_nodes
    global num_samples
    global BIC_constant

    my_data = data.T
    num_nodes = data.shape[1]
    num_samples = data.shape[0]
    BIC_constant = np.log(num_samples)/(num_samples*2)


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

    

    
    # Check that wieghts are legal and prepare moves
    if move_weights is not None:
        if len(move_weights) != 2:
            raise ValueError("Lenght of weights must be 3")
        p_change_color, p_change_edge = move_weights
        if p_change_color<0 or p_change_edge<0 or not np.isclose(sum(move_weights),1):
            raise ValueError("Invalid move probabilities")
    else:
        move_weights = [0.4, 0.6]
    

    # Perform optional setup
    A = np.zeros((num_nodes, num_nodes))
    P = [{i} for i in range(num_nodes)]

    if P0 is not None:
        P = copy.deepcopy(P0)

    if A0 is not None:
        A = A0


    # Setup for iters
    bic, ML_data = score_DAG_full(A, P)


    best_A = A.copy()
    best_P = utils.sorted_partition(P)
    best_bic = bic
    best_iter = 0
    num_fails = 0

    # Run MCMC iters    
    for i in range(num_iters):
        A, P, bic, ML_data, fail = MCMC_iteration(A, P, bic, ML_data)    

        if bic > best_bic:
            best_A = A.copy()
            best_P = utils.sorted_partition(P)
            best_bic = bic
            best_iter = i
        num_fails += fail

    CPDAG_A = utils.getCPDAG(best_A, best_P)

    if debug: 
        return CPDAG_A, best_P, best_bic, best_iter, num_fails
    else:
        return CPDAG_A, best_P, best_bic
    
    
def MCMC_iteration(A, P, bic, ML_data):

    # Gibbs for edge moves
    current_bic = bic
    current_ML_data = copy.deepcopy(ML_data)
    for i in range(num_nodes):
        for j in range(num_nodes):
            edge = (i,j)
            # Skip all cases where the edge cannot be added (prior)
            if i == j:
                continue
            if A[edge] == 0:
                tmp = A.copy()
                tmp[edge] = 1
                if not utils.is_DAG(tmp):
                    continue

            # Do gibbs update
            tmp = A.copy()
            tmp[edge] = 1 - tmp[edge]
            pot_bic, pot_ML_data = score_DAG_edge_edit(tmp, P, current_ML_data, edge)

            amp_power = 1 #4
            choice = random.choices([0,1], k=1, weights=[np.exp(current_bic)**amp_power, np.exp(pot_bic)**amp_power])[0]
            

            if choice == 1:
                A = tmp
                current_bic = pot_bic
                current_ML_data = pot_ML_data


    # Metropolis hastings for color moves
    P, node, old_color, new_color = change_partiton(P)
    pot_bic, pot_ML_data = score_DAG_color_edit(A, P, current_ML_data, old_color, new_color)
        

    # Metropolis algorithm to accept or reject new colored DAG
    if random.random() <= np.exp(pot_bic - current_bic):
        new_bic = pot_bic
        new_ML_data = pot_ML_data
        failed = 0
    else:
        P[new_color].remove(node)
        P[old_color].add(node)  
        new_bic = current_bic
        new_ML_data = current_ML_data
        failed = 1

    
    return A, P, new_bic, new_ML_data, failed



# For moves
def change_partiton(P):
    node_to_change = random.randrange(num_nodes)
    old_color = None
    other_colors = []

    for i, part in enumerate(P):
        if node_to_change in part:
            old_color = i
        elif len(part) != 0:
            other_colors.append(i)
        else:
            empty_color = i

    if len(P[old_color]) != 1:
        other_colors.append(empty_color)

    P[old_color].remove(node_to_change)
    new_color = random.choice(other_colors)
    P[new_color].add(node_to_change)

    return P, node_to_change, old_color, new_color

def change_edge(A):
    did_change = False
    n1 = random.randrange(num_nodes)
    n2 = random.randrange(num_nodes)
    while n2 == n1:
        n2 = random.randrange(num_nodes)
    edge = (n1, n2)

    if A[edge] == 1:
        A[edge] = 0
        did_change = True
    else:
        A[edge] = 1
        if utils.is_DAG(A):
            did_change = True
        else:
            A[edge] = 0
    
    return A, edge, did_change



# For DAG heuristic
def score_DAG_full(A, P):

    # Calculate ML-eval via least sqares
    omegas_ML = [0] * num_nodes
    for node in range(num_nodes):
        parents = utils.get_parents(node, A)
        _, ss_res = calc_lstsq(node, tuple(parents))
        omegas_ML[node] = ss_res / num_samples


    # Calculate decomposed BIC
    bic_decomp = [0] * num_nodes
    for i, block in enumerate(P):
        if len(block) == 0:
            continue
        tot = 0
        for node in block:
            tot += omegas_ML[node]
        block_omega = tot / len(block)

        bic_decomp[i] = -len(block) * (np.log(block_omega) + 1)
    
    # Calculate full BIC
    bic = sum(bic_decomp) / 2
    bic -= BIC_constant * (sum(1 for part in P if len(part)>0) + np.count_nonzero(A))
    
    return bic, [omegas_ML, bic_decomp]

def score_DAG_color_edit(A, P, ML_data, old_color, new_color):
    
    # ML data is the same
    omegas_ML, bic_decomp = ML_data
    omegas_ML = omegas_ML.copy()
    bic_decomp = bic_decomp.copy()

    # Update decomposed BIC
    for block_index in [old_color, new_color]:
        block = P[block_index]
        if len(block) == 0:
             bic_decomp[block_index] = 0
             continue
        tot = 0
        for node in block:
            tot += omegas_ML[node]
        block_omega = tot / len(block)

        bic_decomp[block_index] = -len(block) * (np.log(block_omega) + 1)

    # Calculate full BIC
    bic = sum(bic_decomp) / 2
    bic -= BIC_constant * (sum(1 for part in P if len(part)>0) + np.count_nonzero(A))

    return bic, [omegas_ML, bic_decomp]

def score_DAG_edge_edit(A, P, ML_data, changed_edge):

    # Get old ML-eval
    omegas_ML, bic_decomp = ML_data
    omegas_ML = omegas_ML.copy()
    bic_decomp = bic_decomp.copy()

    
    # Update ML-eval
    _, active_node = changed_edge
    parents = utils.get_parents(active_node, A)
    _, ss_res = calc_lstsq(active_node, tuple(parents))
    omegas_ML[active_node] = ss_res / num_samples

   
    # Update decomposed BIC
    for i, block in enumerate(P):
        if active_node in block:
            tot = 0
            for node in block:
                tot += omegas_ML[node]
            omega = tot / len(block)

            bic_decomp[i] = -len(block) * (np.log(omega) + 1)


    # Calculate full BIC
    bic = sum(bic_decomp) / 2
    bic -= BIC_constant * (sum(1 for part in P if len(part)>0) + np.count_nonzero(A))

    return bic, [omegas_ML, bic_decomp]


@functools.cache
def calc_lstsq(node, parents):
    a = my_data[parents,:]
    b = my_data[node,:]
    beta = np.linalg.solve(a @ a.T, a @ b)
    x = b - a.T @ beta
    ss_res = np.dot(x,x)
    return beta, ss_res




def main():
    random.seed(2)
    np.random.seed(2)
    no_nodes = 8
    no_colors = 4
    edge_prob = 0.6
    sample_size = 1000
    MCMC_iterations = 10_000

    real_partition, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(no_nodes, no_colors, edge_prob)
    real_edge_array = np.array(real_lambda_matrix != 0, dtype=np.int64)

    # Create plots
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3)
    plt.tight_layout()


    # Plot data generating graph
    plt.axes(ax1)
    G = nx.DiGraph(real_edge_array)
    nx.draw_circular(G, node_color=utils.generate_color_map(real_partition), with_labels=True)
    plt.title("Real DAG")


    # GES estimate of graph
    t = time.perf_counter()
    samples = utils.generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)
    res = ges.fit_bic(data=samples)
    GES_edge_array = res[0]
    print("sample-gen+GES", time.perf_counter()-t)

    plt.axes(ax2)
    G = nx.DiGraph(GES_edge_array)
    nx.draw_circular(G, with_labels=True)
    plt.title("GES CPDAG")
    

    t = time.perf_counter()
    edge_array, partition, bic, found_iter, _ = CausalMCMC_gibbs(samples, MCMC_iterations, debug=True)

    print("Result of running causalMCMC")
    print(f"It took {time.perf_counter()-t} seconds")
    print("Found DAG with BIC:", bic)
    print("Found on iteration:", found_iter)
    print("MCMC: SHD to real DAG:", utils.calc_SHD(edge_array, real_edge_array))
    print("GES: SHD to real DAG:", utils.calc_SHD(GES_edge_array, real_edge_array))
    print("The found DAG with correct coloring gives BIC:", utils.score_DAG(samples, edge_array, real_partition))
    print("Correct DAG and correct coloring gives BIC:", real_bic := utils.score_DAG(samples, real_edge_array, real_partition))
    print("Error was", np.exp(bic - real_bic))



    plt.axes(ax3)
    G = nx.DiGraph(edge_array)
    nx.draw_circular(G, node_color=utils.generate_color_map(partition), with_labels=True)
    plt.title("MCMC")


    plt.show()



if __name__ == "__main__":
    main()
