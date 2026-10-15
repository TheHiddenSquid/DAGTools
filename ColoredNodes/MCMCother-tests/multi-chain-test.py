import random
import sys
from pathlib import Path

import ges
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils
from MCMCfuncs import CausalMCMC


def main():
    random.seed(2)
    np.random.seed(2)
    no_nodes = 6
    no_colors = 3
    edge_prob = 0.6
    sample_size = 1000
    MCMC_iterations = 100_000
    no_chains = 3

    real_partition, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(no_nodes, no_colors, edge_prob)
    real_edge_array = np.array(real_lambda_matrix != 0, dtype="int")


    # Create plots
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3)
    plt.tight_layout()


    # Plot data generating graph
    plt.axes(ax1)
    G = nx.DiGraph(real_edge_array)
    nx.draw_circular(G, node_color=utils.generate_color_map(real_partition), with_labels=True)
    plt.title("Real DAG")


    # GES estimate of graph
    samples = utils.generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)
    res = ges.fit_bic(data=samples)
    GES_edge_array = res[0]

    plt.axes(ax2)
    G = nx.DiGraph(GES_edge_array)
    nx.draw_circular(G, with_labels=True)
    plt.title("GES CPDAG")
    


    # RUN MCMC

    chain_bics = []
    chain_cumsum = []


    best_edge_array = None
    best_partition = None
    best_bic = -np.inf
    best_iter = None
    best_chain = None


    for j in range(no_chains):
        initial_partition, initial_edge_array, _ = utils.generate_colored_DAG(no_nodes, no_nodes, 0.5)
        initial_edge_array = np.array(initial_edge_array != 0, dtype="int")
        edge_array, partition, bic, diagnostics = CausalMCMC(
            samples, num_iters=MCMC_iterations,
            A0=initial_edge_array, P0=initial_partition, return_history=True)
        bics = diagnostics["bic_history"]

        if bic > best_bic:
            best_edge_array = edge_array
            best_partition = partition
            best_bic = bic
            best_chain = j + 1
            # Index zero is the initial state; argmax gives the first best visit.
            best_iter = int(np.argmax(bics))

        chain_bics.append(bics)
        chain_cumsum.append(np.cumsum(bics))


    print(f"Ran MCMC for {no_chains * MCMC_iterations} iterations")
    print("Best BIC:", best_bic)
    print(f"Found in chain {best_chain}, iteration {best_iter} (0 = initial state)")
    real_cpdag = utils.getCPDAG(real_edge_array, real_partition)
    print("SHD to real CPDAG was:", utils.calc_SHD(best_edge_array, real_cpdag))
    print("Correct DAG and correct coloring gives BIC:", utils.score_DAG(samples, real_edge_array, real_partition))



    plt.axes(ax3)
    G = nx.DiGraph(best_edge_array)
    nx.draw_circular(G, node_color=utils.generate_color_map(best_partition), with_labels=True)
    plt.title("MCMC CPDAG")
    plt.show()


    plt.figure()
    plt.subplot(1,2,1)
    for i in range(no_chains):
        plt.plot(range(MCMC_iterations+1), chain_bics[i], label=f"Chain {i+1}")
    plt.xlabel("Iteration (0 = initial state)")
    plt.ylabel("BIC")
    plt.legend()

    plt.subplot(1,2,2)
    for i in range(no_chains):
        counts = np.arange(1, len(chain_cumsum[i]) + 1)
        plt.plot(counts - 1, chain_cumsum[i] / counts, label=f"Chain {i+1}")
    plt.xlabel("Iteration (0 = initial state)")
    plt.ylabel("Running mean BIC")
    plt.legend()
    plt.tight_layout()

    plt.show()

    
if __name__ == "__main__":
    main()
