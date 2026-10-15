import random
import sys
import time
from pathlib import Path

import ges
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils
from MCMCfuncs import CausalMCMC


def main():
    no_nodes = 5
    no_colors = 3
    edge_prob = 0.6
    sample_size = 1000
    MCMC_iterations = 100_000

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
    

    t = time.perf_counter()
    edge_array, partition, bic = CausalMCMC(samples, MCMC_iterations, move_weights = [0, 1], P0=real_partition)


    print("MCMC given the correct colors")
    print(f"Ran MCMC for {MCMC_iterations} iterations")
    print(f"It took {time.perf_counter()-t} seconds")
    print("Found DAG with BIC:", bic)
    print("SHD to real DAG was:", utils.calc_SHD(edge_array, real_edge_array))
    print("Correct DAG and correct coloring gives BIC:", utils.score_DAG(samples, real_edge_array, real_partition))


    plt.axes(ax3)
    G = nx.DiGraph(edge_array)
    nx.draw_circular(G, node_color=utils.generate_color_map(partition), with_labels=True)
    plt.title("MCMC")


    plt.show()



    
if __name__ == "__main__":
    main()
