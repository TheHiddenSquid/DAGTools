import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils
from MCMCfuncs import CausalMCMC


def main():
    random.seed(3)
    np.random.seed(3)
    no_nodes = 8
    no_colors = 4
    edge_prob = 0.6
    sample_size = 1000
    MCMC_iterations = 100_000

    real_partition, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(no_nodes, no_colors, edge_prob)
    real_edge_array = np.array(real_lambda_matrix != 0, dtype="int")


    # Create plots
    fig, ((ax1, ax2)) = plt.subplots(1, 2)
    plt.tight_layout()


    # Plot data generating graph
    plt.axes(ax1)
    G = nx.DiGraph(real_edge_array)
    nx.draw_circular(G, node_color=utils.generate_color_map(real_partition), with_labels=True)
    plt.title("Real DAG")


    # GES estimate of graph
    samples = utils.generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)
    
    

    t = time.perf_counter()
    edge_array, partition, bic = CausalMCMC(samples, MCMC_iterations, move_weights = [1,0], A0 = real_edge_array)


    print("MCMC given the correct edges")
    print(f"Ran MCMC for {MCMC_iterations} iterations")
    print(f"It took {time.perf_counter()-t} seconds")
    print("Found DAG with BIC:", bic)
    print("CHD to real DAG was:", utils.calc_CHD(partition, real_partition))
    print("Correct DAG and correct coloring gives BIC:", utils.score_DAG(samples, real_edge_array, real_partition))


    plt.axes(ax2)
    G = nx.DiGraph(edge_array)
    nx.draw_circular(G, node_color=utils.generate_color_map(partition), with_labels=True)
    plt.title("MCMC")


    plt.show()



    
if __name__ == "__main__":
    main()
