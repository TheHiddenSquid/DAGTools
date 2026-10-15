import random
import sys
import time

import ges
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.append("../")
import utils
from MCMCfuncs import CausalMCMC


def main():
    random.seed(1)
    np.random.seed(1)
    no_nodes = 4
    no_colors = 2
    density = 0.6
    sample_size = 1000
    MCMC_iterations = 100_000

    real_partition, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(no_nodes, no_colors, density)
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
    edge_array, partition, visits = CausalMCMC(samples, MCMC_iterations, mode = "map")


    print(f"Ran MCMC for {MCMC_iterations} iterations")
    print(f"It took {time.perf_counter()-t} seconds")
    print("SHD to real DAG was:", utils.calc_SHD(edge_array, real_edge_array))
    print(f"it was visited {visits} of {MCMC_iterations} iterations")


    plt.axes(ax3)
    G = nx.DiGraph(edge_array)
    nx.draw_circular(G, node_color=utils.generate_color_map(partition), with_labels=True)
    plt.title("MCMC")


    plt.show()



    
if __name__ == "__main__":
    main()
