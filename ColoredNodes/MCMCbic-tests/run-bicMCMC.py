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
    random.seed(2)
    np.random.seed(2)
    no_nodes = 4
    no_colors = 3
    edge_prob = 0.9
    sample_size = 1000
    MCMC_iterations = 1_000_000

    # Generate model + sample
    real_partition, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(no_nodes, no_colors, edge_prob)
    real_edge_array = np.array(real_lambda_matrix != 0, dtype=np.int64)
    samples = utils.generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)


    # Run estimates of graph
    trueDAG_bic = utils.score_DAG(samples, real_edge_array, real_partition)

    res = ges.fit_bic(data=samples)
    GES_edge_array = res[0]
    GESbic = res[1]/sample_size

    t = time.perf_counter()
    MCMCedge_array, MCMCpartition, bic, found_iter, _ = CausalMCMC(samples, MCMC_iterations, debug=True)
    taken_time = time.perf_counter()-t

    print("Result of running causalMCMC")
    print(f"It took {taken_time} seconds")
    print("Found DAG with BIC:", bic)
    print("Found on iteration:", found_iter)
    print("MCMC: SHD to real DAG:", utils.calc_SHD(MCMCedge_array, real_edge_array))
    print("GES: SHD to real DAG:", utils.calc_SHD(GES_edge_array, real_edge_array))
    print("Correct DAG and correct coloring gives BIC:", trueDAG_bic)


    # Create plots
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3)
        
    plt.axes(ax1)
    G = nx.DiGraph(real_edge_array)
    nx.draw_circular(G, node_color=utils.generate_color_map(real_partition), with_labels=True)
    plt.title(f"Real DAG: bic={trueDAG_bic:.4f}")

    plt.axes(ax2)
    G = nx.DiGraph(GES_edge_array)
    nx.draw_circular(G, with_labels=True)
    plt.title(f"GES: bic={GESbic:.4f}")

    plt.axes(ax3)
    G = nx.DiGraph(MCMCedge_array)
    nx.draw_circular(G, node_color=utils.generate_color_map(MCMCpartition), with_labels=True)
    plt.title(f"MCMC: bic={bic:.4f}")

    plt.tight_layout()
    plt.show()

    
if __name__ == "__main__":
    main()
