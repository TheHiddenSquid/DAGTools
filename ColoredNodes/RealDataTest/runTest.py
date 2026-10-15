import random
import sys
import time

import ges
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.append("../")
import utils
from GreedyColorfuncs import CausalGreedyColor
from GreedyDPfuncs import CausalGreedyDP
from MCMCfuncs import CausalMCMC


def main():
    # Load data
    with open("winequality-red.csv") as f:
        data = []
        var_labels = f.readline().strip().split(",")
        for line in f:
            data.append(line.strip().split(","))
    
    samples = np.zeros((len(data), len(var_labels)))
    for i, row in enumerate(data):
        samples[i,:] =  [eval(x) for x in row]
    

    random.seed(1)
    np.random.seed(1)

    
    # Run estimations
    res = ges.fit_bic(data=samples)
    GES_edge_array = res[0]
    GESbic = res[1]/len(data)

    GCedge_array, GCpartition, GCbic = CausalGreedyColor(samples, num_starts=20)
    DPedge_array, DPpartition, DPbic = CausalGreedyDP(samples, num_starts=20)
    MCMCedge_array, MCMCpartition, MCMCbic = CausalMCMC(samples, 100_000_000)


    # Create plots
    fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4)
    
    plt.axes(ax1)
    G = nx.DiGraph(GES_edge_array)
    nx.draw_circular(G, with_labels=True)
    plt.title(f"GES: bic={GESbic:.4f}")
    
    plt.axes(ax2)
    G = nx.DiGraph(GCedge_array)
    nx.draw_circular(G, node_color=utils.generate_color_map(GCpartition), with_labels=True)
    plt.title(f"Greedy: bic={GCbic:.4f}")

    plt.axes(ax3)
    G = nx.DiGraph(DPedge_array)
    nx.draw_circular(G, node_color=utils.generate_color_map(DPpartition), with_labels=True)
    plt.title(f"GreedyDP: bic={DPbic:.4f}")

    plt.axes(ax4)
    G = nx.DiGraph(MCMCedge_array)
    nx.draw_circular(G, node_color=utils.generate_color_map(MCMCpartition), with_labels=True)
    plt.title(f"MCMC: bic={MCMCbic:.4f}")

    plt.tight_layout()
    plt.show()

    
if __name__ == "__main__":
    main()
