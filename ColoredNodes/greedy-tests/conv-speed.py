import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")
import utils
from GreedyColorfuncs import CausalGreedyColor


def main():
    random.seed(1)
    np.random.seed(1)
    no_colors = 3
    edge_prob = 0.8
    sample_size = 1000
    

    # RUN MCM
    num_chains = 50 # 50
    max_num_nodes = 21  # 21 takes 30 min
    

    min_time = []
    mean_time = []
    max_time = []


    for num_nodes in range(2, max_num_nodes):
        best_times = []
        print(num_nodes)
        for _ in range(num_chains):
            real_partition, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(num_nodes, no_colors, edge_prob)
            samples = utils.generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)

            t = time.perf_counter()
            CausalGreedyColor(samples, num_starts=5)      

            best_times.append(time.perf_counter()-t)
        
        min_time.append(np.min(best_times))
        mean_time.append(np.mean(best_times))
        max_time.append(np.max(best_times))


    plt.semilogy(range(2,max_num_nodes), min_time, color="C0", linewidth=0.5)
    plt.fill_between(range(2,max_num_nodes), min_time, max_time, color="C0", alpha=.2)
    plt.semilogy(range(2,max_num_nodes), mean_time, color="C0",linestyle="-", marker="s", linewidth=1.5)
    plt.semilogy(range(2,max_num_nodes), max_time, color="C0", linewidth=0.5)


    plt.xlabel("Nodes")
    plt.ylabel("Time (s)")
    plt.title("GPHES: Convergence Speed")
    plt.show()

    
if __name__ == "__main__":
    main()
