import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import MCMCfuncs
import utils


def main():
    random.seed(1)
    np.random.seed(1)
    no_colors = 3
    edge_prob = 0.8
    sample_size = 1000
    

    # RUN MCM
    num_chains = 50
    max_iters = 100_000_000  # Per-chain limit; raise if the target is not reached.
    max_num_nodes = 9  # 9 takes 40 min ---- 4min
    
    required_iters_min = []
    required_iters_mean = []
    required_iters_max = []

    for num_nodes in range(2, max_num_nodes):
        best_iters = []
        print(num_nodes)
        for _ in range(num_chains):
            real_partition, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(num_nodes, no_colors, edge_prob)
            real_edge_array = np.array(real_lambda_matrix != 0, dtype="int")
            samples = utils.generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)
            real_bic = utils.score_DAG(samples, real_edge_array, real_partition)

            _, _, _, diagnostics = MCMCfuncs.CausalMCMC(
                samples, num_iters=max_iters,
                target_bic=real_bic + np.log(0.95))
            if not diagnostics["target_reached"]:
                raise RuntimeError(
                    f"Target not reached for {num_nodes} nodes after {max_iters} iterations. "
                    "Increase max_iters; the capped run is not a convergence time.")
            best_iters.append(diagnostics["num_iters"])
        
        required_iters_min.append(np.min(best_iters))
        required_iters_mean.append(np.mean(best_iters))
        required_iters_max.append(np.max(best_iters))



    plt.plot(range(2,max_num_nodes), required_iters_min, color="C0", linewidth=0.5)
    plt.fill_between(range(2,max_num_nodes), required_iters_min, required_iters_max, color="C0", alpha=.2)
    plt.plot(range(2,max_num_nodes), required_iters_mean, color="C0", linestyle="-", marker="s", linewidth=1.5)
    plt.plot(range(2,max_num_nodes), required_iters_max, color="C0", linewidth=0.5)

    # A chain can meet the target at iteration zero.
    plt.yscale("symlog", linthresh=1)
    plt.xlabel("Nodes")
    plt.ylabel("Iterations")
    plt.title("PH-MCMC: Convergence Speed")
    plt.show()

    
if __name__ == "__main__":
    main()
