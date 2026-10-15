import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils
from MCMCfuncs import CausalMCMC


def main():
    random.seed(1)
    np.random.seed(1)
    num_nodes = 6
    num_colors = 3
    sparse = True
    sample_size = 1000


    # RUN MCM
    MCMC_iterations = 100_000
    num_chains = 50
    param_list = [0.001, 0.05, 0.2, 0.5, 0.8, 0.95, 0.999]
    num_params = len(param_list)
    
    chain_rolling_best_bic = [[] for _ in range(num_params)]

    for i in range(num_chains):
        print(i)
        real_partition, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(num_nodes, num_colors, sparse)
        real_edge_array = np.array(real_lambda_matrix != 0, dtype="int")

        samples = utils.generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)
        real_bic = utils.score_DAG(samples, real_edge_array, real_partition)

        start_partition, start_edge_array, _ = utils.generate_colored_DAG(num_nodes, num_nodes, 0.5)
        start_edge_array = np.array(start_edge_array != 0, dtype="int")

        for j in range(num_params):
            print(j)
            
            params = [1-param_list[j], param_list[j]]

            _, _, _, diagnostics = CausalMCMC(
                samples, num_iters=MCMC_iterations, move_weights=params,
                A0=start_edge_array, P0=start_partition, return_history=True)
            rolling_best_bic = real_bic - np.maximum.accumulate(diagnostics["bic_history"])

            chain_rolling_best_bic[j].append(rolling_best_bic)


 
    param_chains = []
    for group in chain_rolling_best_bic:
        param_chains.append(np.mean(group, axis=0))


    for k in range(num_params):
        plt.plot(range(MCMC_iterations+1), param_chains[k], label = rf"$\pi$ = {param_list[k]}")

    # Include the initial state at iteration zero on the logarithmic plot.
    plt.xscale("symlog", linthresh=1)
    plt.xlabel("iterations")
    plt.ylabel("true bic - best bic")
    plt.legend()
    plt.title(f"Convergence speed ({num_nodes} nodes)")
    plt.show()

    
if __name__ == "__main__":
    main()
