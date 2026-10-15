import random
import sys

import gibbs
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")
import utils


def main():
    random.seed(1)
    np.random.seed(1)
    no_colors = 3
    edge_prob = 0.8
    sample_size = 1000
    

    # RUN MCM
    num_chains = 50
    max_num_nodes = 10  # 9 takes 40 min
    
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

            # MCMC setup
            gibbs.CausalMCMC_gibbs(samples, 0)
            A = np.zeros((num_nodes, num_nodes))
            P = [{i} for i in range(num_nodes)]
            current_bic, current_ML_data = gibbs.score_DAG_full(A, P)
            best_bic = current_bic    
  
            iters = 1
            while np.exp(best_bic - real_bic) <= 0.95:
                iters += 1
                A, P, current_bic, current_ML_data, _ = gibbs.MCMC_iteration(A, P, current_bic, current_ML_data)
                if current_bic > best_bic:
                    best_bic = current_bic
                
                if iters % 100_000 == 0:
                    print(np.exp(best_bic - real_bic))

    
            best_iters.append(iters)
        
        required_iters_min.append(np.min(best_iters))
        required_iters_mean.append(np.mean(best_iters))
        required_iters_max.append(np.max(best_iters))



    plt.semilogy(range(2,max_num_nodes), required_iters_min, color="C0", linewidth=0.5)
    plt.fill_between(range(2,max_num_nodes), required_iters_min, required_iters_max, color="C0", alpha=.2)
    plt.semilogy(range(2,max_num_nodes), required_iters_mean, color="C0", linestyle="-", marker="s", linewidth=1.5)
    plt.semilogy(range(2,max_num_nodes), required_iters_max, color="C0", linewidth=0.5)

    plt.xlabel("nodes")
    plt.ylabel("iterations")
    plt.title("Convergence speed")
    plt.show()

    
if __name__ == "__main__":
    main()
