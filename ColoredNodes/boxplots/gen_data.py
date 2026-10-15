import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import random
import sys
import time
from multiprocessing import Pool

import ges
import numpy as np
import pandas as pd

sys.path.append("../")
import GreedyColorfuncs
import GreedyDPfuncs
import MCMCfuncs
import utils


def get_data_df(num):
    size_options = [5,8,20]
    edge_probabilities = [0.2,0.4,0.6,0.8]
    sample_options = [100, 500, 1000]

    color_options = ["2", "num_nodes/2"]

    i = 0
    t_start = time.perf_counter()
    rows = []

    for num_nodes in size_options: 
        for edge_prob in edge_probabilities:
            for nc in color_options:
                nc_used = int(eval(nc))
                i += 1
                random.seed(int(f"{num}0{i}"))
                np.random.seed(int(f"{num}0{i}"))
                real_partition, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(num_nodes, nc_used, edge_prob)
                real_edge_array = np.array(real_lambda_matrix != 0, dtype=np.int64)
                for num_samples in sample_options:
                    samples = utils.generate_sample(num_samples, real_lambda_matrix, real_omega_matrix)

                    # GES estimate of graph
                    # res = ges.fit_bic(data=samples)
                    # GES_edge_array = res[0]
                    # GES_SHD = utils.calc_SHD(real_edge_array, GES_edge_array)
                    # rows.append([num_nodes, nc_used, edge_prob, num_samples, "GES", GES_SHD, None])
                    
                    # MCMC_BIC estimate of graph
                    # MCMC_edge_array, MCMC_partition, _ = MCMCfuncs.CausalMCMC(samples)
                    # MCMC_SHD = utils.calc_SHD(real_edge_array, MCMC_edge_array)
                    # MCMC_CHD = utils.calc_CHD(real_partition, MCMC_partition)
                    # rows.append([num_nodes, nc_used, edge_prob, num_samples, "MCMC_BIC", MCMC_SHD, MCMC_CHD])

                    # Greedy estimate of graph
                    # greedy_edge_array, greedy_partition, greedy_bic = GreedyColorfuncs.CausalGreedyColor(samples, num_starts=5)
                    # greedy_SHD = utils.calc_SHD(real_edge_array, greedy_edge_array)
                    # greedy_CHD = utils.calc_CHD(real_partition, greedy_partition)
                    # rows.append([num_nodes, nc_used, edge_prob, num_samples, "GreedyColor+turn", greedy_SHD, greedy_CHD])

                    # New Greedy estimate of graph
                    greedyDP_edge_array, greedyDP_partition, greedyDP_bic = GreedyDPfuncs.CausalGreedyDP(samples, num_starts=5)
                    greedyPD_SHD = utils.calc_SHD(real_edge_array, greedyDP_edge_array)
                    greedyPD_CHD = utils.calc_CHD(real_partition, greedyDP_partition)
                    rows.append([num_nodes, nc_used, edge_prob, num_samples, "GreedyDP+turn", greedyPD_SHD, greedyPD_CHD])


    t_end = time.perf_counter()
    df = pd.DataFrame(reversed(rows), columns=["num_nodes", "num_colors", "edge_prob", "num_samples", "Algorithm", "SHD", "CHD"])

    return num, df, t_end-t_start


def main():
    random.seed(1)
    np.random.seed(1)

    num_tests = 40
  
    dfs = []
    print("Start")
    t_start = time.perf_counter()
    with Pool(8) as pool:
        result = pool.imap_unordered(get_data_df, [x for x in range(num_tests)])
        for num, df, duration in result:
            print(f"{num}, took, {duration} s")
            #df.to_csv(f"df{num}out.csv", index=False)
            dfs.append(df)
    final_df = pd.concat(dfs)


    t_end = time.perf_counter()
    print(f"All done in {t_end-t_start} s")
    final_df.to_csv("out_new.csv", index=False)


if __name__ == "__main__":
    main()
