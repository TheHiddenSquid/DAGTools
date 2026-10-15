import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import MCMCfuncs
import utils


def test2d(resolution, sample_size, MCMCiterations):
    no_nodes = 6
    no_colors = 3
    sparse = True

    _, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(no_nodes, no_colors, sparse)
    samples = utils.generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)

    
    # RUN MCMC
    # Fully random colored DAG
    y = np.zeros((resolution+1))

    for i in range(1,resolution):
        random.seed(2)
        np.random.seed(2)
        s = i/resolution
        print(s)
        _, _, _, _, fails = MCMCfuncs.CausalMCMC(samples, MCMCiterations, move_weights=[1-s,s], debug=True)
        y[i] = fails/MCMCiterations
        

    # Plot data
    end = y.shape[0]

    x = np.linspace(0,1,end)

    plt.plot(x[1:end-1],y[1:end-1])
    plt.xlabel(r"$\pi$")
    plt.ylabel(r"% fails")
    plt.title(f"Metropolis fails: 6 nodes")
    plt.show()


def main():
    random.seed(2)
    np.random.seed(2)
    
    test2d(resolution = 40, sample_size = 100, MCMCiterations = 50_000)


    
if __name__ == "__main__":
    main()
    