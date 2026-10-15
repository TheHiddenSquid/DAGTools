import random
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.append("../")
import MCMCfuncs
import utils


def main():
    np.random.seed(5)
    random.seed(5)
    no_nodes = 4
    no_colors = 4
    edge_prob = 0.6
    sample_size = 100

    real_partition, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(no_nodes, no_colors, edge_prob)
    
    real_partition = [[0],[1],[2],[3]]
    real_lambda_matrix = np.array([[0,1,1,0],
                                   [0,0,0,1],
                                   [0,0,0,1],
                                   [0,0,0,0]])
    real_omega_matrix = np.array([1,3,2,1])
    real_edge_array = np.array(real_lambda_matrix != 0, dtype="int")

    global samples
    samples = utils.generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)


    # Create plots
    fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2)
    plt.tight_layout()


    # Plot data generating graph
    plt.axes(ax11)
    G = nx.DiGraph(real_lambda_matrix)
    nx.draw_circular(G, node_color=utils.generate_color_map(real_partition), with_labels=True)
    plt.title("Real DAG")



    # Start with random DAG
    initial_partition, initial_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(no_nodes, no_nodes, 0.5)
    initial_edge_array = np.array(initial_lambda_matrix != 0, dtype="int")
    


    # More MCMC setup
    global current_edge_array
    global current_partition
    global current_bic
    global current_ML_data

    current_edge_array = initial_edge_array.copy()
    current_partition = [[0],[1],[2],[3]]
    MCMCfuncs.CausalMCMC(samples, 0)
    current_bic, current_ML_data = MCMCfuncs.score_DAG_full(current_edge_array, current_partition)

    plt.axes(ax12)
    G = nx.DiGraph(current_edge_array)
    nx.draw_circular(G, node_color=utils.generate_color_map(current_partition), with_labels=True)
    plt.title("MCMC")


    # Setop for BIC plots
    global bics
    global cumsum
    start_bic = current_bic
    bics = [start_bic]
    cumsum = [start_bic]

    global SHDs
    SHDs = [utils.calc_SHD(current_edge_array, real_edge_array)]

    global CHDs
    CHDs = [utils.calc_CHD(current_partition, real_partition)]



    def update(frame):
        # Update graph visuals
        plt.axes(ax12)
        ax12.clear()
        global current_edge_array
        global current_partition
        global current_ML_data
        global current_bic

        move = random.choices([0, 1], k=1, weights=[0.4,0.6])[0]
        move = 1
        current_edge_array, current_partition, current_bic, current_ML_data, _ = MCMCfuncs.MCMC_iteration(move, current_edge_array, current_partition, current_bic, current_ML_data)
        
        G = nx.DiGraph(current_edge_array)
        nx.draw_circular(G, node_color=utils.generate_color_map(current_partition), with_labels=True)
        plt.title("MCMC")

        # Update bic graphics
        global bics
        global cumsum
        global SHDs
        global CHDs

        bics.append(current_bic)
        plt.axes(ax21)
        ax21.clear()
        plt.plot(range(len(bics)),bics)
        plt.xlabel("iterations")
        plt.ylabel("BIC")
        plt.title("BIC")

        plt.axes(ax22)
        ax22.clear()
        cumsum.append(cumsum[-1]+current_bic)

        plt.plot(range(len(bics)),[cumsum[i]/(i+1) for i in range(len(bics))])
        plt.xlabel("iterations")
        plt.ylabel("BIC")
        plt.title("Rolling Average")


    ani = animation.FuncAnimation(fig=fig, func=update, frames=1000, interval=10)
    #ani.save(filename="MCMC_anim.gif", writer="pillow")
    plt.show()



if __name__ == "__main__":
    main()
