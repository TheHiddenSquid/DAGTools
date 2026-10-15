import random
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.append("../")
from collections import defaultdict

import utils
from GreedyColorfuncs import (
    CausalGreedyColor,
    get_sorted_edges,
    iteration,
    score_DAG,
)


def get_all_4node_DAGs():
    AllDAGs = defaultdict(lambda: 0)
    cap = 543
    #cap = 300

    while len(AllDAGs) < cap:
        print(len(AllDAGs))
        A = utils.get_random_DAG(4)
        h1 = A.tobytes()
        AllDAGs[h1] += 1

    dags = list(AllDAGs.keys())
    dags.sort()
    return dags


def main():
    random.seed(8)
    np.random.seed(8)
    # General setup
    num_nodes = 4
    num_colors = 3
    sparse = True
    sample_size = 1000

    real_partition, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(num_nodes, num_colors, sparse)
    real_A = np.array(real_lambda_matrix != 0, dtype="int")
    samples = utils.generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)


    # setup for hypergraph
    global dags
    dags = get_all_4node_DAGs()

    edge_array = np.zeros((543,543), dtype="int")
    for i, DAG in enumerate(dags):
        A = np.reshape(np.frombuffer(DAG, dtype="int"), (4,4))
        
        edges_in, edges_giving, _ = get_sorted_edges(A)

        for edge in edges_in:
            B = A.copy()
            B[edge] = 0
            B = B.tobytes()
            edge_array[i,dags.index(B)] = 1

        for edge in edges_giving:
            B = A.copy()
            B[edge] = 1
            B = B.tobytes()
            edge_array[i,dags.index(B)] = 1


    G = nx.Graph(edge_array)
    pos = nx.spectral_layout(G)
    
    allbics = [score_DAG(samples, np.reshape(np.frombuffer(x, dtype="int"), (4,4)), real_partition)[0] for x in dags]
    node_size = [100*np.exp(2*x) for x in allbics]
    print("True was", sorted(allbics).index(score_DAG(samples, real_A, real_partition)[0]), "best of 543")
    
   


    # setup for MCMC
  
    fig, ax = plt.subplots()

    def init():
        global current_edge_array
        global current_partition
        global current_sorted_edges
        global current_bic
        global labels
        current_edge_array = np.zeros((4,4))
        current_partition = real_partition.copy()
        for _ in range(num_nodes-len(current_partition)):
            current_partition.append(set())
        current_sorted_edges = get_sorted_edges(current_edge_array)
        current_bic = score_DAG(samples, current_edge_array, current_partition)
        CausalGreedyColor(samples, 0)

        labels = {i:0 for i in range(543)}
        labels[0] = 1
        colors = ["lightsteelblue" for _ in range(543)]
        colors[dags.index(real_A.tobytes())] = "gold"
        colors[0] = "lightpink"
        nx.draw(G, pos=pos, labels=labels, node_color=colors, with_labels=True, node_size=node_size, width=0.1)


    def update(frame):
        global labels
        global current_edge_array
        global current_partition
        global current_sorted_edges
        global current_bic
        global current_node

        ax.clear()
        current_edge_array, current_partition, current_bic, current_sorted_edges, _ = iteration(samples, current_edge_array, current_partition, current_bic, current_sorted_edges)

        A = current_edge_array.astype("int")
        labels[dags.index(A.tobytes())] += 1
        colors = ["lightsteelblue" for _ in range(543)]
        colors[dags.index(real_A.tobytes())] = "gold"
        colors[dags.index(A.tobytes())] = "lightpink"
        nx.draw(G, pos=pos, labels=labels, node_color=colors, with_labels=True, node_size=node_size, width=0.1)


    ani = animation.FuncAnimation(fig=fig, func=update, frames=10_000, interval=10, init_func=init)
    plt.show()

    
    
if __name__ == "__main__":
    main()
