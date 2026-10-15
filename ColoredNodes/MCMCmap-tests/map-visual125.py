import random
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.append("../")
from collections import defaultdict

import utils
from MCMCfuncs import CausalMCMC, MCMC_iteration, score_DAG_full
from utils import get_sorted_edges


def get_all_3node_DAGs(color = False):
    AllDAGs = defaultdict(lambda: 0)
    if color == True:
        cap = 125
    else:
        cap = 25

    while len(AllDAGs) < cap:
        partition, A, _ = utils.generate_colored_DAG(3,3, random.random())
        A = np.array(A != 0, dtype="int")
        h1 = A.tobytes()
        if color == True:
            h2 = tuple(tuple(x) for x in utils.sorted_partition(partition))
            AllDAGs[(h1,h2)] += 1
        else:
            AllDAGs[h1] += 1

    dags = list(AllDAGs.keys())
    dags.sort()
    return dags


def main():

    # General setup
    seed = 16 # best 16
    random.seed(seed)
    np.random.seed(seed)
    num_nodes = 3
    num_colors = 3
    sparse = True
    sample_size = 1_000

    real_partition, real_lambda_matrix, real_omega_matrix = utils.generate_colored_DAG(num_nodes, num_colors, sparse)
    real_A = np.array(real_lambda_matrix != 0, dtype="int")
    real_partition = utils.sorted_partition(real_partition)
    real_tuple_P = tuple(tuple(x) for x in real_partition)
    samples = utils.generate_sample(sample_size, real_lambda_matrix, real_omega_matrix)


    # setup for hypergraph
    global dags
    dags = get_all_3node_DAGs(color = True)

    edge_array = np.zeros((125,125), dtype="int")
    for i, cDAG in enumerate(dags):
        A = np.reshape(np.frombuffer(cDAG[0], dtype="int"), (3,3))
        P = cDAG[1]
        edges_in, edges_giving, _ = get_sorted_edges(A)

        for edge in edges_in:
            B = A.copy()
            B[edge] = 0
            B = B.tobytes()
            edge_array[i,dags.index((B,P))] = 1

        for edge in edges_giving:
            B = A.copy()
            B[edge] = 1
            B = B.tobytes()
            edge_array[i,dags.index((B,P))] = 1

        A = A.tobytes()
        oldP = P
        P = [list(x) for x in P]
        if len(P) == 1:
            P.append([])
            P.append([])
        if len(P) == 2:
            P.append([])

        for node in P[0]:
            tmp = P[0].copy()
            tmp.remove(node)
            for new_P in [[tmp, P[1] + [node], P[2]], [tmp, P[1], P[2] + [node]]]:
                new_P = utils.sorted_partition(new_P)
                new_P = tuple(tuple(x) for x in new_P)
                if new_P == oldP: continue
                edge_array[i,dags.index((A,new_P))] = 1
        
        for node in P[1]:
            tmp = P[1].copy()
            tmp.remove(node)
            for new_P in [[P[0] + [node], tmp, P[2]], [P[0], tmp, P[2] + [node]]]:
                new_P = utils.sorted_partition(new_P)
                new_P = tuple(tuple(x) for x in new_P)
                if new_P == oldP: continue
                edge_array[i,dags.index((A,new_P))] = 1
        
        for node in P[2]:
            tmp = P[2].copy()
            tmp.remove(node)
            for new_P in [[P[0] + [node], P[1], tmp], [P[0], P[1] + [node], tmp]]:
                new_P = utils.sorted_partition(new_P)
                new_P = tuple(tuple(x) for x in new_P)
                if new_P == oldP: continue
                edge_array[i,dags.index((A,new_P))] = 1




    G = nx.Graph(edge_array)
    pos = nx.spring_layout(G, seed=1)

    

    allbics = [utils.score_DAG(samples, np.reshape(np.frombuffer(x, dtype="int"), (3,3)), p) for (x,p) in dags]
    node_size = [6*np.exp(2*x) for x in allbics]
    node_size = [200*x for x in allbics]

    print(sorted(allbics))
    print("postive:", sum(1 for x in allbics if x>0))
    print("negative:", sum(1 for x in allbics if x<0))
    print("True was", sorted(allbics).index(utils.score_DAG(samples, real_A, real_partition)), "best of 125")
    
   
   


    # setup for MCMC
  
    fig, ax = plt.subplots()

    def init():
        global current_edge_array
        global current_partition
        global current_bic
        global current_ML_data
        global labels

        current_edge_array = np.zeros((3,3))
        current_partition = [{i} for i in range(3)]
        CausalMCMC(samples, 0)
        current_bic, current_ML_data = score_DAG_full(current_edge_array, current_partition)

        current_partition_tuple = tuple(tuple(x) for x in utils.sorted_partition(current_partition))
        labels = {i:0 for i in range(125)}
        labels[dags.index((current_edge_array.tobytes(), current_partition_tuple))] = 1
        colors = ["lightsteelblue" for _ in range(125)]
        colors[dags.index((real_A.tobytes(),real_tuple_P))] = "gold"
        colors[dags.index((current_edge_array.tobytes(), current_partition_tuple))] = "lightpink"
        nx.draw(G, pos=pos, labels=labels, node_color=colors, with_labels=True, node_size=node_size)


    def update(frame):
        global labels
        global current_edge_array
        global current_partition
        global current_bic
        global current_ML_data
        global current_node

        ax.clear()
        move = random.choices([0, 1], k=1, weights=[0.4,0.6])[0]
        current_edge_array, current_partition, current_bic, current_ML_data,  _ = MCMC_iteration(move, current_edge_array, current_partition, current_bic, current_ML_data)

        A = current_edge_array.astype("int")
        current_partition_tuple = tuple(tuple(x) for x in utils.sorted_partition(current_partition))
        labels[dags.index((A.tobytes(),current_partition_tuple))] += 1
        colors = ["lightsteelblue" for _ in range(125)]
        colors[dags.index((real_A.tobytes(), real_tuple_P))] = "gold"
        colors[dags.index((A.tobytes(),current_partition_tuple))] = "lightpink"
        nx.draw(G, pos=pos, labels=labels, node_color=colors, with_labels=True, node_size=node_size)


    ani = animation.FuncAnimation(fig=fig, func=update, frames=10_000, interval=10, init_func=init)
    plt.show()

    
    
if __name__ == "__main__":
    main()
