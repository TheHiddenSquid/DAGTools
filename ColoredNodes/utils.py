import math
import random
from collections.abc import Iterable

import networkx as nx
import numpy as np
from numba import njit
from scipy.optimize import linear_sum_assignment


class PartitionManager:
    __slots__ = ['n', 'node_to_block', 'P', 'active_blocks', 'block_to_active_idx', 
                 'empty_blocks', 'block_to_empty_idx', 'chunk_size', 'proposal_nodes', 'current_step']  
      
    def __init__(self, n, chunk_size=10_000):
        self.n = n
        self.chunk_size = chunk_size

        # Primary Data Structures
        self.P = [set([i]) for i in range(n)]
        self.node_to_block = list(range(n))
        
        # O(1) Management for Active Blocks
        self.active_blocks = list(range(n))
        self.block_to_active_idx = list(range(n))
        
        # O(1) Management for Empty Blocks
        self.empty_blocks = []
        self.block_to_empty_idx = [-1] * n # -1 means not in empty list

        # Setup for fast RNG
        self.proposal_nodes = np.random.randint(0, n, size=chunk_size).tolist()
        self.current_step = 0

    def _get_random_node(self):
        if self.current_step >= self.chunk_size:
            self.proposal_nodes = np.random.randint(0, self.n, size=self.chunk_size).tolist()
            self.current_step = 0
        
        node = self.proposal_nodes[self.current_step]
        self.current_step += 1
        return node

    def _remove_active(self, b_id):
        idx = self.block_to_active_idx[b_id]
        last_b = self.active_blocks[-1]
        self.active_blocks[idx] = last_b
        self.block_to_active_idx[last_b] = idx
        self.active_blocks.pop()

    def _add_active(self, b_id):
        self.block_to_active_idx[b_id] = len(self.active_blocks)
        self.active_blocks.append(b_id)

    def _remove_empty(self, b_id):
        idx = self.block_to_empty_idx[b_id]
        last_b = self.empty_blocks[-1]
        self.empty_blocks[idx] = last_b
        self.block_to_empty_idx[last_b] = idx
        self.empty_blocks.pop()

    def _add_empty(self, b_id):
        self.block_to_empty_idx[b_id] = len(self.empty_blocks)
        self.empty_blocks.append(b_id)

    def propose_move(self):
        """
        Returns (node, old_block, new_block)
        """
        node = self._get_random_node()
        old_idx = self.node_to_block[node]
        new_idx = random.choice(self.active_blocks)

        if len(self.P[old_idx]) == 1:
            while new_idx == old_idx:
                new_idx = random.choice(self.active_blocks)
        else:
            if new_idx == old_idx:
                new_idx = self.empty_blocks[0]
                
        return node, old_idx, new_idx
    
    def perform_move(self, node, old_idx, new_idx):
        """
        Update self.P for easy MCMC function eval
        """
        self.P[old_idx].remove(node)
        self.P[new_idx].add(node)

    def undo_move(self, node, old_idx, new_idx):
        """
        Undoes preform_move in case MCMC fails
        """
        self.P[new_idx].remove(node)
        self.P[old_idx].add(node)
                
    def commit_move(self, node, old_idx, new_idx):
        """Updates the data-structure after MCMC acceptance"""
        # 1. Update Old Block
        if len(self.P[old_idx]) == 0:
            self._remove_active(old_idx)
            self._add_empty(old_idx)
            
        # 2. Update New Block
        if  len(self.P[new_idx]) == 1:
            self._remove_empty(new_idx)
            self._add_active(new_idx)
            
        self.node_to_block[node] = new_idx

    def __repr__(self):
        return str(self.P)
    
    def __str__(self):
        return str(self.P)

# DAG generation
def get_random_DAG(num_nodes, edge_prob = 0.5):
    # Generate erdős-rényi under-triangular matrix
    edge_array = np.zeros((num_nodes, num_nodes), dtype=np.int64)
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i > j:
                if edge_prob > random.random():
                    edge_array[i,j] = 1      
    
    # Shuffle nodes
    G = nx.DiGraph(edge_array)
    node_mapping = dict(zip(G.nodes(), sorted(G.nodes(), key=lambda k: random.random())))
    G_new = nx.relabel_nodes(G, node_mapping)
    edge_array = nx.adjacency_matrix(G_new, node_mapping).todense()

    return edge_array

def generate_colored_DAG(num_nodes, num_colors, edge_prob = 0.5):
    
    # Create partition for colors
    partition = generate_random_partition(num_nodes, num_colors)

    # Generate lambda matrix
    lambda_matrix = get_random_DAG(num_nodes, edge_prob)
    lambda_matrix = lambda_matrix.astype(np.float64)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if lambda_matrix[i,j] == 1:
                lambda_matrix[i,j] = random.uniform(0.25,1) * random.randrange(-1,2,2)

    # Generate omega matrix
    choices = [random.uniform(0.2,2) for _ in range(num_colors)]
    omega_matrix = [None] * num_nodes
    for i, part in enumerate(partition):
        for node in part:
            omega_matrix[node] = choices[i]


    return partition, lambda_matrix, omega_matrix

def generate_sample(size, lambda_matrix, omega_matrix):
    no_nodes = len(omega_matrix)
    errors = np.random.normal(loc=0.0, scale=np.sqrt(omega_matrix)[:, None], size=(no_nodes, size))
    X = np.linalg.inv(np.eye(no_nodes) - lambda_matrix).T
    sample = X @ errors
    return sample.T






# Graph functions

def is_DAG(A):
    numnodes = A.shape[0]
    if numnodes < 8:
        P = A
    else:
        P = A.astype(np.float32)
    power = 1
    while power < numnodes:
        P = P @ P
        power *= 2
    if np.argmax(P) != 0:
        return False
    return not P[0,0]

@njit(cache=True)
def is_reachable(A, i, j):
    """
    Returns true if node i reachable from node j given adjacency matrix A
    """
    n = A.shape[0]
    if i == j:
        return True
    
    visited = np.zeros(n, dtype=np.bool_)
    visited[j] = True
    
    queue = np.empty(n, dtype=np.int32)
    queue[0] = j
    head = 0
    tail = 1
    
    while head < tail:
        curr = queue[head]
        head += 1
        
        for neighbor in range(n):
            if A[curr, neighbor]:
                if neighbor == i:
                    return True
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue[tail] = neighbor
                    tail += 1
                    
    return False

@njit(cache=True)
def get_parents(node, A):
    parents = []
    n = np.shape(A)[0]
    for i in range(n):
        if A[i, node] == 1:
            parents.append(i)
    return parents

def is_reachable_masks(parent_masks, i, j):
    """Return whether i is reachable from j, using incoming-edge bitmasks.

    Walk ancestors of i to find j. Thus this checks whether adding i -> j
    would create a cycle, with the same argument order as is_reachable().
    Python integer masks support node indices beyond 63.
    """
    i, j = int(i), int(j)
    if i == j:
        return True

    target = 1 << j
    visited = 1 << i
    pending = parent_masks[i]
    while pending:
        if pending & target:
            return True
        bit = pending & -pending
        pending ^= bit
        visited |= bit
        pending |= parent_masks[bit.bit_length() - 1] & ~visited
    return False

def parent_masks_to_adjacency(parent_masks):
    """Convert incoming-edge masks to an int64 matrix with A[parent, child] = 1."""
    n = len(parent_masks)
    A = np.zeros((n, n), dtype=np.int64)
    for child, mask in enumerate(parent_masks):
        while mask:
            bit = mask & -mask
            A[bit.bit_length() - 1, child] = 1
            mask ^= bit
    return A

def get_sorted_edges(edge_array):
  
    tmp_edge_array = edge_array.copy()
    n = np.shape(tmp_edge_array)[0]

    edges_in_DAG = []
    edges_giving_DAGs = []
    edges_not_giving_DAGs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if tmp_edge_array[i, j] == 1:
                edges_in_DAG.append((i,j))
                continue

            tmp_edge_array[i, j] = 1
            if is_DAG(tmp_edge_array):
                edges_giving_DAGs.append((i,j))
            else:
                edges_not_giving_DAGs.append((i,j))
            tmp_edge_array[i,j] = 0

    return [edges_in_DAG, edges_giving_DAGs, edges_not_giving_DAGs]

def score_DAG(data, A, P, *, center=True):
    """Score a DAG and coloring with sample-size-normalized Gaussian BIC.

    Center each column by default without changing the input array. Set
    center=False for already-centered data or an intentionally fixed zero
    mean. Variables are not standardized.
    """
    if center:
        data = data - data.mean(axis=0)
    my_data = data.T
    num_nodes = data.shape[1]
    num_samples = data.shape[0]
    BIC_constant = np.log(num_samples)/(num_samples*2)

    # Calculate ML-eval via least sqares
    omegas_ML = [0] * num_nodes
    for node in range(num_nodes):
        parents = get_parents(node, A)
        a = my_data[parents,:]
        b = my_data[node,:]
        beta = np.linalg.solve(a @ a.T, a @ b)
        x = b - a.T @ beta
        ss_res = np.dot(x,x)
        omegas_ML[node] = ss_res / num_samples


    # Calculate decomposed BIC
    bic_decomp = [0] * num_nodes
    for i, block in enumerate(P):
        if len(block) == 0:
            continue
        tot = 0
        for node in block:
            tot += omegas_ML[node]
        block_omega = tot / len(block)

        bic_decomp[i] = -len(block) * (np.log(block_omega) + 1)
    
    # Calculate full BIC
    bic = sum(bic_decomp) / 2
    bic -= BIC_constant * (sum(1 for part in P if len(part)>0) + np.sum(A))
    
    return bic


# Partition generation and manipulation

def generate_random_partition(no_nodes, no_colors):
    partition = [set() for _ in range(no_colors)]
    for node in range(no_nodes):
        color = random.randrange(no_colors)
        partition[color].add(node)
    partition = [set(x) for x in sorted_partition(partition)]
    for _ in range(no_colors-len(partition)):
        partition.append(set())
    return partition

def sorted_partition(partition):
    num_nodes = sum(len(x) for x in partition)
    dones = [False] * num_nodes
    sorted_partition = []
    for i in range(num_nodes):
        if dones[i] == True:
            continue
        for part in partition:
            if i in part:
                sorted_partition.append(sorted(part))
                for k in part:
                    dones[k] = True
    return sorted_partition



# Get CPDAG given DAG and partition
def R1(A):
    num_nodes = A.shape[0]
    for node1 in range(num_nodes):
        for node2 in range(num_nodes):
            if node2 == node1:
                continue
            if A[node1, node2] == 1 and A[node2, node1] == 0:
                for node3 in range(num_nodes):
                    if node3 == node1 or node3 == node2:
                        continue
                    if A[node2, node3] == 1 and A[node3, node2] == 1:
                        if A[node1, node3] == 0 and A[node3, node1] == 0:
                            A[node3, node2] = 0
                            return A, True
    return A, False
def R2(A):
    num_nodes = A.shape[0]
    for node1 in range(num_nodes):
        for node2 in range(num_nodes):
            if node2 == node1:
                continue
            if A[node1, node2] == 1 and A[node2, node1] == 0:
                for node3 in range(num_nodes):
                    if node3 == node1 or node3 == node2:
                        continue
                    if A[node2, node3] == 1 and A[node3, node2] == 0:
                        if A[node1, node3] == 1 and A[node3, node1] == 1:
                            A[node3, node1] = 0
                            return A, True
    return A, False
def R3(A):
    num_nodes = A.shape[0]
    for node1 in range(num_nodes):
        for node2 in range(num_nodes):
            if node2 == node1:
                continue
            if A[node1, node2] == 1 and A[node2, node1] == 0:
                for node3 in range(num_nodes):
                    if node3 == node1 or node3 == node2:
                        continue
                    if A[node3, node2] == 1 and A[node2, node3] == 0:
                        if A[node1, node3] == 0 and A[node3, node1] == 0:
                            for node4 in range(num_nodes):
                                if node4 == node1 or node4 == node2 or node4 == node3:
                                    continue
                                if A[node1, node4] == 1 and A[node4, node1] == 1:
                                    if A[node2, node4] == 1 and A[node4, node2] == 1:
                                        if A[node3, node4] == 1 and A[node4, node3] == 1:
                                            A[node2, node4] = 0
                                            return A, True
    return A, False

def getCPDAG(A, P):
    # Skeleton
    num_nodes = A.shape[0]
    newA = np.array(A + A.T != 0, dtype=np.int64)
 
    # V-structures
    v_triplets = list(nx.dag.v_structures(nx.DiGraph(A)))
    for pa1, col, pa2 in v_triplets:
        newA[(col, pa1)] = 0
        newA[(col, pa2)] = 0

    # Repeat R1, R2, R3
    while True:
        newA, did_change1 = R1(newA)
        newA, did_change2 = R2(newA)
        newA, did_change3 = R3(newA)
        if not (did_change1 or did_change2 or did_change3):
            break
    
    # Add color info
    nodes_to_look_at = []
    for part in P:
        if len(part) >= 2:
            nodes_to_look_at += part
    
    for node in nodes_to_look_at:
        for other in range(num_nodes):
            if A[node, other] == 1:
                newA[other, node] = 0
            if A[other, node] == 1:
                newA[node, other] = 0  
    
    # Repeat R1, R2, R3
    while True:
        newA, did_change1 = R1(newA)
        newA, did_change2 = R2(newA)
        if not (did_change1 or did_change2):
            break

    return newA



# Metric calculation functions

def calc_SHD(A, B):
    return np.sum(np.abs(A-B))

def calc_CHD(P1, P2):
    tmp_p1 = P1.copy()
    tmp_p2 = P2.copy()

    num_nodes = sum(len(x) for x in tmp_p1)
    num_colors = max(len(tmp_p1), len(tmp_p2))

    tmp_p1 += [[]]*(num_colors - len(tmp_p1))
    tmp_p2 += [[]]*(num_colors - len(tmp_p2))

    cost_matrix = np.zeros((num_colors, num_colors), dtype="int")
    for i in range(num_colors):
        for j in range(num_colors):
            cost_matrix[i,j] = len(set(tmp_p1[i]).intersection(set(tmp_p2[j])))
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)

    return num_nodes - cost_matrix[row_ind, col_ind].sum()



# Other functions

def generate_color_map(P):
    if len(P) > 12:
        raise ValueError("Too many colors needed for color-map generation")
    colors = ["red", "green", "blue", "yellow", "purple", "brown", "white", "black", "orange", "pink", "cyan", "gray"]
    length = sum([len(x) for x in P])
    color_map = [None] * length

    for i, part in enumerate(P):
        for node in part:
            color_map[node] = colors[i]

    return color_map

def hash_DAG(edge_array, partition):
    return (edge_array.tobytes(), tuple(tuple(x) for x in sorted_partition(partition)))

def calc_ss_res(sample, A, nodes):
    num_samples = sample.shape[0]
    S = (1/num_samples) * sample.T @ sample

    if not isinstance(nodes, Iterable):
        node = nodes
        parents = get_parents(node, A)
        return calc_lstsq(S, node, tuple(parents))   
    else:
        ss_res = [0] * len(nodes)
        for i, node in enumerate(nodes):
            parents = get_parents(node, A)
            ss_res[i] = calc_lstsq(S, node, tuple(parents))
        return ss_res
    
# @functools.cache
def calc_lstsq(S, node, parents):
    g_nn = S[node, node]
    
    if len(parents) == 0:
        return g_nn
    
    g_pa = S[parents, node]
    G_pa = S[parents, :][:, parents]
    beta = np.linalg.solve(G_pa, g_pa)
    ss_res = g_nn - np.dot(beta, g_pa)
    
    return ss_res




# Solve dynamic programming to find optimal partition
def get_optimal_partition(ss_res, penalty, size_terms=None):
    """Return the optimal variance partition and its negative BIC contribution.

    The returned cost includes color penalties but excludes the edge penalty.
    For a graph with e edges, its BIC is -cost - penalty*e.
    """
    residuals = np.asarray(ss_res, dtype=np.float64)

    if np.any(residuals <= 0):
        raise ValueError("Residual variances must be positive")

    order = np.argsort(residuals, kind="stable")
    sorted_residuals = residuals[order]

    best_score, previous = optimal_partition_core(sorted_residuals, penalty, size_terms)

    partition = []
    end = len(residuals)

    while end > 0:
        start = int(previous[end])
        partition.append(set(int(x) for x in order[start:end]))
        end = start

    partition.reverse()
    return sorted_partition(partition), best_score
    
@njit(cache=True)
def optimal_partition_core(r, penalty, size_terms=None):
    p = len(r)
    if size_terms is None:
        size_terms = np.empty(p + 1)
        size_terms[0] = 0.0
        for size in range(1, p + 1):
            size_terms[size] = size * (math.log(size) - 1.0)

    prefix = np.empty(p + 1)
    prefix[0] = 0.0
    for i in range(p):
        prefix[i + 1] = prefix[i] + r[i]

    dp = np.empty(p + 1)
    previous = np.empty(p + 1, dtype=np.int64)

    dp[0] = 0.0
    previous[0] = -1

    for end in range(1, p + 1):
        best_score = np.inf
        best_start = -1

        for start in range(end):
            size = end - start
            block_sum = prefix[end] - prefix[start]
            block_cost = 0.5 * (size * math.log(block_sum) - size_terms[size]) + penalty
            candidate = dp[start] + block_cost

            if candidate < best_score:
                best_score = candidate
                best_start = start

        dp[end] = best_score
        previous[end] = best_start

    return dp[p], previous

