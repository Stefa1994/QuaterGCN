'''
Code for creating digraph with antiparallel edges
'''


from typing import Tuple
import math
import itertools
import math
import networkx as nx
from networkx.utils import py_random_state
import numpy as np
import scipy.sparse as sp
import numpy.random as rnd
from random import randint
import random

import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix, triu
import scipy


@py_random_state(3)
def stochastic_block_model_2(
    sizes, p, nodelist=None, seed=None, directed=True, selfloops=False, sparse=True, negative_edges=False
):
    """Returns a stochastic block model graph.
    """
    # Check if dimensions match
    if len(sizes) != len(p):
        raise nx.NetworkXException("'sizes' and 'p' do not match.")
    # Check for probability symmetry (undirected) and shape (directed)
    for row in p:
        if len(p) != len(row):
            raise nx.NetworkXException("'p' must be a square matrix.")
    if not directed:
        p_transpose = [list(i) for i in zip(*p)]
        for i in zip(p, p_transpose):
            for j in zip(i[0], i[1]):
                if abs(j[0] - j[1]) > 1e-08:
                    raise nx.NetworkXException("'p' must be symmetric.")
    # Check for probability range
    for row in p:
        for prob in row:
            if prob < 0 or prob > 1:
                raise nx.NetworkXException("Entries of 'p' not in [0,1].")
    # Check for nodelist consistency
    if nodelist is not None:
        if len(nodelist) != sum(sizes):
            raise nx.NetworkXException("'nodelist' and 'sizes' do not match.")
        if len(nodelist) != len(set(nodelist)):
            raise nx.NetworkXException("nodelist contains duplicate.")
    else:
        nodelist = range(0, sum(sizes))

    # Setup the graph conditionally to the directed switch.
    block_range = range(len(sizes))
    if directed:
        g = nx.DiGraph()
        block_iter = itertools.product(block_range, block_range)
    else:
        g = nx.Graph()
        block_iter = itertools.combinations_with_replacement(block_range, 2)
    # Split nodelist in a partition (list of sets).
    size_cumsum = [sum(sizes[0:x]) for x in range(0, len(sizes) + 1)]
    g.graph["partition"] = [
        set(nodelist[size_cumsum[x] : size_cumsum[x + 1]])
        for x in range(0, len(size_cumsum) - 1)
    ]
    # Setup nodes and graph name
    for block_id, nodes in enumerate(g.graph["partition"]):
        for node in nodes:
            g.add_node(node, block=block_id)

    g.name = "stochastic_block_model"

    # Test for edge existence
    parts = g.graph["partition"]
    for i, j in block_iter:
        if i == j:
            if directed:
                if selfloops:
                    edges = itertools.product(parts[i], parts[i])
                else:
                    edges = itertools.permutations(parts[i], 2)
            else:
                edges = itertools.combinations(parts[i], 2)
                if selfloops:
                    edges = itertools.chain(edges, zip(parts[i], parts[i]))
            for e in edges:
                if seed.random() < p[i][j]:
                    if negative_edges:
                        value = randint(-4, 4)
                    else:
                        value = randint(2, 4) # Nuovi valori
                    g.add_edge(*e, weight=value)  # __safe
        else:
            edges = itertools.product(parts[i], parts[j])
        if sparse:
            if p[i][j] == 1:  # Test edges cases p_ij = 0 or 1
                for e in edges:
                    if negative_edges:
                        value = randint(-4, 4)
                    else:
                        value = randint(2, 4) # Nuovi valori
                    g.add_edge(*e, weight=value)  # __safe
            elif p[i][j] > 0:
                while True:
                    try:
                        logrand = math.log(seed.random())
                        skip = math.floor(logrand / math.log(1 - p[i][j]))
                        # consume "skip" edges
                        next(itertools.islice(edges, skip, skip), None)
                        e = next(edges)
                        if negative_edges:
                            value = randint(-4, 4)
                        else:
                            value = randint(2, 4) # Nuovi valori
                        g.add_edge(*e, weight=value)  # __safe
                    except StopIteration:
                        break
        else:
            for e in edges:
                if seed.random() < p[i][j]:
                    if negative_edges:
                        value = randint(-4, 4)
                    else:
                        value = randint(2, 1000) # Nuovi valori
                    g.add_edge(*e, weight=value)  # __safe
    return g

def to_dataset(A, label, save_path):
    import pickle as pk
    import scipy.sparse as sparse
    import torch
    from numpy import linalg as LA
    from torch_geometric.data import Data

   
    label = torch.from_numpy(label).long()

    s_A = sparse.csr_matrix(A)
    coo = s_A.tocoo()
    values = coo.data
    
    indices = np.vstack((coo.row, coo.col))
    indices = torch.from_numpy(indices).long()
    '''
    A_sym = 0.5*(A + A.T)
    A_sym[A_sym > 0] = 1
    d_out = np.sum(np.array(A_sym), axis = 1)
    _, v = LA.eigh(d_out - A_sym)
    features = torch.from_numpy(np.sum(v, axis = 1, keepdims = True)).float()
    '''
    #s = np.random.normal(0, 1.0, (len(A), 1))
    #features = torch.from_numpy(s).float()

    data = Data(#x=features,
                edge_index=indices, edge_weight=coo.data, y=label)


    pk.dump(data, open(save_path, 'wb'))
    return data

def desymmetric_stochastic(sizes = [100, 100, 100],
                probs = [[0.5, 0.45, 0.45],
                         [0.45, 0.5, 0.45],
                         [0.45, 0.45, 0.5]],
                seed = 0,
                off_diag_prob = 0.9,
                undirected_percentage = 0.5,
                opposite_sign = False,
                directed = True,
                negative_edges = False, 
                norm = False):
    from sklearn.model_selection import train_test_split
    
    g = stochastic_block_model_2(sizes, probs, seed=seed, directed=directed, negative_edges =negative_edges)
    original_A = nx.adjacency_matrix(g).todense()
    A = original_A.copy()
    
    # for blocks represent adj within clusters --> elimino tutti collegamenti undirected
    accum_size = 0
    B = np.zeros((A.shape[0], A.shape[0]), int)
    B_first = B.copy()
    value = 0
    for s in sizes:
        if value%2 == 0:
            x, y = np.where(np.triu(original_A[accum_size:s+accum_size,accum_size:s+accum_size]))
        else:            
            x, y = np.where(np.tril(original_A[accum_size:s+accum_size,accum_size:s+accum_size]))
        B_first[x+accum_size, y+accum_size]= A[x+accum_size, y+accum_size]
        accum_size += s
        value += 1
    x, y = np.where(B_first)
    x1, x2, y1, y2 = train_test_split(x, y, test_size=undirected_percentage)
    A[x1, y1] = 0 # remove a portion of antiparalell edges

    # This flag allows assigning an opposite weight to certain elements
    if opposite_sign:
        A[y2, x2] = - A[x2, y2]    


    # for blocks represent adj out of clusters (cluster2cluster edges)
    accum_x, accum_y = 0, 0
    n_cluster = len(sizes)
    
    for i in range(n_cluster):
        accum_y = accum_x + sizes[i]
        for j in range(i+1, n_cluster):
            x, y = np.where(original_A[accum_x:sizes[i]+accum_x, accum_y:sizes[j]+accum_y])
            B[accum_x:sizes[i]+accum_x, accum_y:sizes[j]+accum_y]= A[accum_x:sizes[i]+accum_x, accum_y:sizes[j]+accum_y]
            accum_y += sizes[j]
        accum_x += sizes[i]
    B_new = B + B.T
    x, y = np.where(B_new)

    # This initial operation aims to preserve undirected edges while treating the others as directed.
    x1, x2, y1, y2 =  train_test_split(x, y, test_size=undirected_percentage)

    # It enables the assignment of a weight with an opposite value
    if opposite_sign:
        A[y2, x2] = - A[x2, y2]    
    
    # Remove all the undirected edges
    B_new[x2, y2] = 0
    B_new[y2, x2] = 0
    
    # The direction of flow is determined for the directed connections
    accum_x, accum_y = 0, 0
    for i in range(n_cluster):
        accum_y = accum_x + sizes[i]
        for j in range(i+1, n_cluster):
            x3, y3 = np.where(B_new[accum_x:sizes[i]+accum_x, accum_y:sizes[j]+accum_y])
            x4, x5, y4, y5 = train_test_split(x3, y3, test_size=off_diag_prob)
            A[x4+accum_x, y4+accum_y] = 0
            A[y5+accum_y, x5+accum_x] = 0
                
            accum_y += sizes[j]
            
        accum_x += sizes[i]



    # label assignment based on parameter sizes 
    label = []
    for i, s in enumerate(sizes):
        label.extend([i]*s)
    label = np.array(label)
  
    return np.array(original_A), np.array(A), label

def DSBM(p_in, p_inter, p_q, sizes, cluster, undirected_percentage, opposite_sign, negative_edges, directed):
 
  prob = np.diag([p_in]*cluster)
  prob[prob == 0] = p_inter
  for seed in [10]:
    _, A, label = desymmetric_stochastic(sizes = sizes, probs = prob, off_diag_prob = p_q, seed=seed, undirected_percentage = undirected_percentage, opposite_sign = opposite_sign, \
                                        negative_edges = negative_edges, directed=directed)
    
    data = to_dataset(A, label, save_path = '../data/synthetic_digons/dataset_nodes' + str(sizes[0])  + '_alpha' + str(p_inter)+ '_beta' + str(round(1-p_q, 2)) 
                      + '_undirected-percentage' + str(undirected_percentage) + '_opposite-sign' + str(opposite_sign) +  '_negative-edges' + str(negative_edges) 
                      + '_directed' + str(directed) + '.pk')


if __name__ == "__main__":
    node = 150
    cluster = 5
    sizes = [node]*cluster
    opposite_sign = False
    negative_edges = False
    directed = True
    p_in = 0.1 # collegamento tra elementi interni allo stesso cluster
    # Collegamento tra i diversi cluster del modello (p-inter)
    for p_undirected in [0.2, 0.5, 0.7]:
        undirected_percentage = p_undirected
        for p_inter in [0.1]: 
        # Probability della direzione del flusso a --> b and b --> a
            for p_q in [0.8]:
                DSBM(p_in, p_inter, p_q, sizes, cluster, undirected_percentage, opposite_sign, negative_edges, directed)