import numpy as np
import pickle
from typing import Tuple, Any

PATH = 'data/bilibili1'


def read() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open(f'{PATH}/features.pkl', 'rb') as fp:
        nf = pickle.load(fp)
    with open(f'{PATH}/features_tag.pkl', 'rb') as fp:
        tf = pickle.load(fp)
    nodes = np.genfromtxt(f'{PATH}/node.txt', np.int)
    edges = np.genfromtxt(f'{PATH}/edge.txt', np.int)
    labels = np.genfromtxt(f'{PATH}/label.txt', np.str)
    # print(nodes)
    # print(edges)
    # print(tags)

    n_node = nodes.shape[0]
    adj = np.zeros([n_node, n_node], dtype=np.int)
    for u, v in edges:
        adj[u, v] = 1
    # print(adj)

    label_set = set(labels)
    t2i = {t: i for i, t in enumerate(label_set)}
    label_onehot = np.zeros([n_node, len(label_set)], dtype=np.int)
    for i, t in enumerate(labels):
        label_onehot[i, t2i[t]] = 1

    return nf, tf, adj, label_onehot
