import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
import urllib.request
import tarfile
import os

def load_cora():
    if not os.path.exists("cora"):
        url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
        urllib.request.urlretrieve(url, "cora.tgz")
        tarfile.open("cora.tgz").extractall(".")
    idx_features_labels = np.genfromtxt("cora/cora.content", dtype=str)
    features = np.array(idx_features_labels[:, 1:-1], dtype=float)
    labels = LabelEncoder().fit_transform(idx_features_labels[:,-1])
    node_ids = idx_features_labels[:,0]
    id_map = {j:i for i,j in enumerate(node_ids)}
    edges_unordered = np.genfromtxt("cora/cora.cites", dtype=str)
    edge_index = np.array([[id_map[x[0]], id_map[x[1]]] for x in edges_unordered], dtype=int)
    N = features.shape[0]
    adj = sp.coo_matrix((np.ones(edge_index.shape[0]), (edge_index[:,0], edge_index[:,1])),
                        shape=(N,N), dtype=float)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return features, labels, adj

def split_masks(n_total, train_size=140, val_size=500, test_size=1000, seed=42):
    np.random.seed(seed)
    idx = np.arange(n_total)
    np.random.shuffle(idx)
    train_mask = np.zeros(n_total, dtype=bool)
    val_mask = np.zeros(n_total, dtype=bool)
    test_mask = np.zeros(n_total, dtype=bool)
    train_mask[idx[:train_size]] = True
    val_mask[idx[train_size:train_size+val_size]] = True
    test_mask[idx[train_size+val_size:train_size+val_size+test_size]] = True
    return train_mask, val_mask, test_mask
