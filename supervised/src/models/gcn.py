import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp

def normalize_adj(adj):
    """
    Symmetric normalization of adjacency (as in Kipf & Welling, 2017).
    Given scipy.sparse adjacency matrix, adds self-loops and computes:
    D^-0.5 * (A + I) * D^-0.5
    Returns dense torch tensor for computation.
    """
    adj = adj + sp.eye(adj.shape[0])
    deg = np.array(adj.sum(1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
    adj_norm = torch.tensor(adj_norm.todense(), dtype=torch.float32)
    return adj_norm

class GCNLayer(nn.Module):
    """
    A single GCN layer (as described by Kipf & Welling, 2017). Modular for stacking.
    Computes: H' = ReLU( D^-0.5 * (A + I) * D^-0.5 * H * W )
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, features, adj):
        # features: Tensor [num_nodes, in_features]
        # adj: scipy.sparse matrix
        adj_norm = normalize_adj(adj).to(features.device)
        support = torch.mm(adj_norm, features)  # aggregate neighbors with normalization
        out = self.linear(support)              # apply learned weight transformation
        return torch.relu(out)

class GCN(nn.Module):
    """
    Two-layer GCN for node classification (input->hidden->output).
    """
    def __init__(self, in_features, hidden_dim, out_features):
        super().__init__()
        self.gcn1 = GCNLayer(in_features, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, out_features)

    def forward(self, features, adj):
        h = self.gcn1(features, adj)
        h = self.gcn2(h, adj)
        return h  # for logits (no softmax; use CrossEntropyLoss)
