import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Single-head Graph Attention Layer.
    Implements GAT from Veličković et al., 2018.
    """
    def __init__(self, in_dim, out_dim, dropout=0.6, alpha=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(2 * out_dim, 1))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, adj):
        if not torch.is_tensor(adj):
            adj = torch.tensor(adj.todense(), dtype=features.dtype, device=features.device)

        Wh = features @ self.W                 # [N, F_out]
        N = Wh.size(0)

        Wh_repeat_i = Wh.repeat(1, N).view(N * N, -1)
        Wh_repeat_j = Wh.repeat(N, 1)
        a_input = torch.cat([Wh_repeat_i, Wh_repeat_j], dim=1).view(N, N, 2 * self.out_dim)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))   # [N, N]

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)

class GAT(nn.Module):
    """
    Single-layer (single-head) GAT wrapper.
    """
    def __init__(self, in_dim, out_dim, dropout=0.6, alpha=0.2):
        super().__init__()
        self.gat = GraphAttentionLayer(in_dim, out_dim, dropout=dropout, alpha=alpha)

    def forward(self, features, adj):
        return self.gat(features, adj)
