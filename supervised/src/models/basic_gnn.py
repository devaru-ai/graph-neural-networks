import torch
import numpy as np

def aggregate(H, adj):
    # Mean aggregation of neighbor features
    deg = np.clip(adj.sum(1).A1, 1, None)
    agg_H = adj @ H / deg[:,None]
    return agg_H

def combine(H_last, c_agg):
    return H_last + c_agg

class BasicGNNLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No learnable weights in this basic example

    def forward(self, features, adj):
        # features: Tensor (n_nodes, n_feats)
        H_np = features.detach().cpu().numpy()
        agg = torch.tensor(aggregate(H_np, adj), dtype=features.dtype, device=features.device)
        return combine(features, agg)
