import torch
import torch.nn as nn

class MPNNLayer(nn.Module):
    """
    Fast, vectorized Message Passing Layer for nodewise aggregation.
    """
    def __init__(self, in_dim, out_dim, message_fn=None, update_fn=None):
        super().__init__()
        self.message_fn = message_fn if message_fn is not None else self.default_message
        self.update_fn = update_fn if update_fn is not None else self.default_update
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, features, adj, edge_attr=None):
        # Convert adj to dense tensor if needed
        if hasattr(adj, 'todense'):
            adj = torch.tensor(adj.todense(), dtype=features.dtype, device=features.device)
        N, F = features.shape

        # Fast message aggregation: each node aggregates its neighbors' features
        agg_msg = adj @ features  # [N, F_in]

        # Optional: degree normalization (like GCN)
        deg = adj.sum(1).clamp(min=1).unsqueeze(-1)
        agg_msg = agg_msg / deg

        # Update (default: add own features to aggregated message)
        updated = self.update_fn(features, agg_msg)
        return self.lin(updated)

    # Default message function: just use neighbor features
    def default_message(self, h_i, h_j, e_ij):
        return h_j

    # Default update function: add own features to aggregated message
    def default_update(self, h, m):
        return h + m
