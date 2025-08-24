import torch

def compute_accuracy(model, features, labels, adj, mask, edge_attr=None, W=None, is_basic_gnn=False):
    model.eval()
    with torch.no_grad():
        if is_basic_gnn:
            H1 = model(features, adj)
            logits = H1 @ W.t()
        elif edge_attr is not None:
            logits = model(features, adj, edge_attr)
        else:
            logits = model(features, adj)
        preds = logits[mask].argmax(dim=1)
        acc = (preds == labels[mask]).float().mean().item()
    return acc

def evaluate_basic(gnn, W, features, labels, adj, mask):
    return compute_accuracy(gnn, features, labels, adj, mask, W=W, is_basic_gnn=True)

def evaluate_gcn(model, features, labels, adj, mask):
    return compute_accuracy(model, features, labels, adj, mask)

def evaluate_gat(model, features, labels, adj, mask):
    return compute_accuracy(model, features, labels, adj, mask)

def evaluate_mpnn(model, features, labels, adj, mask, edge_attr=None):
    return compute_accuracy(model, features, labels, adj, mask, edge_attr=edge_attr)
