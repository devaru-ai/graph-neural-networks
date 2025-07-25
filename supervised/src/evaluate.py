import torch

def evaluate(gnn, W, features, labels, adj, mask):
    gnn.eval()
    with torch.no_grad():
        H1 = gnn(features, adj)
        logits = H1 @ W.t()
        preds = logits[mask].argmax(dim=1)
        acc = (preds == labels[mask]).float().mean().item()
    return acc
