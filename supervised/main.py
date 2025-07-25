import sys
import argparse

sys.path.append('src')
from data_utils import load_cora, split_masks
from train import train
from evaluate import evaluate
from models.basic_gnn import BasicGNNLayer

# (future expansion: import other models as needed)

def get_model(name, **kwargs):
    if name == "basic_gnn":
        return BasicGNNLayer()
    # elif name == "gcn":
    #     from gcn import GCNLayer
    #     return GCNLayer(**kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")

def main():
    parser = argparse.ArgumentParser(description="GNN Node Classification")
    parser.add_argument('--model', type=str, default="basic_gnn", help="Model to use: basic_gnn, gcn, etc.")
    args = parser.parse_args()

    X, y, adj = load_cora()
    train_mask, val_mask, test_mask = split_masks(X.shape[0])
    import torch
    features = torch.tensor(X, dtype=torch.float)
    labels   = torch.tensor(y, dtype=torch.long)
    # adj as numpy or torch—pass whichever your model expects

    model = get_model(args.model)
    n_classes = int(labels.max().item()) + 1
    W = torch.nn.Parameter(torch.randn(n_classes, features.shape[1]) / features.shape[1] ** 0.5)

    print("Training...")
    train(model, W, features, labels, adj, train_mask, epochs=100)
    print("Evaluating...")
    acc = evaluate(model, W, features, labels, adj, test_mask)
    print(f"Test Accuracy: {acc:.4f}")

if __name__ == '__main__':
    main()
