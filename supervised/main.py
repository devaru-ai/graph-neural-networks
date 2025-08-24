import sys
import argparse

sys.path.append('src')
from data_utils import load_cora, split_masks
from train import train_basic, train_gcn, train_gat, train_mpnn
from evaluate import evaluate_basic, evaluate_gcn, evaluate_gat, evaluate_mpnn
from models.basic_gnn import BasicGNNLayer
from models.gcn import GCNLayer
from models.gat import GAT
from models.mpnn import MPNNLayer


def get_model(name, **kwargs):
    if name == "basic_gnn":
        return BasicGNNLayer()
    elif name == "gcn":
        return GCNLayer(kwargs['in_dim'], kwargs['out_dim'])
    elif name == "gat":
        return GAT(kwargs['in_dim'], kwargs['out_dim'])
    elif name == "mpnn":
        return MPNNLayer(kwargs['in_dim'], kwargs['out_dim'])
    else:
        raise ValueError(f"Unknown model: {name}")

def main():
    parser = argparse.ArgumentParser(description="GNN Node Classification")
    parser.add_argument('--model', type=str, default="basic_gnn", help="Model to use: basic_gnn, gcn, etc.")
    parser.add_argument('--hidden_dim', type=int, default=16, help="Hidden dimension size for GCN")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    args = parser.parse_args()

    X, y, adj = load_cora()
    train_mask, val_mask, test_mask = split_masks(X.shape[0])
    import torch
    features = torch.tensor(X, dtype=torch.float)
    labels   = torch.tensor(y, dtype=torch.long)
    n_classes = int(labels.max().item()) + 1

    if args.model == "basic_gnn":
        model = get_model(args.model)
        W = torch.nn.Parameter(torch.randn(n_classes, features.shape[1]) / features.shape[8] ** 0.5)
        print("Training...")
        train_basic(model, W, features, labels, adj, train_mask, epochs=args.epochs)
        print("Evaluating...")
        acc = evaluate_basic(model, W, features, labels, adj, test_mask)
    elif args.model == "gcn":
        model = get_model(args.model, in_dim=features.shape[1], out_dim=n_classes)
        print("Training...")
        train_gcn(model, features, labels, adj, train_mask, epochs=args.epochs)
        print("Evaluating...")
        acc = evaluate_gcn(model, features, labels, adj, test_mask)
    elif args.model == "gat":
        model = get_model(args.model, in_dim=features.shape[1], out_dim=n_classes)
        print("Training...")
        train_gat(model, features, labels, adj, train_mask, epochs=args.epochs)
        print("Evaluating...")
        acc = evaluate_gat(model, features, labels, adj, test_mask)
    elif args.model == "mpnn":
        model = get_model(args.model, in_dim=features.shape[1], out_dim=n_classes)
        print("Training...")
        train_mpnn(model, features, labels, adj, train_mask, epochs=args.epochs)
        print("Evaluating...")
        acc = evaluate_mpnn(model, features, labels, adj, test_mask)  
    else:
        raise ValueError(f"Unknown model: {args.model}")
    print(f"Test Accuracy: {acc:.4f}")

if __name__ == '__main__':
    main()
