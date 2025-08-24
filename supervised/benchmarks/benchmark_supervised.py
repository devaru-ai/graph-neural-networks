import sys
import os
import time
import torch
import csv
import pandas as pd

# Set src path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from data_utils import load_cora, split_masks
from train import train_basic, train_gcn, train_gat, train_mpnn
from evaluate import evaluate_basic, evaluate_gcn, evaluate_gat, evaluate_mpnn
from models.basic_gnn import BasicGNNLayer
from models.gcn import GCNLayer
from models.gat import GAT
from models.mpnn import MPNNLayer

RESULTS_CSV = "../results/benchmarks.csv"
CURVE_DIR = "../results/accuracy_curves/"
EPOCHS = 100

def main():
    X, y, adj = load_cora()
    train_mask, val_mask, test_mask = split_masks(X.shape[0])
    features = torch.tensor(X, dtype=torch.float)
    labels = torch.tensor(y, dtype=torch.long)
    n_classes = int(labels.max().item()) + 1

    results = []

    # List: (name, instantiate_fn, train_fn, eval_fn)
    algos = [
        ("basic_gnn", lambda: BasicGNNLayer(), train_basic, evaluate_basic),
        ("gcn", lambda: GCNLayer(features.shape[1], n_classes), train_gcn, evaluate_gcn),
        ("gat", lambda: GAT(features.shape[1], n_classes), train_gat, evaluate_gat),
        ("mpnn", lambda: MPNNLayer(features.shape[1], n_classes), train_mpnn, evaluate_mpnn),
    ]

    os.makedirs("../results", exist_ok=True)
    os.makedirs(CURVE_DIR, exist_ok=True)

    print("\n================= BENCHMARKING =================\n")
    for name, make_model, train_fn, eval_fn in algos:
        print(f"\n--- Benchmarking: {name} ---")
        model = make_model()
        start = time.time()

        if name == "basic_gnn":
            W = torch.nn.Parameter(torch.randn(n_classes, features.shape[1]) / features.shape[1] ** 0.5)
            # Train, log accuracies
            train_accs, val_accs, test_accs = train_fn(model, W, features, labels, adj, train_mask, val_mask, test_mask, epochs=EPOCHS)
            acc = eval_fn(model, W, features, labels, adj, test_mask)
            params = W.numel()
        else:
            train_accs, val_accs, test_accs = train_fn(model, features, labels, adj, train_mask, val_mask, test_mask, epochs=EPOCHS)
            acc = eval_fn(model, features, labels, adj, test_mask)
            params = sum(p.numel() for p in model.parameters())

        train_time = time.time() - start
        print(f"Test Accuracy: {acc:.4f} | Train Time: {train_time:.2f}s | Params: {params}")
        results.append([name, f"{acc:.4f}", f"{train_time:.2f}", params])

        # Save accuracy curves as CSV per model
        pd.DataFrame({
            "epoch": range(1, EPOCHS+1),
            "train_acc": train_accs,
            "val_acc": val_accs,
            "test_acc": test_accs
        }).to_csv(f"{CURVE_DIR}/{name}_acc.csv", index=False)

    # Write summary results to CSV
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "test_acc", "train_time_s", "num_params"])
        writer.writerows(results)

    print("\nAll benchmarks finished! Results stored in results/benchmarks.csv and per-epoch curves in results/accuracy_curves/\n")

if __name__ == "__main__":
    main()
