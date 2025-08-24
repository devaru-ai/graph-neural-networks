import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

CURVE_DIR = "../results/accuracy_curves/"
PLOT_DIR = "../results/plots/"
os.makedirs(PLOT_DIR, exist_ok=True)

color_map = {
    "basic_gnn": "gray",
    "gcn": "blue",
    "gat": "green",
    "mpnn": "orange"
}
model_names = ["basic_gnn", "gcn", "gat", "mpnn"]

plt.figure(figsize=(10,6))
for name in model_names:
    csv_path = f"{CURVE_DIR}/{name}_acc.csv"
    if not os.path.exists(csv_path):
        continue
    df = pd.read_csv(csv_path)
    plt.plot(df['epoch'], df['val_acc'], label=name.upper(), color=color_map.get(name, None), linewidth=2)
    # Mark oversmoothing (peak val accuracy)
    peak_epoch = df['val_acc'].idxmax() + 1
    peak_val = df['val_acc'].max()
    plt.scatter(peak_epoch, peak_val, color=color_map.get(name, None), s=70)
    plt.axvline(peak_epoch, color=color_map.get(name, None), linestyle='--', alpha=0.4)

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Oversmoothing Comparison Across GNN Architectures")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/gnn_oversmoothing_comparison.png")
plt.show()
