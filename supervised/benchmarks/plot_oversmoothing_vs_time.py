import pandas as pd
import matplotlib.pyplot as plt
import os

CURVE_DIR = "../results/accuracy_curves/"
RESULTS_CSV = "../results/benchmarks.csv"
PLOT_DIR = "../results/plots/"
os.makedirs(PLOT_DIR, exist_ok=True)

model_names = ["basic_gnn", "gcn", "gat", "mpnn"]
color_map = {
    "basic_gnn": "gray",
    "gcn": "blue",
    "gat": "green",
    "mpnn": "orange"
}

# Gather oversmoothing epochs and train times
oversmooth_epochs = []
train_times = []
labels = []

# Read total training time per model
bench_df = pd.read_csv(RESULTS_CSV).set_index('model')

for name in model_names:
    csv_path = f"{CURVE_DIR}/{name}_acc.csv"
    if not os.path.exists(csv_path) or name not in bench_df.index:
        continue
    curve_df = pd.read_csv(csv_path)
    # Find oversmoothing epoch (peak validation accuracy)
    peak_epoch = curve_df['val_acc'].idxmax() + 1
    oversmooth_epochs.append(peak_epoch)
    train_times.append(float(bench_df.loc[name, 'train_time_s']))
    labels.append(name.upper())

plt.figure(figsize=(8,6))
plt.scatter(oversmooth_epochs, train_times, c=[color_map.get(m, "black") for m in model_names], s=120)
for i, label in enumerate(labels):
    plt.text(oversmooth_epochs[i], train_times[i]+2, label, fontsize=12, ha='center', va='bottom')
plt.xlabel("Oversmoothing Epoch (Peak Validation Accuracy)")
plt.ylabel("Total Training Time (s)")
plt.title("Oversmoothing Point vs. Training Time by GNN Model")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/oversmoothing_vs_train_time.png")
plt.show()
