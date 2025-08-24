import matplotlib.pyplot as plt
import pandas as pd
import os

RESULTS_CSV = "../results/benchmarks.csv"
PLOTS_DIR = "../results/plots"

def plot_benchmarks():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df = pd.read_csv(RESULTS_CSV)

    # Accuracy Bar Chart
    plt.figure(figsize=(7,4))
    plt.bar(df['model'], df['test_acc'].astype(float), color='teal')
    plt.ylabel("Test Accuracy")
    plt.xlabel("Model")
    plt.title("GNN Model Accuracy Comparison")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/accuracy_bar.png")
    plt.close()

    # Training Time Bar Chart
    plt.figure(figsize=(7,4))
    plt.bar(df['model'], df['train_time_s'].astype(float), color='orange')
    plt.ylabel("Total Training Time (s)")
    plt.xlabel("Model")
    plt.title("GNN Model Training Time Comparison")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/train_time_bar.png")
    plt.close()

    # Number of Parameters Bar Chart
    plt.figure(figsize=(7,4))
    plt.bar(df['model'], df['num_params'].astype(int), color='purple')
    plt.ylabel("Number of Parameters")
    plt.xlabel("Model")
    plt.title("GNN Model Parameter Count")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/param_count_bar.png")
    plt.close()

    print(f"Plots saved in {PLOTS_DIR}/")

if __name__ == "__main__":
    plot_benchmarks()
