import pandas as pd
import matplotlib.pyplot as plt
import glob, os

CURVE_DIR = "../results/accuracy_curves/"
PLOT_DIR = "../results/plots/"
os.makedirs(PLOT_DIR, exist_ok=True)

for csv_path in glob.glob(CURVE_DIR + "*.csv"):
    name = os.path.basename(csv_path).split("_acc.csv")[0]
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8, 5))
    plt.plot(df['epoch'], df['train_acc'], label='Train', color='teal')
    plt.plot(df['epoch'], df['val_acc'], label='Val', color='orange')
    plt.plot(df['epoch'], df['test_acc'], label='Test', color='red')

    # Find oversmoothing point = epoch where validation accuracy peaks
    peak_epoch = df['val_acc'].idxmax() + 1
    peak_val = df['val_acc'].max()
    plt.axvline(peak_epoch, color='purple', linestyle='--', label=f'Oversmoothing (epoch {peak_epoch})')
    plt.scatter(peak_epoch, peak_val, color='purple', zorder=10, s=50)
    plt.title(f"Accuracy vs Epochs: {name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{name}_oversmoothing.png")
    plt.close()

print("Oversmoothing plots saved in results/plots/")
