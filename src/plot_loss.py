import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_loss(log_file="logs/black-noaug/gemco.csv"):
    df = pd.read_csv(log_file)
    
    plt.figure(figsize=(10,6))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(df["epoch"], df["val_loss"], label="val Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training vs Val Loss", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.xticks(np.linspace(df["epoch"].min(), df["epoch"].max(), 10, dtype=int))

    plt.tight_layout()
    plt.savefig("img-loss/gemco.png")
    plt.show()

if __name__ == "__main__":
    plot_loss()
