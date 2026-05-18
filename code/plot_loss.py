
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_loss(log_file="logs/mei/scenario2.csv"):
    df = pd.read_csv(log_file)
    
    plt.figure(figsize=(10,6))
    
    # Plot train & val loss tanpa marker
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss", linewidth=2)
    
    # Highlight epoch dengan val_loss minimum
    min_val_idx = df["val_loss"].idxmin()
    plt.scatter(df["epoch"][min_val_idx], df["val_loss"][min_val_idx],
                color='red', s=100, zorder=5, label='Min Val Loss')
    plt.text(df["epoch"][min_val_idx], df["val_loss"][min_val_idx] + 0.05,
             f"{df['val_loss'][min_val_idx]:.3f}", color='red', fontsize=10, fontweight='bold')
    
    # Watermark center
    plt.text(0.5, 0.5, "Scenario 2: DETR BISINDO", fontsize=40, color='gray', alpha=0.3,
             ha='center', va='center', rotation=30, transform=plt.gca().transAxes)
    
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training vs Val Loss", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.xticks(np.linspace(df["epoch"].min(), df["epoch"].max(), 10, dtype=int))
    plt.tight_layout()
    plt.savefig("img-loss/mei/scenario2.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_loss()




# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# def plot_loss(log_file="logs/black-noaug/gemco.csv"):
#     df = pd.read_csv(log_file)
    
#     plt.figure(figsize=(10,6))
#     plt.plot(df["epoch"], df["train_loss"], label="Train Loss", linewidth=2)
#     plt.plot(df["epoch"], df["val_loss"], label="val Loss", linewidth=2)
#     plt.xlabel("Epoch", fontsize=12)
#     plt.ylabel("Loss", fontsize=12)
#     plt.title("Training vs Val Loss", fontsize=14, fontweight="bold")
#     plt.legend()
#     plt.grid(True, linestyle="--", alpha=0.6)

#     plt.xticks(np.linspace(df["epoch"].min(), df["epoch"].max(), 10, dtype=int))

#     plt.tight_layout()
#     plt.savefig("img-loss/gemco.png")
#     plt.show()

# if __name__ == "__main__":
#     plot_loss()


