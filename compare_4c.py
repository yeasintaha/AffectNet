import pandas as pd
import matplotlib.pyplot as plt

resnet = pd.read_csv("runs/resnet18_ferplus_majority_128px_sqrt_inverse/history.csv")
alexnet = pd.read_csv("runs/alexnet_ferplus_majority_128px_sqrt_inverse_30ep/history.csv")

metrics = [("train_loss", "Train loss"),
           ("val_loss", "Validation loss"),
           ("accuracy", "Validation accuracy"),
           ("macro_f1", "Validation macro-F1")]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, (col, title) in zip(axes.flat, metrics):
    ax.plot(resnet.epoch, resnet[col], marker="o", label="ResNet-18")
    ax.plot(alexnet.epoch, alexnet[col], marker="s", label="AlexNet")
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle("AlexNet vs ResNet-18 in training", fontsize=14)
fig.tight_layout()
fig.savefig("figure_4c.png", dpi=150, bbox_inches="tight")