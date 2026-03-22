"""
visualize.py
Plotting utilities:
  - Training curves (loss, accuracy)
  - Federated vs centralised comparison
  - Confusion matrix heatmap
  - Per-node accuracy across rounds
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path


# ── Style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor" : "white",
    "axes.facecolor"   : "#f8f9fa",
    "axes.grid"        : True,
    "grid.alpha"       : 0.4,
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})

COLORS = {
    "fedprox"     : "#2196F3",
    "fedavg"      : "#FF9800",
    "centralized" : "#4CAF50",
    "node_1"      : "#E91E63",
    "node_2"      : "#9C27B0",
    "node_3"      : "#00BCD4",
}


def plot_training_curves(
    history: dict,
    save_path: str = None,
    title: str = "Training Curves",
):
    """
    Plots loss and accuracy curves side by side.

    Args:
        history   : {"train_loss": [...], "val_loss": [...],
                     "train_acc": [...],  "val_acc": [...]}
        save_path : if provided, saves figure to this path
        title     : plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train",
                 color="#E53935", linewidth=2)
    axes[0].plot(epochs, history["val_loss"],   label="Val",
                 color="#1E88E5", linewidth=2, linestyle="--")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train",
                 color="#E53935", linewidth=2)
    axes[1].plot(epochs, history["val_acc"],   label="Val",
                 color="#1E88E5", linewidth=2, linestyle="--")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x:.1f}%")
    )
    axes[1].legend()

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_federated_rounds(
    round_metrics: dict,
    save_path: str = None,
):
    """
    Plots accuracy per federated round for each strategy.

    Args:
        round_metrics : {
            "fedprox"     : [acc_r1, acc_r2, ...],
            "fedavg"      : [...],
            "centralized" : float  (single value, drawn as hline)
        }
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    for strategy, accs in round_metrics.items():
        if strategy == "centralized":
            ax.axhline(
                y=accs, color=COLORS["centralized"],
                linestyle=":", linewidth=2,
                label=f"Centralised Baseline ({accs:.1f}%)"
            )
        else:
            rounds = range(1, len(accs) + 1)
            ax.plot(rounds, accs,
                    color=COLORS.get(strategy, "grey"),
                    linewidth=2.5, marker="o", markersize=4,
                    label=strategy.upper())

    ax.set_title("Federated Learning — Accuracy per Round",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Accuracy (%)")
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x:.1f}%")
    )
    ax.legend()
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_confusion_matrix(
    cm: list,
    classes: list = None,
    save_path: str = None,
    title: str = "Confusion Matrix",
):
    """
    Plots a labeled confusion matrix heatmap.

    Args:
        cm      : 2D list or numpy array [[TP, FP], [FN, TN]]
        classes : class label names
        save_path, title : optional
    """
    if classes is None:
        classes = ["Normal", "Tuberculosis"]

    cm = np.array(cm)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2%",
        cmap="Blues", ax=ax,
        xticklabels=classes, yticklabels=classes,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Proportion"},
    )
    # Overlay raw counts
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j + 0.5, i + 0.72,
                f"(n={cm[i, j]})",
                ha="center", va="center",
                fontsize=9, color="grey"
            )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_node_distributions(
    distributions: dict,
    save_path: str = None,
):
    """
    Bar chart showing class distribution per hospital node.

    Args:
        distributions : {
            "Node 1": {"Normal": 350, "Tuberculosis": 120},
            "Node 2": {...},
            ...
        }
    """
    nodes   = list(distributions.keys())
    classes = list(next(iter(distributions.values())).keys())
    x       = np.arange(len(nodes))
    width   = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, cls in enumerate(classes):
        counts = [distributions[n][cls] for n in nodes]
        bars   = ax.bar(
            x + i * width, counts, width,
            label=cls,
            color=["#1E88E5", "#E53935"][i],
            alpha=0.85,
        )
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                str(int(bar.get_height())),
                ha="center", va="bottom", fontsize=9