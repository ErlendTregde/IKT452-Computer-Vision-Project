"""Save training history as a figure suitable for reports."""

import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works on headless servers
import matplotlib.pyplot as plt


def save_training_plot(
    history: Dict[str, List[float]],
    save_dir: str = "checkpoints",
    filename: str = "training_history.png",
) -> str:
    """
    Save a multi-panel training history figure.

    Panels:
      1. Train loss vs. Validation loss
      2. mAP@0.50 and mAP@0.50:0.95
      3. Precision, Recall, F1

    Args:
        history: dict returned by train()
        save_dir: directory to write the figure
        filename: output filename

    Returns:
        Absolute path of the saved figure.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    has_metrics = len(history.get("map_50", [])) > 0

    n_panels = 3 if has_metrics else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    # --- Panel 1: loss ---
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train loss", linewidth=2)
    if history.get("val_loss"):
        ax.plot(epochs, history["val_loss"], label="Val loss", linewidth=2)
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if has_metrics:
        metric_epochs = range(1, len(history["map_50"]) + 1)

        # --- Panel 2: mAP ---
        ax = axes[1]
        ax.plot(metric_epochs, history["map_50"], label="mAP@0.50", linewidth=2)
        if history.get("map_50_95"):
            ax.plot(metric_epochs, history["map_50_95"], label="mAP@0.50:0.95", linewidth=2)
        ax.set_title("Mean Average Precision")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("mAP")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- Panel 3: precision / recall / F1 ---
        ax = axes[2]
        if history.get("precision"):
            ax.plot(metric_epochs, history["precision"], label="Precision", linewidth=2)
        if history.get("recall"):
            ax.plot(metric_epochs, history["recall"], label="Recall", linewidth=2)
        if history.get("f1"):
            ax.plot(metric_epochs, history["f1"], label="F1", linewidth=2)
        ax.set_title("Precision / Recall / F1")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Training plot saved to {out_path}")
    return out_path
