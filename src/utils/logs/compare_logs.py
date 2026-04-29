"""
Compare training metrics from multiple .log files on the same plots.

Usage:
    uv run src/compare_logs.py logs/old/faster-rcnn.log logs/old/FPN.log
    uv run src/compare_logs.py logs/**/*.log --out results/comparison.png
    uv run src/compare_logs.py logs/old/faster-rcnn.log:FasterRCNN logs/old/FPN.log:FPN

Each log can optionally have a label appended with a colon:
    path/to/file.log:MyLabel
Otherwise the label is derived from the filename.
"""

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_log(path: str) -> dict:
    history: dict = {
        "train_loss": [], "val_loss": [],
        "map_50": [], "map_50_95": [],
        "precision": [], "recall": [], "f1": [],
    }
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r"Train Loss:\s*([\d.]+)", line)
            if m:
                history["train_loss"].append(float(m.group(1)))
            m = re.match(r"Val Loss:\s*([\d.]+)", line)
            if m:
                history["val_loss"].append(float(m.group(1)))
            m = re.search(
                r"mAP@0\.50=([\d.]+).*mAP@0\.50:0\.95=([\d.]+).*"
                r"Precision=([\d.]+).*Recall=([\d.]+).*F1=([\d.]+)",
                line,
            )
            if m:
                history["map_50"].append(float(m.group(1)))
                history["map_50_95"].append(float(m.group(2)))
                history["precision"].append(float(m.group(3)))
                history["recall"].append(float(m.group(4)))
                history["f1"].append(float(m.group(5)))
    return history


def label_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def main():
    parser = argparse.ArgumentParser(description="Compare multiple training logs")
    parser.add_argument(
        "logs", nargs="+",
        help="log files to compare, optionally with labels: path/to/file.log:Label"
    )
    parser.add_argument("--out", default="results/comparison.png",
                        help="output image path (default: results/comparison.png)")
    parser.add_argument("--metric", default="map_50",
                        choices=["map_50", "map_50_95", "precision", "recall", "f1"],
                        help="which metric to show in addition to loss (default: map_50)")
    args = parser.parse_args()

    runs = []
    for entry in args.logs:
        if ":" in entry:
            path, label = entry.rsplit(":", 1)
        else:
            path, label = entry, label_from_path(entry)
        history = parse_log(path)
        if not history["train_loss"]:
            print(f"WARNING: no data found in {path}, skipping.")
            continue
        runs.append((label, history))

    if not runs:
        print("No valid log files found.")
        sys.exit(1)

    metric_label = {
        "map_50": "mAP@0.50", "map_50_95": "mAP@0.50:0.95",
        "precision": "Precision", "recall": "Recall", "f1": "F1",
    }[args.metric]

    has_metrics = any(len(h[args.metric]) > 0 for _, h in runs)

    plt.rcParams.update({"font.size": 13})

    if has_metrics:
        fig, axes_grid = plt.subplots(2, 2, figsize=(16, 12))
        axes = [axes_grid[0, 0], axes_grid[0, 1], axes_grid[1, 0], axes_grid[1, 1]]
    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(8, 6))
        axes = [ax0]

    for label, history in runs:
        epochs = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["train_loss"], label=f"{label} train", linewidth=2)
        if history["val_loss"]:
            axes[0].plot(epochs, history["val_loss"], label=f"{label} val",
                         linewidth=2, linestyle="--")

    axes[0].set_title("Loss", fontsize=15, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    if has_metrics:
        for label, history in runs:
            data = history[args.metric]
            if data:
                axes[1].plot(range(1, len(data) + 1), data, label=label, linewidth=2)

        axes[1].set_title(metric_label, fontsize=15, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel(metric_label)
        axes[1].set_ylim(0, 1)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)

        for label, history in runs:
            p, r = history["precision"], history["recall"]
            ep = range(1, len(p) + 1)
            if p:
                axes[2].plot(ep, p, label=f"{label}", linewidth=2)
            if r:
                axes[2].plot(ep, r, label=f"{label}", linewidth=2, linestyle="--")

        axes[2].set_title("Precision (—) / Recall (--)", fontsize=15, fontweight="bold")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Score")
        axes[2].set_ylim(0, 1)
        axes[2].legend(fontsize=11)
        axes[2].grid(True, alpha=0.3)

        for label, history in runs:
            f = history["f1"]
            if f:
                axes[3].plot(range(1, len(f) + 1), f, label=label, linewidth=2)

        axes[3].set_title("F1 Score", fontsize=15, fontweight="bold")
        axes[3].set_xlabel("Epoch")
        axes[3].set_ylabel("F1")
        axes[3].set_ylim(0, 1)
        axes[3].legend(fontsize=11)
        axes[3].grid(True, alpha=0.3)

    fig.suptitle("Model Comparison", fontsize=17, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison plot saved to {args.out}")


if __name__ == "__main__":
    main()
