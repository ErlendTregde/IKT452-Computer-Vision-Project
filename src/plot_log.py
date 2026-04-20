"""
Parse a run.log produced by main.py and save a training history plot.

Usage:
    uv run src/plot_log.py run.log
    uv run src/plot_log.py run.log --out results/run1.png
"""

import argparse
import re
import sys

sys.path.insert(0, "src")
from utils.plot import save_training_plot


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", help="path to run.log")
    parser.add_argument("--out", default="checkpoints/training_history.png",
                        help="output image path (default: checkpoints/training_history.png)")
    args = parser.parse_args()

    history = parse_log(args.log)
    if not history["train_loss"]:
        print("No training data found in log.")
        return

    import os
    out_dir = os.path.dirname(args.out) or "."
    out_file = os.path.basename(args.out)
    save_training_plot(history, save_dir=out_dir, filename=out_file)


if __name__ == "__main__":
    main()
