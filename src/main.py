"""
Fine-tune EfficientDet-D0 on the NorgesGruppen grocery dataset.

Run from the project root:
    uv run python src/main.py
"""

import os
import sys

import torch
from effdet import DetBenchTrain

# Allow `from models.*` and `from utils.*` imports
sys.path.insert(0, os.path.dirname(__file__))

from models.efficientDet import efficientdet_d0
from utils.train import get_dataloaders, train_one_epoch
from utils.eval import evaluate


# ── Config ──────────────────────────────────────────────────────────────────
ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(ROOT, "dataset")
CHECKPOINT   = os.path.join(ROOT, "efficientdet_d0_grocery.pth")

IMG_SIZE   = 512   # D0 input resolution
BATCH_SIZE = 4     # lower to 2 if you run out of GPU memory
EPOCHS     = 20
LR         = 1e-4
# ────────────────────────────────────────────────────────────────────────────


def print_metrics(metrics: dict):
    print(f"  mAP@0.50:      {metrics['mAP@0.50']:.4f}")
    print(f"  mAP@0.50:0.95: {metrics['mAP@0.50:0.95']:.4f}")
    print(f"  Precision:     {metrics['Precision']:.4f}")
    print(f"  Recall:        {metrics['Recall']:.4f}")
    print(f"  F1:            {metrics['F1']:.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Model
    model = efficientdet_d0(pretrained_backbone=True)
    bench = DetBenchTrain(model).to(device)

    # Data
    train_dl, val_dl = get_dataloaders(DATASET_ROOT, IMG_SIZE, BATCH_SIZE)
    print(f"Train: {len(train_dl.dataset)} images")
    print(f"Val:   {len(val_dl.dataset)} images\n")

    # Optimizer + cosine LR schedule
    optimizer = torch.optim.AdamW(bench.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_map50 = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(bench, optimizer, train_dl, device, epoch, IMG_SIZE)
        scheduler.step()

        metrics = evaluate(model, val_dl, device, IMG_SIZE)

        print(f"\nEpoch {epoch:>2}/{EPOCHS} | train_loss {train_loss:.4f}")
        print_metrics(metrics)

        if metrics["mAP@0.50"] > best_map50:
            best_map50 = metrics["mAP@0.50"]
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"  => Saved best model (mAP@0.50={best_map50:.4f})")

    print("\n── Final Results ──────────────────────────────")
    final = evaluate(model, val_dl, device, IMG_SIZE)
    print_metrics(final)
    print(f"Best mAP@0.50 during training: {best_map50:.4f}")


if __name__ == "__main__":
    main()
