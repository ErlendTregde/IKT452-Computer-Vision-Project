"""Checkpoint save/load utilities."""

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    is_best: bool = False,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, path)

    if is_best:
        best_path = os.path.join(os.path.dirname(path), "best_model.pth")
        torch.save(checkpoint, best_path)
        print(f"Saved best model to {best_path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    device: torch.device,
) -> Tuple[int, float]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["loss"]
