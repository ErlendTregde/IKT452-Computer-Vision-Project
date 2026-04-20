"""Validation loss computation."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """
    Compute average validation loss.

    Torchvision detection models only return losses in training mode, so we keep
    the model in train() but wrap the forward pass in torch.no_grad().
    """
    model.train()
    model.to(device)
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            total_loss += sum(loss_dict.values()).item() #type: ignore
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else float("inf")
