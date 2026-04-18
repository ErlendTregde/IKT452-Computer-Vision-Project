"""Training loop for Faster R-CNN."""

import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.eval import evaluate_detection_metrics
from utils.plot import save_training_plot
from utils.validate import evaluate


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    print_freq: int = 10,
) -> float:
    model.train()
    model.to(device)
    total_loss = 0.0

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward() #type: ignore
        optimizer.step()

        total_loss += loss.item() #type: ignore

        # Release cached GPU memory to prevent allocator fragmentation accumulating
        # across batches — important for models with many intermediate tensors.
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if batch_idx % print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                f"Loss: {loss.item():.4f}, LR: {lr:.6f}" #type: ignore
            )

    return total_loss / len(dataloader)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    num_epochs: int,
    device: torch.device,
    lr: float = 0.001,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    optimizer_type: str = "sgd",
    checkpoint_dir: str = "checkpoints",
    resume: Optional[str] = None,
    compute_detection_metrics: bool = True,
    metric_score_threshold: float = 0.01,
) -> Dict[str, List[float]]:
    os.makedirs(checkpoint_dir, exist_ok=True)

    params = (
        model.get_param_groups(lr) #type: ignore
        if hasattr(model, "get_param_groups")
        else model.parameters()
    )
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    # Smooth cosine decay — avoids the sharp LR-to-zero collapse from StepLR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    start_epoch = 0
    best_loss = float("inf")
    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [],
        "map_50": [], "map_50_95": [],
        "precision": [], "recall": [], "f1": [],
    }

    if resume and os.path.exists(resume):
        start_epoch, best_loss = load_checkpoint(model, optimizer, resume, device)
        print(f"Resumed from epoch {start_epoch}, best loss: {best_loss:.4f}")
        start_epoch += 1

    print(f"Starting training from epoch {start_epoch} to {num_epochs}")
    print(f"Device: {device}, Learning rate: {lr}")

    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*50}\nEpoch {epoch}/{num_epochs}\n{'='*50}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        history["train_loss"].append(train_loss)
        print(f"\nTrain Loss: {train_loss:.4f}")

        scheduler.step()

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device)
            history["val_loss"].append(val_loss)
            print(f"Val Loss: {val_loss:.4f}")

            if compute_detection_metrics:
                metrics = evaluate_detection_metrics(
                    model, val_loader, device,
                    score_threshold=metric_score_threshold,
                )
                history["map_50"].append(float(metrics["mAP@0.50"])) #type: ignore
                history["map_50_95"].append(float(metrics["mAP@0.50:0.95"])) #type: ignore
                history["precision"].append(float(metrics["precision"])) #type: ignore
                history["recall"].append(float(metrics["recall"])) #type: ignore
                history["f1"].append(float(metrics["f1"])) #type: ignore
                print(
                    f"mAP@0.50={metrics['mAP@0.50']:.4f}, "
                    f"mAP@0.50:0.95={metrics['mAP@0.50:0.95']:.4f}, "
                    f"Precision={metrics['precision']:.4f}, "
                    f"Recall={metrics['recall']:.4f}, "
                    f"F1={metrics['f1']:.4f}"
                )

            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
        else:
            is_best = train_loss < best_loss
            if is_best:
                best_loss = train_loss

        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
        save_checkpoint(model, optimizer, epoch, best_loss, checkpoint_path, is_best)

    print(f"\nTraining complete. Best validation loss: {best_loss:.4f}")
    save_training_plot(history, save_dir=checkpoint_dir)
    return history
