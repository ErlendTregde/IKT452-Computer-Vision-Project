"""Display utilities for visualizing dataset samples and model predictions."""

import torch
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as transforms
from typing import Optional, Dict, Tuple
from torch.utils.data import DataLoader


def show_image(
    image: torch.Tensor,
    target: Dict,
    show_labels: bool = True,
    figsize: Tuple[int, int] = (15, 10),
    class_names: Optional[Dict[int, str]] = None,
    dataset: Optional[object] = None,
) -> None:
    """Display a single image tensor with ground-truth bounding boxes."""
    img_pil = transforms.ToPILImage()(image)
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(img_pil)

    colors = plt.cm.tab10.colors  # type: ignore

    for box, label in zip(target["boxes"], target["labels"]):
        x1, y1, x2, y2 = box.tolist()
        label_id = label.item()
        color = colors[label_id % 10]

        ax.add_patch(plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, color=color, linewidth=2,
        ))

        if show_labels:
            if dataset and hasattr(dataset, "class_names"):
                inverse = {v: k for k, v in dataset.class_names.items()}
                original_id = inverse.get(label_id, label_id)
            else:
                original_id = label_id

            text = class_names[original_id] if class_names and original_id in class_names else str(original_id)
            ax.text(x1, y1, text, color=color, fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_sample(
    dataloader: DataLoader,
    idx: int = 0,
    show_labels: bool = True,
    figsize: Tuple[int, int] = (15, 10),
    class_names: Optional[Dict[int, str]] = None,
) -> None:
    """Display a single dataset sample (by index) with bounding boxes."""
    image, target = dataloader.dataset[idx]
    show_image(image, target,
               show_labels=show_labels, figsize=figsize,
               class_names=class_names, dataset=dataloader.dataset)


def show_batch(
    dataloader: DataLoader,
    num_images: int = 4,
    show_labels: bool = True,
    figsize: Tuple[int, int] = (20, 15),
    class_names: Optional[Dict[int, str]] = None,
) -> None:
    """Display a batch of images with bounding boxes."""
    images, targets = next(iter(dataloader))
    num_images = min(num_images, len(images))
    ncols = 2
    nrows = (num_images + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if num_images > 1 else [axes]
    colors = plt.cm.tab10.colors  # type: ignore

    for i, ax in enumerate(axes):
        if i >= num_images:
            ax.axis("off")
            continue

        ax.imshow(transforms.ToPILImage()(images[i]))

        for box, label in zip(targets[i]["boxes"], targets[i]["labels"]):
            x1, y1, x2, y2 = box.tolist()
            label_id = label.item()
            color = colors[label_id % 10]

            ax.add_patch(plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, color=color, linewidth=2,
            ))

            if show_labels:
                if hasattr(dataloader.dataset, "class_names"):
                    inverse = {v: k for k, v in dataloader.dataset.class_names.items()}
                    original_id = inverse.get(label_id, label_id)
                else:
                    original_id = label_id
                text = class_names[original_id] if class_names and original_id in class_names else str(original_id)
                ax.text(x1, y1, text, color=color, fontsize=8,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        ax.axis("off")

    plt.tight_layout()
    plt.show()


def show_predictions(
    model: torch.nn.Module,
    image: torch.Tensor,
    threshold: float = 0.5,
    figsize: Tuple[int, int] = (15, 10),
    class_names: Optional[Dict[int, str]] = None,
) -> None:
    """Display a single image with model predictions."""
    model.eval()
    with torch.no_grad():
        predictions = model.predict(image, threshold=threshold)

    img_pil = transforms.ToPILImage()(image)
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(img_pil)
    colors = plt.cm.tab10.colors  # type: ignore

    for box, label, score in zip(predictions["boxes"], predictions["labels"], predictions["scores"]):
        x1, y1, x2, y2 = box.tolist()
        label_id = label.item()
        color = colors[label_id % 10]

        ax.add_patch(plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, color=color, linewidth=2,
        ))

        text = f"{class_names[label_id]}: {score.item():.2f}" if class_names and label_id in class_names \
            else f"Class {label_id}: {score.item():.2f}"
        ax.text(x1, y1, text, color=color, fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.axis("off")
    plt.tight_layout()
    plt.show()
