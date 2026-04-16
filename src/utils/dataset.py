"""
Dataset utilities for loading YOLO format data for Faster R-CNN.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, Optional


def yolo_to_xyxy(
    x_center: float, y_center: float, width: float, height: float,
    img_width: int, img_height: int
) -> Tuple[float, float, float, float]:
    """
    Convert YOLO format (normalized center coordinates) to XYXY format (absolute pixels).

    Args:
        x_center: Normalized x center (0-1)
        y_center: Normalized y center (0-1)
        width: Normalized width (0-1)
        height: Normalized height (0-1)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Tuple of (x1, y1, x2, y2) in absolute pixel coordinates
    """
    # Convert from normalized to absolute
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    width_abs = width * img_width
    height_abs = height * img_height

    # Convert from center-size to corners
    x1 = x_center_abs - width_abs / 2
    y1 = y_center_abs - height_abs / 2
    x2 = x_center_abs + width_abs / 2
    y2 = y_center_abs + height_abs / 2

    return x1, y1, x2, y2


class YOLODataset(Dataset):
    """
    Dataset class for loading YOLO format annotations for Faster R-CNN training.

    Expected directory structure:
        dataset/
        ├── images/
        │   ├── train/
        │   │   ├── img1.jpg
        │   │   └── ...
        │   └── val/
        │       └── ...
        └── labels/
            ├── train/
            │   ├── img1.txt
            │   └── ...
            └── val/
                └── ...

    Each label file contains one object per line:
        <class_id> <x_center> <y_center> <width> <height>
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transforms = None,
        class_mapping: Optional[Dict[int, int]] = None,
    ):
        """
        Args:
            root_dir: Root directory of dataset
            split: "train" or "val"
            transforms: Optional transforms to apply to images (e.g., ToTensor)
            class_mapping: Optional mapping from dataset class IDs to consecutive IDs
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transforms = transforms
        self.class_mapping = class_mapping

        # Find all images and labels
        # Structure: dataset/train/images/, dataset/train/labels/
        self.image_dir = self.root_dir / split / "images"
        self.label_dir = self.root_dir / split / "labels"

        # Get list of image files
        self.image_files = list(self.image_dir.glob("*.jpg")) + \
                          list(self.image_dir.glob("*.png")) + \
                          list(self.image_dir.glob("*.jpeg"))

        # Sort for reproducibility
        self.image_files.sort()

        # Faster R-CNN expects labels in [1, num_classes - 1], where 0 is background.
        if self.class_mapping is None:
            self.class_mapping = build_class_mapping(self.label_dir)

        self.num_classes = len(self.class_mapping) + 1

    @property
    def class_names(self) -> dict[int, int]:
        """Return mapping from dataset class IDs to consecutive IDs (1, 2, 3, ...)."""
        if self.class_mapping is None:
            raise ValueError("Class mapping not provided. Cannot return class names.")
        return dict(self.class_mapping)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        """
        Load and return one sample.

        Returns:
            image: Tensor of shape (3, H, W)
            target: Dict with "boxes" and "labels"
        """

        if self.class_mapping is None: 
            raise ValueError("Class mapping not provided. Cannot load labels.")

        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        img_width, img_height = image.size

        # Load labels
        label_path = self.label_dir / f"{img_path.stem}.txt"
        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Convert to XYXY format
                    x1, y1, x2, y2 = yolo_to_xyxy(
                        x_center, y_center, width, height,
                        img_width, img_height
                    )

                    # Apply class mapping if provided
                    class_id = self.class_mapping[class_id]

                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_id)

        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            # Handle images with no annotations
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        # Apply transforms
        if self.transforms is not None:
            image = self.transforms(image)

        # Create target dict (Faster R-CNN expects this format)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        return image, target


def collect_class_ids(label_root: Path) -> list[int]:
    """Collect all class IDs from YOLO label files below a directory."""
    class_ids = set()
    for label_path in label_root.glob("**/*.txt"):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_ids.add(int(parts[0]))
    return sorted(class_ids)


def build_class_mapping(label_root: str | Path) -> Dict[int, int]:
    """
    Map dataset class IDs to Faster R-CNN class IDs.

    Background is reserved as class 0, so object classes start at 1.
    """
    class_ids = collect_class_ids(Path(label_root))
    return {old_id: new_id for new_id, old_id in enumerate(class_ids, start=1)}


def create_dataloaders(
    dataset_dir: str,
    batch_size: int = 4,
    num_workers: int = 0,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        dataset_dir: Path to dataset root
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torchvision.transforms import v2 as transforms

    # Define transforms
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])

    class_mapping = build_class_mapping(Path(dataset_dir))

    # Create datasets with one shared mapping so train/val agree on label indices.
    train_dataset = YOLODataset(
        dataset_dir,
        split="train",
        transforms=transform,
        class_mapping=class_mapping,
    )
    val_dataset = YOLODataset(
        dataset_dir,
        split="val",
        transforms=transform,
        class_mapping=class_mapping,
    )

    if len(train_dataset) == 0:
        raise ValueError(
            f"No training images found in '{dataset_dir}'. "
            "Make sure the dataset path is correct."
        )

    # Create dataloaders
    # Note: shuffle=True for train, False for val
    # Collate function handles variable-size images
    def collate_fn(batch):
        return tuple(zip(*batch))


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader
