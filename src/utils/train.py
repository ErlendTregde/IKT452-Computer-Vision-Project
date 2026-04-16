"""
Dataset loading and training loop for EfficientDet.

Label format on disk: YOLO  ->  class_id  cx  cy  w  h  (all normalized 0-1)
effdet target format: bbox [y1, x1, y2, x2] in pixel coords, cls 1-indexed
                      (0 = background / padding in effdet convention)
"""

import os
import glob
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]


class GroceryDataset(Dataset):
    def __init__(self, img_dir: str, label_dir: str, img_size: int = 512):
        self.img_paths = sorted(
            glob.glob(f"{img_dir}/*.jpg") +
            glob.glob(f"{img_dir}/*.jpeg") +
            glob.glob(f"{img_dir}/*.JPG")
        )
        self.label_dir = label_dir
        self.img_size  = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path   = self.img_paths[idx]
        img_tensor = self.transform(Image.open(img_path).convert("RGB"))

        label_path = os.path.join(self.label_dir, Path(img_path).stem + ".txt")
        boxes, classes = [], []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                    # YOLO (cx,cy,w,h normalized) -> pixel (y1,x1,y2,x2) for effdet
                    x1 = (cx - w / 2) * self.img_size
                    y1 = (cy - h / 2) * self.img_size
                    x2 = (cx + w / 2) * self.img_size
                    y2 = (cy + h / 2) * self.img_size
                    boxes.append([y1, x1, y2, x2])
                    classes.append(cls_id + 1)  # effdet: 0=background, 1+=foreground

        return img_tensor, boxes, classes


def collate_fn(batch):
    """Pad boxes and class labels to the same length across the batch."""
    images, all_boxes, all_classes = zip(*batch)
    images = torch.stack(images)

    B        = len(images)
    max_boxes = max((len(b) for b in all_boxes), default=1)

    padded_boxes = torch.zeros(B, max_boxes, 4, dtype=torch.float32)
    padded_cls   = torch.zeros(B, max_boxes, dtype=torch.long)  # 0 = padding

    for i, (boxes, classes) in enumerate(zip(all_boxes, all_classes)):
        n = len(boxes)
        if n > 0:
            padded_boxes[i, :n] = torch.tensor(boxes,   dtype=torch.float32)
            padded_cls[i, :n]   = torch.tensor(classes, dtype=torch.long)

    return images, padded_boxes, padded_cls


def get_dataloaders(dataset_root: str, img_size: int = 512, batch_size: int = 4):
    train_ds = GroceryDataset(
        f"{dataset_root}/train/images",
        f"{dataset_root}/train/labels",
        img_size=img_size,
    )
    val_ds = GroceryDataset(
        f"{dataset_root}/val/images",
        f"{dataset_root}/val/labels",
        img_size=img_size,
    )
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    return train_dl, val_dl


def train_one_epoch(bench, optimizer, dataloader, device, epoch, img_size):
    bench.train()
    total_loss = 0.0

    for step, (images, boxes, classes) in enumerate(dataloader):
        images = images.to(device)
        B      = images.shape[0]

        target = {
            "bbox":      boxes.to(device),
            "cls":       classes.to(device),
            "img_scale": torch.ones(B, dtype=torch.float32, device=device),
            "img_size":  torch.tensor(
                [[img_size, img_size]] * B, dtype=torch.float32, device=device
            ),
        }

        loss_dict = bench(images, target)
        loss      = loss_dict["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bench.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item()

        if step % 10 == 0:
            print(f"  Epoch {epoch} | step {step}/{len(dataloader)}"
                  f" | loss {loss.item():.4f}")

    return total_loss / len(dataloader)
