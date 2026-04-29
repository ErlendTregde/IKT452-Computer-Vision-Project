# Automated Grocery Product Detection in Retail Environments

**IKT452 Computer Vision — University of Agder**

Erlend Tregde · Sander Wesstøl

---

## Overview

This project investigates automated grocery product detection in retail shelf images, using **Deep Residual Learning** (ResNet — CVPR 2016 Best Paper) as a foundation. Four object detection architectures are implemented and compared on the [NorgesGruppen dataset](https://app.ainm.no/docs/norgesgruppen-data/overview) of 248 real shelf images spanning 356 product categories (~22,700 annotated bounding boxes).

The core finding: D4 dihedral data augmentation (8× training set expansion) was essential for all models on this small, fine-grained dataset, with **YOLOv9 + augmentation** achieving the best overall result (mAP@0.50 = **0.7155**, Precision = **0.7631**).

Full details are in [`docs/Semester_Assignment.pdf`](docs/Semester_Assignment.pdf).

---

## Results

| Model | mAP@0.50 | mAP@0.50:0.95 | Precision | Recall | F1 |
|---|---|---|---|---|---|
| FPN | 0.4403 | 0.2555 | 0.5500 | 0.6289 | 0.5868 |
| FPN + Aug | 0.6080 | 0.3577 | 0.6444 | **0.7213** | 0.6807 |
| Faster R-CNN | 0.1992 | 0.0783 | 0.3519 | 0.4023 | 0.3754 |
| Faster R-CNN + Aug | 0.5946 | 0.3115 | 0.6202 | 0.6904 | 0.6534 |
| EfficientDet | 0.0100 | 0.0028 | 0.0100 | 0.2064 | 0.0192 |
| EfficientDet + Aug | 0.4208 | 0.2164 | 0.3702 | 0.6304 | 0.4665 |
| YOLOv9 | 0.5255 | 0.3467 | 0.5945 | 0.5220 | 0.5559 |
| **YOLOv9 + Aug** | **0.7155** | **0.4571** | **0.7631** | 0.6451 | **0.6992** |

All models share a ResNet-50 backbone (~40M parameters) and were trained for 50 epochs with AdamW (lr = 5×10⁻⁴, cosine annealing).

---

## Models

All four architectures use a ResNet-50 backbone pretrained on ImageNet.

| Architecture | File | Description |
|---|---|---|
| **Faster R-CNN** | `src/models/faster_rcnn.py` | Two-stage detector; plain ResNet-50 backbone with a 1×1 projection (2048→256ch) to keep parameter count comparable to FPN |
| **FPN** | `src/models/fpn_rcnn.py` | Faster R-CNN with a Feature Pyramid Network; builds multi-scale feature maps (P2–P6) for better detection across object sizes |
| **EfficientDet** | `src/models/efficientdet.py` | Single-stage detector with a 4-layer BiFPN (256ch) over P4–P7; uses focal loss and depthwise-separable prediction heads |
| **YOLOv9** | `src/models/YOLOv9.py` | Real-time detector combining Programmable Gradient Information (PGI) and GELAN; fine-tuned via the Ultralytics framework |

---

## Dataset

The dataset was provided by NorgesGruppen as part of the [Norwegian AI Championship 2026](https://app.ainm.no/docs/norgesgruppen-data/overview).

```
dataset/
├── train/
│   ├── images/    # 211 original + 1477 augmented = 1688 total
│   └── labels/    # YOLO format: <class_id> <cx> <cy> <w> <h> (normalised)
└── val/
    ├── images/    # 37 images
    └── labels/
```

- **356 product classes**, averaging ~0.6 training images per class before augmentation
- Labels are in YOLO format (normalised bounding boxes)

### Data Augmentation

To address the very small training set, all 7 non-trivial symmetries of the **D4 dihedral group** were applied, expanding the training split 8× (211 → 1688 images):

| Transform | Description |
|---|---|
| `flip_h` | Horizontal mirror |
| `flip_v` | Vertical mirror |
| `rot90` | 90° clockwise |
| `rot180` | 180° rotation |
| `rot270` | 270° clockwise |
| `rot90_flip_h` | 90° CW + horizontal mirror |
| `rot90_flip_v` | 90° CW + vertical mirror |

Bounding boxes are transformed analytically — no interpolation artefacts are introduced.

```bash
uv run src/utils/augment_dataset.py               # augments dataset/train in-place
uv run src/utils/augment_dataset.py --dataset path/to/dataset
```

---

## Project Structure

```
├── src/
│   ├── demo.ipynb              # Inference demo — all 4 models on one validation image
│   ├── train.py                # Training entry point
│   ├── models/
│   │   ├── faster_rcnn.py      # Faster R-CNN (plain ResNet-50)
│   │   ├── fpn_rcnn.py         # Faster R-CNN + FPN
│   │   ├── efficientdet.py     # EfficientDet (BiFPN)
│   │   └── YOLOv9.py           # YOLOv9 wrapper
│   └── utils/
│       ├── augment_dataset.py  # D4 dihedral augmentation
│       ├── dataset.py          # YOLO-format dataset loader + class mapping
│       ├── train.py            # Training loop
│       ├── validate.py         # Validation loop
│       ├── eval.py             # mAP / precision / recall computation
│       ├── checkpoint.py       # Save / load checkpoints
│       ├── args.py             # CLI argument parsing
│       ├── display.py          # Visualisation utilities
│       └── plot.py             # Training curve plotting
├── dataset/                    # NorgesGruppen shelf images + YOLO labels
├── docs/
│   └── Semester_Assignment.pdf # Full project report (IEEE format)
├── results/                    # Saved inference demo images
├── logs/                       # Training logs
└── pyproject.toml              # uv/pip dependencies
```

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install dependencies
uv sync

# Register the environment as a Jupyter kernel (for demo.ipynb)
uv add ipykernel
uv run python -m ipykernel install --user --name ikt452 --display-name "IKT452 (uv)"
```

Alternatively, install with pip:

```bash
pip install -r requirements.txt
```

**Requirements** (see `pyproject.toml`): Python ≥ 3.13, PyTorch, torchvision, Ultralytics, Pillow, matplotlib, huggingface-hub.

---

## Inference Demo

Open `src/demo.ipynb` and select the **IKT452 (uv)** kernel. Running all cells will:

1. Download all four model checkpoints from HuggingFace (`ErlendTregde/YOLOv9-Norgesgruppen`)
2. Load the first validation image and its ground-truth bounding boxes
3. Run inference with each model and display a side-by-side comparison (ground truth vs predictions)
4. Report per-model precision and recall at IoU ≥ 0.5
5. Save the output figures to `results/`

All checkpoints are loaded from `aug_checkpoints/` (the augmentation-trained versions).

---

## Training

```bash
# Train a specific model
uv run src/train.py --model faster-rcnn --dataset dataset --epochs 50
uv run src/train.py --model fpn         --dataset dataset --epochs 50
uv run src/train.py --model efficientdet --dataset dataset --epochs 50
uv run src/train.py --model yolov9      --dataset dataset --epochs 50

# Resume from a checkpoint
uv run src/train.py --model fpn --resume checkpoints/epoch_30.pth

# Key options
#   --batch-size N       images per batch (default: 4)
#   --lr LR              initial learning rate (default: 0.001)
#   --optimizer adamw    use AdamW instead of SGD
#   --patience N         early stopping patience (0 = disabled)
#   --no-metrics         skip per-epoch mAP (faster training)
```

Checkpoints are written to `checkpoints/` by default; the best epoch is saved as `best_model.pth`.

---

## Pretrained Checkpoints (HuggingFace)

All checkpoints trained with D4 augmentation are available at  
**[ErlendTregde/YOLOv9-Norgesgruppen](https://huggingface.co/ErlendTregde/YOLOv9-Norgesgruppen)**

| Model | Path in repo |
|---|---|
| YOLOv9 | `aug_checkpoints/YOLOv9/best.pt` |
| Faster R-CNN | `aug_checkpoints/faster-rcnn/best_model.pth` |
| FPN | `aug_checkpoints/fpn/best_model.pth` |
| EfficientDet | `aug_checkpoints/efficientdet/best_model.pth` |

Older checkpoints (trained without augmentation) are in `old_checkpoints/`.

---

## References

1. He et al., "Deep Residual Learning for Image Recognition," *CVPR* 2016 (Best Paper)
2. Lin et al., "Feature Pyramid Networks for Object Detection," *CVPR* 2017
3. Tan et al., "EfficientDet: Scalable and Efficient Object Detection," *CVPR* 2020
4. Wang et al., "YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information," *arXiv* 2024
5. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks," *ICCV* 2017
