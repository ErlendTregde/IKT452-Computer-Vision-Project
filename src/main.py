import torch

# Swap this import to switch models:
#   faster_rcnn  — plain ResNet-50 backbone, no FPN (baseline)
#   fpn_rcnn     — ResNet-50 + Feature Pyramid Network
from models.faster_rcnn import FasterRCNN
from utils.args import parse_args
from utils.dataset import create_dataloaders
from utils.train import train


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        print("CUDA not available — training on CPU.")
        return torch.device("cpu")

    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) < (7, 5):
        name = torch.cuda.get_device_name(0)
        print(f"GPU {name} (sm_{major}{minor}) not supported by this PyTorch build — using CPU.")
        return torch.device("cpu")

    return torch.device("cuda")


def main():
    args = parse_args()
    device = get_device()

    train_loader, val_loader = create_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    num_classes = train_loader.dataset.num_classes  # type: ignore

    model = FasterRCNN(num_classes=num_classes)

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        device=device,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        compute_detection_metrics=not args.no_metrics,
        metric_score_threshold=args.score_threshold,
    )


if __name__ == "__main__":
    main()
