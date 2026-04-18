"""Command-line argument parsing."""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train object detection model on a YOLO-format dataset")

    parser.add_argument("--model", default="faster-rcnn",
                        choices=["faster-rcnn", "fpn", "efficientdet", "rtdetr"],
                        help="model architecture to train (default: faster-rcnn)")
    parser.add_argument("--dataset", default="dataset", metavar="PATH",
                        help="path to dataset root (default: dataset)")
    parser.add_argument("--epochs", type=int, default=10, metavar="N",
                        help="number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=4, metavar="N",
                        help="images per batch (default: 4)")
    parser.add_argument("--lr", type=float, default=0.001, metavar="LR",
                        help="initial learning rate (default: 0.001)")
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"],
                        help="optimizer (default: sgd)")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="SGD momentum (default: 0.9)")
    parser.add_argument("--weight-decay", type=float, default=0.0005,
                        help="weight decay (default: 0.0005)")
    parser.add_argument("--num-workers", type=int, default=2, metavar="N",
                        help="dataloader worker processes (default: 2)")
    parser.add_argument("--checkpoint-dir", default="checkpoints", metavar="DIR",
                        help="directory to write checkpoints (default: checkpoints)")
    parser.add_argument("--resume", default=None, metavar="PATH",
                        help="resume training from a checkpoint file")
    parser.add_argument("--no-metrics", action="store_true",
                        help="skip detection metric computation each epoch (faster)")
    parser.add_argument("--score-threshold", type=float, default=0.01, metavar="T",
                        help="minimum score for detections in metric evaluation (default: 0.01)")

    return parser.parse_args()
