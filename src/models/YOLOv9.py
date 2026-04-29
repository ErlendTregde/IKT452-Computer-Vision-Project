from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import yaml

def _collect_class_ids(label_root: Path) -> list[int]:
    class_ids = set()
    for label_path in label_root.glob("**/*.txt"):
        with label_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if parts:
                    class_ids.add(int(parts[0]))
    return sorted(class_ids)


class YOLOv9:
    DEFAULT_WEIGHTS = "yolov9m.pt"

    def __init__(self, weights: str = DEFAULT_WEIGHTS):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "YOLOv9 training requires the 'ultralytics' package. "
                "Install it with `uv add ultralytics`."
            ) from exc

        self.weights = weights
        self.model = YOLO(weights)

    def parameter_counts(self) -> tuple[int, int]:
        """Return total and trainable parameter counts for the loaded model."""
        inner = getattr(self.model, "model", self.model)
        total_params = sum(param.numel() for param in inner.parameters())
        trainable_params = sum(param.numel() for param in inner.parameters() if param.requires_grad)
        if trainable_params == 0:
            trainable_params = total_params
        return total_params, trainable_params

    def train_model(
        self,
        dataset_dir: str = "dataset",
        epochs: int = 50,
        batch_size: int = 4,
        imgsz: int = 800,
        device: str | int = "cpu",
        checkpoint_dir: str = "checkpoints_yolov9",
        log_path: str = "logs/old/yolov9.log",
        lr0: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        workers: int = 2,
        patience: int = 0,
        resume: str | None = None,
    ) -> Path:
        """
        Train YOLOv9 on the project dataset and write a repo-local metrics log.

        Returns:
            The path to the generated log file.
        """
        checkpoint_root = Path(checkpoint_dir).resolve()
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        data_config_path = self.prepare_data_config(dataset_dir, checkpoint_root)

        train_kwargs: dict[str, Any] = {
            "data": str(data_config_path),
            "epochs": epochs,
            "batch": batch_size,
            "imgsz": imgsz,
            "device": device,
            "workers": workers,
            "project": str(checkpoint_root.parent),
            "name": checkpoint_root.name,
            "exist_ok": True,
            "optimizer": "SGD",
            "lr0": lr0,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "patience": patience if patience > 0 else epochs,
            "plots": True,
            "save": True,
            "verbose": True,
            "pretrained": True,
        }
        if resume:
            train_kwargs["resume"] = resume

        results = self.model.train(**train_kwargs)
        save_dir = self._resolve_save_dir(results, checkpoint_root)
        log_file = self.write_training_log(
            results_csv=save_dir / "results.csv",
            log_path=Path(log_path),
            save_dir=save_dir,
            num_epochs=epochs,
        )
        return log_file

    @staticmethod
    def prepare_data_config(dataset_dir: str, output_dir: str | Path) -> Path:
        """Create a YOLO dataset YAML file from the repo's train/val layout."""
        dataset_root = Path(dataset_dir).resolve()
        class_ids = _collect_class_ids(dataset_root)
        expected_ids = list(range(len(class_ids)))
        if class_ids != expected_ids:
            raise ValueError(
                "YOLOv9 expects contiguous class ids starting at 0. "
                f"Found ids: {class_ids!r}"
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "yolov9_data.yaml"
        config = {
            "path": str(dataset_root),
            "train": "train/images",
            "val": "val/images",
            "nc": len(class_ids),
            "names": [f"class_{class_id}" for class_id in class_ids],
        }
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        return config_path

    @staticmethod
    def write_training_log(
        results_csv: Path,
        log_path: Path,
        save_dir: Path,
        num_epochs: int,
    ) -> Path:
        """Convert Ultralytics CSV metrics into the project's plain-text log format."""
        if not results_csv.exists():
            raise FileNotFoundError(f"Expected Ultralytics metrics at '{results_csv}'.")

        log_path.parent.mkdir(parents=True, exist_ok=True)
        rows = YOLOv9._read_results_rows(results_csv)

        best_map50 = max(
            (YOLOv9._metric_value(row, ["metrics/mAP50(B)", "metrics/mAP50"]) for row in rows),
            default=0.0,
        )
        best_map50_95 = max(
            (
                YOLOv9._metric_value(row, ["metrics/mAP50-95(B)", "metrics/mAP50-95"])
                for row in rows
            ),
            default=0.0,
        )

        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(f"Results directory: {save_dir}\n")
            for row in rows:
                epoch = int(float(row.get("epoch", 0)))
                train_loss = YOLOv9._summed_loss(
                    row,
                    [
                        ["train/box_loss", "train/box_loss(B)"],
                        ["train/cls_loss", "train/cls_loss(B)"],
                        ["train/dfl_loss", "train/dfl_loss(B)"],
                        ["train/obj_loss", "train/obj_loss(B)"],
                    ],
                )
                val_loss = YOLOv9._summed_loss(
                    row,
                    [
                        ["val/box_loss", "val/box_loss(B)"],
                        ["val/cls_loss", "val/cls_loss(B)"],
                        ["val/dfl_loss", "val/dfl_loss(B)"],
                        ["val/obj_loss", "val/obj_loss(B)"],
                    ],
                )
                precision = YOLOv9._metric_value(row, ["metrics/precision(B)", "metrics/precision"])
                recall = YOLOv9._metric_value(row, ["metrics/recall(B)", "metrics/recall"])
                map_50 = YOLOv9._metric_value(row, ["metrics/mAP50(B)", "metrics/mAP50"])
                map_50_95 = YOLOv9._metric_value(row, ["metrics/mAP50-95(B)", "metrics/mAP50-95"])
                f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

                handle.write(f"\n{'=' * 50}\n")
                handle.write(f"Epoch {epoch}/{num_epochs}\n")
                handle.write(f"{'=' * 50}\n")
                handle.write(f"Train Loss: {train_loss:.4f}\n")
                handle.write(f"Val Loss: {val_loss:.4f}\n")
                handle.write(
                    f"mAP@0.50={map_50:.4f}, "
                    f"mAP@0.50:0.95={map_50_95:.4f}, "
                    f"Precision={precision:.4f}, "
                    f"Recall={recall:.4f}, "
                    f"F1={f1:.4f}\n"
                )

            handle.write(f"\nTraining complete. Best mAP@0.50: {best_map50:.4f}\n")
            handle.write(f"Best mAP@0.50:0.95: {best_map50_95:.4f}\n")

        return log_path

    @staticmethod
    def _resolve_save_dir(results: Any, checkpoint_root: Path) -> Path:
        """Find the Ultralytics run directory from the training result."""
        save_dir = getattr(results, "save_dir", None)
        if save_dir is None:
            save_dir = getattr(getattr(results, "trainer", None), "save_dir", None)
        if save_dir is None:
            save_dir = checkpoint_root
        return Path(save_dir)

    @staticmethod
    def _read_results_rows(results_csv: Path) -> list[dict[str, str]]:
        with results_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows: list[dict[str, str]] = []
            for row in reader:
                cleaned = {
                    (key or "").strip(): (value or "").strip()
                    for key, value in row.items()
                }
                rows.append(cleaned)
        return rows

    @staticmethod
    def _metric_value(row: dict[str, str], candidates: list[str]) -> float:
        for key in candidates:
            value = row.get(key)
            if value:
                return float(value)
        return 0.0

    @staticmethod
    def _summed_loss(row: dict[str, str], groups: list[list[str]]) -> float:
        return sum(YOLOv9._metric_value(row, candidates) for candidates in groups)
