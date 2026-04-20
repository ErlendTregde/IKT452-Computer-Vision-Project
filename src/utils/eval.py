"""
Shared detection evaluation utilities.

These metrics work with any detector that returns predictions in the common
object-detection format:
    {
        "boxes": Tensor[N, 4],
        "labels": Tensor[N],
        "scores": Tensor[N],
    }

For models with a different inference API, pass a small ``prediction_fn`` that
returns a list of dicts in that format for a batch of images.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import Callable

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou


Prediction = dict[str, torch.Tensor]
PredictionFn = Callable[[torch.nn.Module, list[torch.Tensor]], list[Prediction]]


@contextmanager
def _lowered_score_thresh(model: torch.nn.Module, threshold: float):
    """
    Temporarily lower the model's internal score threshold so that low-confidence
    predictions from an early-stage model are not silently discarded before they
    reach our evaluation code.  Our own score_threshold filter is applied
    afterwards in evaluate_detection_metrics.
    """
    inner = getattr(model, "model", model)           # unwrap FasterRCNN wrapper
    roi_heads = getattr(inner, "roi_heads", None)
    if roi_heads is not None and hasattr(roi_heads, "score_thresh"):
        old = roi_heads.score_thresh
        roi_heads.score_thresh = threshold
        try:
            yield
        finally:
            roi_heads.score_thresh = old
    else:
        yield


def _empty_boxes() -> torch.Tensor:
    return torch.zeros((0, 4), dtype=torch.float32)


def _standardize_target(target: dict) -> Prediction:
    labels = target.get("labels", torch.zeros((0,), dtype=torch.int64))
    return {
        "boxes": target.get("boxes", _empty_boxes()).detach().cpu().to(torch.float32),
        "labels": labels.detach().cpu().to(torch.int64),
        "scores": torch.ones((len(labels),), dtype=torch.float32),
    }


def _standardize_prediction(prediction: dict) -> Prediction:
    boxes = prediction.get("boxes", _empty_boxes()).detach().cpu().to(torch.float32)
    labels = prediction.get("labels", torch.zeros((0,), dtype=torch.int64))
    labels = labels.detach().cpu().to(torch.int64)
    scores = prediction.get("scores", torch.ones((len(labels),), dtype=torch.float32))
    scores = scores.detach().cpu().to(torch.float32)

    if boxes.numel() == 0:
        boxes = _empty_boxes()
    if labels.numel() == 0:
        labels = torch.zeros((0,), dtype=torch.int64)
    if scores.numel() == 0:
        scores = torch.zeros((0,), dtype=torch.float32)

    return {"boxes": boxes, "labels": labels, "scores": scores}


def default_prediction_fn(
    model: torch.nn.Module, images: list[torch.Tensor]
) -> list[Prediction]:
    """
    Run detector inference for models that already return standard predictions.
    """
    outputs = model(images)
    if isinstance(outputs, dict):
        outputs = [outputs]
    if not isinstance(outputs, list):
        raise TypeError(
            "Model inference must return a list of prediction dicts or a single "
            "prediction dict. Pass prediction_fn=... for custom model outputs."
        )
    return [_standardize_prediction(output) for output in outputs]


def _compute_ap(precision: torch.Tensor, recall: torch.Tensor) -> float:
    if precision.numel() == 0 or recall.numel() == 0:
        return 0.0

    precision = torch.cat(
        [torch.tensor([0.0]), precision, torch.tensor([0.0])]
    )
    recall = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])

    for idx in range(precision.numel() - 2, -1, -1):
        precision[idx] = torch.maximum(precision[idx], precision[idx + 1])

    delta = recall[1:] - recall[:-1]
    return float(torch.sum(delta * precision[1:]).item())


def _match_detections(
    gt_by_image: dict[int, torch.Tensor],
    predictions: list[dict[str, torch.Tensor | float | int]],
    iou_threshold: float,
) -> tuple[int, int, int, float]:
    num_gt = sum(len(boxes) for boxes in gt_by_image.values())
    if num_gt == 0:
        return 0, len(predictions), 0, float("nan")
    if not predictions:
        return 0, 0, num_gt, 0.0

    matched = {
        image_id: torch.zeros(len(boxes), dtype=torch.bool)
        for image_id, boxes in gt_by_image.items()
    }

    predictions = sorted(
        predictions, key=lambda pred: float(pred["score"]), reverse=True
    )

    tp = []
    fp = []

    for pred in predictions:
        image_id = int(pred["image_id"])
        pred_box = pred["box"].unsqueeze(0)
        gt_boxes = gt_by_image.get(image_id, _empty_boxes())

        if len(gt_boxes) == 0:
            tp.append(0.0)
            fp.append(1.0)
            continue

        ious = box_iou(pred_box, gt_boxes).squeeze(0)
        best_iou, best_idx = torch.max(ious, dim=0)

        if best_iou >= iou_threshold and not matched[image_id][best_idx]:
            matched[image_id][best_idx] = True
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)

    tp_tensor = torch.tensor(tp, dtype=torch.float32)
    fp_tensor = torch.tensor(fp, dtype=torch.float32)
    cum_tp = torch.cumsum(tp_tensor, dim=0)
    cum_fp = torch.cumsum(fp_tensor, dim=0)

    precision = cum_tp / torch.clamp(cum_tp + cum_fp, min=1e-8)
    recall = cum_tp / max(num_gt, 1)
    ap = _compute_ap(precision, recall)

    total_tp = int(cum_tp[-1].item()) if len(cum_tp) > 0 else 0
    total_fp = int(cum_fp[-1].item()) if len(cum_fp) > 0 else 0
    total_fn = num_gt - total_tp

    return total_tp, total_fp, total_fn, ap


def evaluate_detection_metrics(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    score_threshold: float = 0.01,
    iou_thresholds: list[float] | None = None,
    pr_iou_threshold: float = 0.5,
    prediction_fn: PredictionFn | None = None,
) -> dict[str, float | dict[int, dict[str, float]]]:
    """
    Evaluate a detector with common detection metrics.

    Returns aggregate metrics plus per-class metrics:
        - mAP@0.50
        - mAP@0.50:0.95
        - precision
        - recall
        - f1
        - per_class
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5 + 0.05 * idx for idx in range(10)]

    if prediction_fn is None:
        prediction_fn = default_prediction_fn

    was_training = model.training
    model.eval()
    model.to(device)

    gt_by_class: dict[int, dict[int, torch.Tensor]] = defaultdict(dict)
    predictions_by_class: dict[int, list[dict[str, torch.Tensor | float | int]]] = (
        defaultdict(list)
    )
    fallback_image_id = 0

    # Lower the model's internal score_thresh so early-stage models (whose
    # per-class softmax scores are spread thinly) can still produce candidates.
    # Our own score_threshold filter is applied below after collecting outputs.
    with _lowered_score_thresh(model, threshold=0.0), torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = prediction_fn(model, images)
            if len(outputs) != len(targets):
                raise ValueError(
                    "Prediction count does not match batch size. "
                    "Make sure prediction_fn returns one prediction dict per image."
                )

            for output, target in zip(outputs, targets):
                target_std = _standardize_target(target)
                output_std = _standardize_prediction(output)
                image_id_tensor = target.get("image_id")
                if image_id_tensor is not None:
                    image_id = int(image_id_tensor.reshape(-1)[0].item())
                else:
                    image_id = fallback_image_id
                    fallback_image_id += 1

                gt_labels = target_std["labels"]
                gt_boxes = target_std["boxes"]
                for class_id in gt_labels.unique().tolist():
                    class_mask = gt_labels == class_id
                    gt_by_class[int(class_id)][image_id] = gt_boxes[class_mask]

                pred_scores = output_std["scores"]
                pred_keep = pred_scores >= score_threshold
                pred_labels = output_std["labels"][pred_keep]
                pred_boxes = output_std["boxes"][pred_keep]
                pred_scores = pred_scores[pred_keep]

                for pred_box, pred_label, pred_score in zip(
                    pred_boxes, pred_labels, pred_scores
                ):
                    predictions_by_class[int(pred_label.item())].append(
                        {
                            "image_id": image_id,
                            "box": pred_box,
                            "score": float(pred_score.item()),
                        }
                    )

    if was_training:
        model.train()

    class_ids = sorted(set(gt_by_class.keys()) | set(predictions_by_class.keys()))
    if not class_ids:
        return {
            "mAP@0.50": 0.0,
            "mAP@0.50:0.95": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "per_class": {},
        }

    map50_scores = []
    map5095_scores = []
    per_class: dict[int, dict[str, float]] = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for class_id in class_ids:
        gt_for_class = gt_by_class.get(class_id, {})
        preds_for_class = predictions_by_class.get(class_id, [])
        ap_scores = []
        ap50_value = 0.0

        pr_tp, pr_fp, pr_fn, _ = _match_detections(
            gt_for_class, preds_for_class, pr_iou_threshold
        )
        total_tp += pr_tp
        total_fp += pr_fp
        total_fn += pr_fn

        for iou_threshold in iou_thresholds:
            _, _, _, ap = _match_detections(gt_for_class, preds_for_class, iou_threshold)
            if ap == ap:
                ap_scores.append(ap)
                if iou_threshold == 0.5:
                    ap50_value = ap
                    map50_scores.append(ap)

        precision = pr_tp / (pr_tp + pr_fp) if (pr_tp + pr_fp) > 0 else 0.0
        recall = pr_tp / (pr_tp + pr_fn) if (pr_tp + pr_fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        if ap_scores:
            map5095_scores.append(sum(ap_scores) / len(ap_scores))

        per_class[class_id] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "AP@0.50": ap50_value,
            "AP@0.50:0.95": sum(ap_scores) / len(ap_scores) if ap_scores else 0.0,
            "support": float(sum(len(boxes) for boxes in gt_for_class.values())),
        }

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "mAP@0.50": sum(map50_scores) / len(map50_scores) if map50_scores else 0.0,
        "mAP@0.50:0.95": (
            sum(map5095_scores) / len(map5095_scores) if map5095_scores else 0.0
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_class": per_class,
    }
