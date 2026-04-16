"""
Detection metrics for EfficientDet on the grocery dataset.

Metrics computed:
  mAP@0.50       -- COCO-style, IoU threshold 0.50
  mAP@0.50:0.95  -- COCO-style, IoU thresholds 0.50..0.95 (primary metric)
  Precision      -- TP / (TP + FP) at IoU=0.5, confidence=0.3
  Recall         -- TP / (TP + FN) at IoU=0.5, confidence=0.3
  F1             -- 2 * P * R / (P + R)

Class matching is required for TP (prediction class must equal GT class).
"""

import torch
from torchvision.ops import box_iou
from torchmetrics.detection import MeanAveragePrecision
from effdet.bench import DetBenchPredict


def evaluate(model, dataloader, device, img_size, score_threshold=0.3):
    """
    Run inference on the validation set and return all detection metrics.

    Args:
        model:            Base EfficientDet model (not DetBenchTrain).
        dataloader:       Validation DataLoader.
        device:           torch.device.
        img_size:         Input image size in pixels (e.g. 512 for D0).
        score_threshold:  Confidence cut-off for Precision / Recall / F1.
                          mAP is always computed over all score levels.

    Returns:
        dict with keys: mAP@0.50, mAP@0.50:0.95, Precision, Recall, F1
    """
    pred_bench = DetBenchPredict(model).to(device)
    pred_bench.eval()

    metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy")

    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for images, boxes, classes in dataloader:
            images = images.to(device)
            B      = images.shape[0]

            img_info = {
                "img_scale": torch.ones(B, dtype=torch.float32, device=device),
                "img_size":  torch.tensor(
                    [[img_size, img_size]] * B, dtype=torch.float32, device=device
                ),
            }

            # detections: [B, max_det, 6]  ->  x1, y1, x2, y2, score, class (0-indexed)
            detections = pred_bench(images, img_info)

            for i in range(B):
                dets = detections[i]                    # [max_det, 6]
                valid = dets[:, 4] > 0                  # score > 0 means real detection
                dets  = dets[valid]

                if len(dets) > 0:
                    pred_boxes  = dets[:, :4].cpu()
                    pred_scores = dets[:, 4].cpu()
                    pred_labels = dets[:, 5].long().cpu()   # already 0-indexed
                else:
                    pred_boxes  = torch.zeros(0, 4)
                    pred_scores = torch.zeros(0)
                    pred_labels = torch.zeros(0, dtype=torch.long)

                all_preds.append({
                    "boxes":  pred_boxes,
                    "scores": pred_scores,
                    "labels": pred_labels,
                })

                # GT: stored as [y1,x1,y2,x2] pixel coords, 1-indexed class
                gt_raw = boxes[i]    # [max_boxes, 4]
                gt_cls = classes[i]  # [max_boxes]

                valid_gt = gt_cls > 0          # 0 = padding slot
                gt_b = gt_raw[valid_gt]
                gt_c = (gt_cls[valid_gt] - 1)  # 1-indexed -> 0-indexed

                # Reorder [y1,x1,y2,x2] -> [x1,y1,x2,y2] for torchmetrics
                if len(gt_b) > 0:
                    gt_boxes_xy = gt_b[:, [1, 0, 3, 2]]
                else:
                    gt_boxes_xy = torch.zeros(0, 4)

                all_targets.append({
                    "boxes":  gt_boxes_xy.cpu(),
                    "labels": gt_c.long().cpu(),
                })

    # mAP via torchmetrics (handles all IoU thresholds internally)
    metric.update(all_preds, all_targets)
    map_result = metric.compute()

    precision, recall, f1 = _prf1(
        all_preds, all_targets, iou_thr=0.5, score_thr=score_threshold
    )

    return {
        "mAP@0.50":      map_result["map_50"].item(),
        "mAP@0.50:0.95": map_result["map"].item(),
        "Precision":     precision,
        "Recall":        recall,
        "F1":            f1,
    }


def _prf1(preds, targets, iou_thr=0.5, score_thr=0.5):
    """
    Compute Precision, Recall, F1 at a fixed IoU and confidence threshold.
    A prediction is TP only if:
      - IoU with an unmatched GT box >= iou_thr
      - Predicted class == GT class
    """
    tp = fp = fn = 0

    for pred, target in zip(preds, targets):
        mask = pred["scores"] >= score_thr
        pb   = pred["boxes"][mask]
        pl   = pred["labels"][mask]
        gb   = target["boxes"]
        gl   = target["labels"]

        n_pred = len(pb)
        n_gt   = len(gb)

        if n_pred == 0 and n_gt == 0:
            continue
        if n_pred == 0:
            fn += n_gt
            continue
        if n_gt == 0:
            fp += n_pred
            continue

        iou_matrix = box_iou(pb, gb)   # [n_pred, n_gt]
        matched_gt = set()

        for i in range(n_pred):
            best_iou = iou_matrix[i].max().item()
            best_j   = iou_matrix[i].argmax().item()

            if (best_iou >= iou_thr
                    and best_j not in matched_gt
                    and pl[i].item() == gl[best_j].item()):
                tp += 1
                matched_gt.add(best_j)
            else:
                fp += 1

        fn += n_gt - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1
