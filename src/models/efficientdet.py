"""
EfficientDet with ResNet-50 backbone.

BiFPN multi-scale feature fusion [Tan et al., 2020] with ResNet-50 instead of
EfficientNet, keeping total parameter count ~40M for fair comparison.

Architecture:
  ResNet-50 → C4/C5 → project to 256ch → P6/P7 → 4× BiFPN (4 levels P4-P7)
  → class + box heads → anchor decoding

Anchor design: 9 anchors per location (3 aspect ratios × 3 scale octaves),
P4 (stride 16) – P7 (stride 128).  Dropping P3 (stride 8) avoids the 150K-anchor
grid that causes OOM at 800px input resolution.

Parameter breakdown (~40M):
  ResNet-50 backbone                 23.5 M
  C4/C5/P6 projections (→256ch)       1.3 M
  BiFPN  (4 layers, 256ch, std conv) 14.2 M
  Class + box heads (4 dw-sep layers)  0.6 M
  Total                              ~39.6 M
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import box_iou, clip_boxes_to_image, batched_nms


def _focal_loss(inputs: torch.Tensor, targets: torch.Tensor,
                alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """
    Focal loss with stop-gradient on the focal weight.

    Torchvision's sigmoid_focal_loss keeps every intermediate tensor alive for
    backprop (p, p_t, alpha_t, …), which is expensive.  Computing the weight
    inside no_grad() means only one extra tensor lives on GPU; BCE gradients
    flow as normal because binary_cross_entropy_with_logits is differentiable.
    """
    with torch.no_grad():
        p   = inputs.sigmoid()
        p_t = p * targets + (1 - p) * (1 - targets)
        w   = (alpha * targets + (1 - alpha) * (1 - targets)) * (1 - p_t).pow(gamma)
    return F.binary_cross_entropy_with_logits(inputs, targets, weight=w, reduction='sum')


# ── Anchor config ─────────────────────────────────────────────────────────────

_STRIDES       = [16, 32, 64, 128]          # P4, P5, P6, P7
_SCALES        = [1.0, 2 ** (1/3), 2 ** (2/3)]
_ASPECT_RATIOS = [0.5, 1.0, 2.0]
_NUM_ANCHORS   = len(_SCALES) * len(_ASPECT_RATIOS)  # 9


# ── Building blocks ───────────────────────────────────────────────────────────

class _DWSConv(nn.Module):
    """Depthwise-separable conv + BN + SiLU (used in prediction heads)."""
    def __init__(self, channels: int):
        super().__init__()
        self.dw  = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pw  = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn  = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class _BiFPNNode(nn.Module):
    """Fuse N same-resolution feature maps (standard 3×3 conv for parameter density)."""
    def __init__(self, channels: int, num_inputs: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs))
        # Standard conv: ~590K params per node at 256ch → hits ~40M at 256ch width
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: list) -> torch.Tensor:
        w = F.relu(self.weights)
        w = w / (w.sum() + 1e-4)
        target = inputs[0].shape[-2:]
        aligned = [
            F.interpolate(x, size=target, mode='nearest') if x.shape[-2:] != target else x
            for x in inputs
        ]
        return self.conv(sum(w[i] * x for i, x in enumerate(aligned)))


class _BiFPNLayer(nn.Module):
    """
    One BiFPN layer over 4 levels (P4–P7).
    Top-down: P7 → P6 → P5 → P4.
    Bottom-up: P4 → P5 → P6 → P7.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.td_p6 = _BiFPNNode(channels, 2)
        self.td_p5 = _BiFPNNode(channels, 2)
        self.td_p4 = _BiFPNNode(channels, 2)   # final P4
        self.bu_p5 = _BiFPNNode(channels, 3)
        self.bu_p6 = _BiFPNNode(channels, 3)
        self.bu_p7 = _BiFPNNode(channels, 2)   # final P7

    def forward(self, features: list) -> list:
        p4, p5, p6, p7 = features

        p6_td  = self.td_p6([p6, F.interpolate(p7, size=p6.shape[-2:], mode='nearest')])
        p5_td  = self.td_p5([p5, F.interpolate(p6_td, size=p5.shape[-2:], mode='nearest')])
        p4_out = self.td_p4([p4, F.interpolate(p5_td, size=p4.shape[-2:], mode='nearest')])

        p5_out = self.bu_p5([p5, p5_td, F.max_pool2d(p4_out, 2, stride=2, ceil_mode=True)])
        p6_out = self.bu_p6([p6, p6_td, F.max_pool2d(p5_out, 2, stride=2, ceil_mode=True)])
        p7_out = self.bu_p7([p7, F.max_pool2d(p6_out, 2, stride=2, ceil_mode=True)])

        return [p4_out, p5_out, p6_out, p7_out]


class _PredictionHead(nn.Module):
    """
    Shared class or box head across all BiFPN levels.
    Weights shared across levels; BN is per-level (EfficientDet practice).
    """
    def __init__(self, channels: int, num_layers: int, out_channels: int, num_levels: int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.dw_convs = nn.ModuleList(
            [nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
             for _ in range(num_layers)]
        )
        self.pw_convs = nn.ModuleList(
            [nn.Conv2d(channels, channels, 1, bias=False) for _ in range(num_layers)]
        )
        self.bns = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(channels) for _ in range(num_levels)])
             for _ in range(num_layers)]
        )
        self.act       = nn.SiLU(inplace=True)
        self.predictor = nn.Conv2d(channels, out_channels, 1)

        if out_channels > 4:
            prior_prob = 0.01
            nn.init.constant_(self.predictor.bias, -math.log((1 - prior_prob) / prior_prob))

    def forward(self, features: list) -> list:
        results = []
        for lvl, f in enumerate(features):
            for i in range(self.num_layers):
                f = self.act(self.bns[i][lvl](self.pw_convs[i](self.dw_convs[i](f))))
            results.append(self.predictor(f))
        return results


# ── Anchor helpers ────────────────────────────────────────────────────────────

def _make_level_anchors(
    fh: int, fw: int, stride: int,
    scales: list = _SCALES,
    aspect_ratios: list = _ASPECT_RATIOS,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    Anchors for one feature level, shape (fh*fw*num_anchors, 4).

    Ordering must match the prediction tensor produced by
      pred.permute(1, 2, 0).reshape(-1, C)   # (H,W,C) → (H*W*num_anchors, per_anchor_C)
    which has anchor type varying FASTEST within each spatial location.
    We achieve this by computing (fh*fw, num_anchors, 4) and then flattening,
    so index = spatial_idx * num_anchors + anchor_type_idx.
    """
    # Per-anchor type offsets relative to cell centre, shape (num_anchors, 4)
    offsets = []
    for s in scales:
        area = (stride * s) ** 2
        for ar in aspect_ratios:
            w = math.sqrt(area / ar)
            h = w * ar
            offsets.append([-w / 2, -h / 2, w / 2, h / 2])
    offsets = torch.tensor(offsets, device=device, dtype=torch.float32)  # (A, 4)

    cx = (torch.arange(fw, device=device, dtype=torch.float32) + 0.5) * stride
    cy = (torch.arange(fh, device=device, dtype=torch.float32) + 0.5) * stride
    grid_y, grid_x = torch.meshgrid(cy, cx, indexing='ij')
    # centres: (fh*fw, 1, 4)
    centres = torch.stack([grid_x, grid_y, grid_x, grid_y], dim=-1).reshape(-1, 1, 4)
    # broadcast → (fh*fw, num_anchors, 4), then flatten to (fh*fw*num_anchors, 4)
    return (centres + offsets.unsqueeze(0)).reshape(-1, 4)


def _encode_boxes(anchors: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]
    ax = (anchors[:, 0] + anchors[:, 2]) / 2
    ay = (anchors[:, 1] + anchors[:, 3]) / 2
    gw = gt[:, 2] - gt[:, 0]
    gh = gt[:, 3] - gt[:, 1]
    gx = (gt[:, 0] + gt[:, 2]) / 2
    gy = (gt[:, 1] + gt[:, 3]) / 2
    return torch.stack([(gx - ax) / aw, (gy - ay) / ah,
                        torch.log(gw / aw), torch.log(gh / ah)], dim=1)


def _decode_boxes(anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]
    ax = (anchors[:, 0] + anchors[:, 2]) / 2
    ay = (anchors[:, 1] + anchors[:, 3]) / 2
    gx = deltas[:, 0] * aw + ax
    gy = deltas[:, 1] * ah + ay
    gw = aw * deltas[:, 2].clamp(-4, 4).exp()
    gh = ah * deltas[:, 3].clamp(-4, 4).exp()
    return torch.stack([gx - gw/2, gy - gh/2, gx + gw/2, gy + gh/2], dim=1)


# ── Main model ────────────────────────────────────────────────────────────────

class EfficientDet(nn.Module):

    def __init__(
        self,
        num_classes: int,
        pretrained_backbone: bool = True,
        bifpn_channels: int = 256,
        bifpn_layers: int = 4,
        head_depth: int = 4,
        min_size: int = 800,
        max_size: int = 1333,
    ):
        super().__init__()
        self.num_classes = num_classes
        C = bifpn_channels

        self.transform = GeneralizedRCNNTransform(
            min_size=min_size, max_size=max_size,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )

        # ── Backbone (ResNet-50) ──
        r50 = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained_backbone else None)
        self.bb_stem   = nn.Sequential(r50.conv1, r50.bn1, r50.relu, r50.maxpool)
        self.bb_layer1 = r50.layer1
        self.bb_layer2 = r50.layer2   # C3: 512ch, stride 8  (used but not in BiFPN)
        self.bb_layer3 = r50.layer3   # C4: 1024ch, stride 16
        self.bb_layer4 = r50.layer4   # C5: 2048ch, stride 32

        # ── Input projections C4/C5 → C, P6/P7 from C5 ──
        def _proj(in_ch):
            return nn.Sequential(nn.Conv2d(in_ch, C, 1, bias=False), nn.BatchNorm2d(C))

        self.proj_c4 = _proj(1024)
        self.proj_c5 = _proj(2048)
        self.proj_p6 = nn.Sequential(nn.Conv2d(2048, C, 1, bias=False), nn.BatchNorm2d(C),
                                     nn.MaxPool2d(2, stride=2))
        self.pool_p7 = nn.MaxPool2d(2, stride=2)

        # ── BiFPN ──
        self.bifpn = nn.ModuleList([_BiFPNLayer(C) for _ in range(bifpn_layers)])

        # ── Prediction heads ──
        self.cls_head = _PredictionHead(C, head_depth, _NUM_ANCHORS * num_classes, num_levels=4)
        self.box_head = _PredictionHead(C, head_depth, _NUM_ANCHORS * 4, num_levels=4)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _backbone(self, x: torch.Tensor) -> list:
        x   = self.bb_stem(x)
        x   = self.bb_layer1(x)
        x   = self.bb_layer2(x)      # C3 (consumed by layer3)
        c4  = self.bb_layer3(x)
        c5  = self.bb_layer4(c4)
        p4  = self.proj_c4(c4)
        p5  = self.proj_c5(c5)
        p6  = self.proj_p6(c5)
        p7  = self.pool_p7(p6)
        return [p4, p5, p6, p7]

    def _compute_losses(self, cls_preds: list, box_preds: list, targets: list) -> dict:
        """
        Compute losses level-by-level to avoid allocating a single huge
        (B × total_anchors × num_classes) tensor that causes OOM.
        """
        device = cls_preds[0].device
        B      = cls_preds[0].shape[0]

        # Per-level anchor tensors (generated from actual feature-map sizes)
        level_anchors = [
            _make_level_anchors(*cls_preds[i].shape[-2:], _STRIDES[i], device=device)
            for i in range(len(cls_preds))
        ]
        all_anchors = torch.cat(level_anchors, dim=0)           # (N_total, 4)
        level_sizes = [a.shape[0] for a in level_anchors]

        cls_losses, reg_losses = [], []
        num_pos = 0

        for i, target in enumerate(targets):
            gt_boxes  = target['boxes']
            gt_labels = target['labels']

            # Matching (using all anchors at once — this tensor is 4×smaller than cls_flat)
            if gt_boxes.shape[0] > 0:
                iou              = box_iou(all_anchors, gt_boxes)
                max_iou, best_gt = iou.max(dim=1)
                pos_mask         = max_iou >= 0.5
                ignore_mask      = (max_iou >= 0.4) & ~pos_mask
                valid_mask       = ~ignore_mask
            else:
                pos_mask = valid_mask = torch.zeros(all_anchors.shape[0], dtype=torch.bool, device=device)

            # ── Classification loss: one level at a time to keep tensors small ──
            offset = 0
            for lvl, (cls_p, n) in enumerate(zip(cls_preds, level_sizes)):
                cls_lvl = cls_p[i].permute(1, 2, 0).reshape(-1, self.num_classes)  # (n, C)

                cls_tgt = torch.zeros(n, self.num_classes, device=device)
                lvl_pos = pos_mask[offset:offset + n]
                if lvl_pos.any():
                    pl = gt_labels[best_gt[offset:offset + n][lvl_pos]] - 1
                    cls_tgt[lvl_pos, pl] = 1.0

                lvl_valid = valid_mask[offset:offset + n]
                if lvl_valid.any():
                    cls_losses.append(_focal_loss(cls_lvl[lvl_valid], cls_tgt[lvl_valid]))
                offset += n

            # ── Regression loss (positive anchors only) ──
            if pos_mask.any():
                offset = 0
                pos_box_preds, pos_anchors_list, pos_gt_list = [], [], []
                for lvl, (box_p, n) in enumerate(zip(box_preds, level_sizes)):
                    lvl_pos = pos_mask[offset:offset + n]
                    if lvl_pos.any():
                        box_lvl = box_p[i].permute(1, 2, 0).reshape(-1, 4)
                        pos_box_preds.append(box_lvl[lvl_pos])
                        pos_anchors_list.append(level_anchors[lvl][lvl_pos])
                        pos_gt_list.append(gt_boxes[best_gt[offset:offset + n][lvl_pos]])
                    offset += n

                if pos_box_preds:
                    pb  = torch.cat(pos_box_preds)
                    pa  = torch.cat(pos_anchors_list)
                    pgt = torch.cat(pos_gt_list)
                    reg_losses.append(
                        F.smooth_l1_loss(pb, _encode_boxes(pa, pgt), reduction='sum', beta=0.1)
                    )
                    num_pos += int(pos_mask.sum())

        normalizer = max(num_pos, 1)
        loss_cls = sum(cls_losses) / normalizer if cls_losses else cls_preds[0].sum() * 0.0
        loss_reg = sum(reg_losses) / normalizer if reg_losses else box_preds[0].sum() * 0.0
        return {'loss_cls': loss_cls, 'loss_reg': loss_reg}

    def _decode_predictions(self, cls_preds, box_preds, images_t, original_sizes):
        device     = cls_preds[0].device
        image_size = images_t.tensors.shape[-2:]
        results    = []

        for i in range(cls_preds[0].shape[0]):
            boxes_all, labels_all, scores_all = [], [], []

            for lvl, (cls_p, box_p) in enumerate(zip(cls_preds, box_preds)):
                fh, fw = cls_p.shape[-2:]
                anchors = _make_level_anchors(fh, fw, _STRIDES[lvl], device=device)

                cls_lvl = cls_p[i].permute(1, 2, 0).reshape(-1, self.num_classes).sigmoid()
                box_lvl = box_p[i].permute(1, 2, 0).reshape(-1, 4)

                max_scores, labels = cls_lvl.max(dim=1)
                keep = max_scores > 0.05
                if not keep.any():
                    continue

                boxes  = _decode_boxes(anchors[keep], box_lvl[keep])
                boxes  = clip_boxes_to_image(boxes, image_size)
                boxes_all.append(boxes)
                labels_all.append(labels[keep] + 1)
                scores_all.append(max_scores[keep])

            if boxes_all:
                boxes_cat  = torch.cat(boxes_all)
                labels_cat = torch.cat(labels_all)
                scores_cat = torch.cat(scores_all)
                idx = batched_nms(boxes_cat, scores_cat, labels_cat, iou_threshold=0.5)
                boxes_cat, labels_cat, scores_cat = boxes_cat[idx], labels_cat[idx], scores_cat[idx]
            else:
                boxes_cat  = torch.zeros(0, 4, device=device)
                labels_cat = torch.zeros(0, dtype=torch.long, device=device)
                scores_cat = torch.zeros(0, device=device)

            results.append({'boxes': boxes_cat, 'labels': labels_cat, 'scores': scores_cat})

        return self.transform.postprocess(results, images_t.image_sizes, original_sizes)

    # ── Public API ────────────────────────────────────────────────────────────

    def forward(self, images, targets=None):
        original_sizes = [tuple(img.shape[-2:]) for img in images]
        images_t, targets_t = self.transform(images, targets)

        features = self._backbone(images_t.tensors)
        for layer in self.bifpn:
            features = layer(features)

        cls_preds = self.cls_head(features)
        box_preds = self.box_head(features)

        if self.training:
            return self._compute_losses(cls_preds, box_preds, targets_t)
        return self._decode_predictions(cls_preds, box_preds, images_t, original_sizes)

    @torch.no_grad()
    def predict(self, image: torch.Tensor, threshold: float = 0.5) -> dict:
        self.eval()
        pred = self.forward([image])[0]
        keep = pred['scores'] >= threshold
        return {k: v[keep] for k, v in pred.items()}

    def get_param_groups(self, lr: float = 0.001) -> list:
        backbone_prefixes = ('bb_stem', 'bb_layer1', 'bb_layer2', 'bb_layer3', 'bb_layer4')
        params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_bb = any(name.startswith(p) for p in backbone_prefixes)
            params.append({'params': param, 'lr': lr * 0.1 if is_bb else lr})
        return params
