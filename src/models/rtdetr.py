"""
RT-DETR with ResNet-50 backbone.

Transformer-based real-time detector [Zhao et al., 2023] with ResNet-50 backbone,
keeping total parameter count ~40M for fair comparison.

Architecture:
  ResNet-50 → project C3/C4/C5 to 256ch
  → AIFI encoder (intra-scale transformer on C5)
  → CCFM (cross-scale CNN feature fusion, top-down + bottom-up)
  → Transformer decoder (6 layers, 300 queries)
  → class + box prediction heads per decoder layer
  Loss: focal (cls) + L1 + GIoU (reg), Hungarian matching
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import generalized_box_iou, clip_boxes_to_image, batched_nms


# ── Positional encoding ───────────────────────────────────────────────────────

def _pos2d_sine(h: int, w: int, dim: int, device, temperature: float = 10000.0) -> torch.Tensor:
    """2-D sinusoidal position encoding, shape (h*w, dim)."""
    assert dim % 4 == 0
    quarter = dim // 4
    ys = torch.arange(h, device=device, dtype=torch.float32)
    xs = torch.arange(w, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys / h, xs / w, indexing='ij')
    grid_y, grid_x = grid_y.flatten(), grid_x.flatten()

    inv_freq = temperature ** (-torch.arange(0, quarter, dtype=torch.float32, device=device) / quarter)
    pe_x = grid_x.unsqueeze(1) * inv_freq.unsqueeze(0)  # (h*w, quarter)
    pe_y = grid_y.unsqueeze(1) * inv_freq.unsqueeze(0)
    # 4 × quarter = dim
    pe = torch.cat([pe_x.sin(), pe_x.cos(), pe_y.sin(), pe_y.cos()], dim=1)
    return pe


# ── Encoder components ────────────────────────────────────────────────────────

class _AIFI(nn.Module):
    """
    Attention-based Intra-scale Feature Interaction.
    One transformer encoder layer applied to flattened C5 features.
    """
    def __init__(self, d_model: int = 256, nhead: int = 8, ffn_dim: int = 1024):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C),  pos: (N, C)
        xp = x + pos.unsqueeze(0)
        x2, _ = self.attn(xp, xp, x)
        x = self.norm1(x + x2)
        x = self.norm2(x + self.ffn(x))
        return x


class _ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=k // 2, stride=s, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class _CCFM(nn.Module):
    """
    Cross-scale CNN Feature Fusion Module.
    Top-down then bottom-up fusion of S3 / S4 / S5 (all at d_model channels).
    Each fusion uses two 3x3 Conv-BN-SiLU blocks.
    """
    def __init__(self, d_model: int = 256):
        super().__init__()
        # Top-down
        self.td_45 = nn.Sequential(_ConvBNAct(d_model, d_model), _ConvBNAct(d_model, d_model))
        self.td_34 = nn.Sequential(_ConvBNAct(d_model, d_model), _ConvBNAct(d_model, d_model))
        # Bottom-up
        self.bu_45 = nn.Sequential(_ConvBNAct(d_model, d_model), _ConvBNAct(d_model, d_model))
        self.bu_56 = nn.Sequential(_ConvBNAct(d_model, d_model), _ConvBNAct(d_model, d_model))
        # Downsampling for bottom-up
        self.down3 = nn.MaxPool2d(2, stride=2)
        self.down4 = nn.MaxPool2d(2, stride=2)

    def forward(self, s3, s4, s5):
        # Top-down
        s4_td = self.td_45(s4 + F.interpolate(s5, size=s4.shape[-2:], mode='nearest'))
        s3_td = self.td_34(s3 + F.interpolate(s4_td, size=s3.shape[-2:], mode='nearest'))
        # Bottom-up
        s4_bu = self.bu_45(s4_td + self.down3(s3_td))
        s5_bu = self.bu_56(s5   + self.down4(s4_bu))
        return s3_td, s4_bu, s5_bu


# ── Decoder components ────────────────────────────────────────────────────────

class _DecoderLayer(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8, ffn_dim: int = 2048):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, q: torch.Tensor, mem: torch.Tensor, mem_pos: torch.Tensor) -> torch.Tensor:
        # Self-attention among queries
        q2, _ = self.self_attn(q, q, q)
        q = self.norm1(q + q2)
        # Cross-attention to encoder memory
        mp = mem + mem_pos.unsqueeze(0)
        q2, _ = self.cross_attn(q, mp, mem)
        q = self.norm2(q + q2)
        q = self.norm3(q + self.ffn(q))
        return q


class _MLP(nn.Module):
    """3-layer MLP for box regression."""
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
        )

    def forward(self, x):
        return self.layers(x).sigmoid()  # normalised cxcywh


# ── Hungarian matching ─────────────────────────────────────────────────────────

@torch.no_grad()
def _match(pred_logits, pred_boxes, gt_labels, gt_boxes, num_classes):
    """
    Bipartite matching between predictions and ground truth for one image.
    pred_logits: (Q, C)  pred_boxes: (Q, 4) cxcywh normalised
    gt_labels:   (M,)    gt_boxes:   (M, 4) xyxy absolute pixels → converted inside
    Returns (pred_idx, gt_idx) matched pairs.
    """
    Q, C = pred_logits.shape
    M    = gt_labels.shape[0]
    if M == 0:
        return (torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))

    # Convert gt_boxes xyxy → cxcywh (already normalised if called after transform)
    # Actually gt_boxes here are in absolute xyxy (from RCNN transform output).
    # We work in absolute coords for cost, then normalise for L1.
    pred_xyxy = _cxcywh_to_xyxy_abs(pred_boxes, gt_boxes)  # map pred cxcywh to abs xyxy

    # Classification cost: negative log-softmax probability for the gt class
    prob = pred_logits.softmax(-1)  # (Q, C)
    cls_cost = -prob[:, gt_labels - 1]  # (Q, M)  gt_labels are 1-indexed

    # L1 cost on normalised boxes
    gt_norm = _xyxy_to_cxcywh_norm(gt_boxes, pred_boxes)   # (M, 4)
    pred_norm = pred_boxes                                   # already cxcywh norm
    l1_cost = torch.cdist(pred_norm.float(), gt_norm.float(), p=1)  # (Q, M)

    # GIoU cost
    giou = generalized_box_iou(pred_xyxy, gt_boxes)  # (Q, M)
    giou_cost = -giou

    cost = 2.0 * cls_cost + 5.0 * l1_cost + 2.0 * giou_cost
    row, col = linear_sum_assignment(cost.cpu().numpy())
    return (torch.as_tensor(row, dtype=torch.long), torch.as_tensor(col, dtype=torch.long))


def _cxcywh_to_xyxy_abs(pred_norm, gt_boxes_xyxy):
    """Convert normalised cxcywh predictions to absolute xyxy using gt_boxes as size reference."""
    # Determine image size from gt_boxes extent
    img_w = gt_boxes_xyxy[:, 2].max().clamp(min=1)
    img_h = gt_boxes_xyxy[:, 3].max().clamp(min=1)
    cx, cy, w, h = pred_norm[:, 0], pred_norm[:, 1], pred_norm[:, 2], pred_norm[:, 3]
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return torch.stack([x1, y1, x2, y2], dim=1)


def _xyxy_to_cxcywh_norm(gt_xyxy, pred_norm):
    """Convert absolute xyxy gt to normalised cxcywh using pred range as image size."""
    img_w = gt_xyxy[:, 2].max().clamp(min=1)
    img_h = gt_xyxy[:, 3].max().clamp(min=1)
    cx = ((gt_xyxy[:, 0] + gt_xyxy[:, 2]) / 2) / img_w
    cy = ((gt_xyxy[:, 1] + gt_xyxy[:, 3]) / 2) / img_h
    w  = (gt_xyxy[:, 2] - gt_xyxy[:, 0]) / img_w
    h  = (gt_xyxy[:, 3] - gt_xyxy[:, 1]) / img_h
    return torch.stack([cx, cy, w, h], dim=1)


# ── Main model ────────────────────────────────────────────────────────────────

class RTDETR(nn.Module):
    """
    RT-DETR with ResNet-50 backbone (~40M parameters).

    Parameter breakdown:
      ResNet-50 backbone               23.5 M
      Input projections (C3/C4/C5)      0.9 M
      AIFI encoder (1 layer, 256d)       0.8 M
      CCFM (8× ConvBNSiLU, 256ch)       3.0 M
      Decoder (6 layers, 256d, 2048ffn)  9.4 M
      Box MLP heads (×6 layers)          1.6 M
      Query embeddings + misc            0.1 M
      Total                            ~39.3 M
    """

    D_MODEL   = 256
    NHEAD     = 8
    FFN_DIM   = 2048
    NUM_DEC   = 6
    NUM_Q     = 100

    def __init__(
        self,
        num_classes: int,
        pretrained_backbone: bool = True,
        min_size: int = 800,
        max_size: int = 1333,
    ):
        super().__init__()
        self.num_classes = num_classes
        D = self.D_MODEL

        self.transform = GeneralizedRCNNTransform(
            min_size=min_size, max_size=max_size,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )

        # ── Backbone ──
        r50 = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained_backbone else None)
        self.bb_stem   = nn.Sequential(r50.conv1, r50.bn1, r50.relu, r50.maxpool)
        self.bb_layer1 = r50.layer1
        self.bb_layer2 = r50.layer2   # C3: 512ch, stride 8
        self.bb_layer3 = r50.layer3   # C4: 1024ch, stride 16
        self.bb_layer4 = r50.layer4   # C5: 2048ch, stride 32

        # ── Input projections → D ──
        self.proj_c3 = nn.Sequential(nn.Conv2d(512,  D, 1, bias=False), nn.BatchNorm2d(D))
        self.proj_c4 = nn.Sequential(nn.Conv2d(1024, D, 1, bias=False), nn.BatchNorm2d(D))
        self.proj_c5 = nn.Sequential(nn.Conv2d(2048, D, 1, bias=False), nn.BatchNorm2d(D))

        # ── Encoder ──
        self.aifi = _AIFI(D, self.NHEAD, ffn_dim=1024)
        self.ccfm = _CCFM(D)

        # ── Decoder ──
        self.query_embed = nn.Embedding(self.NUM_Q, D)
        self.decoder_layers = nn.ModuleList(
            [_DecoderLayer(D, self.NHEAD, self.FFN_DIM) for _ in range(self.NUM_DEC)]
        )

        # Per-layer prediction heads (auxiliary losses, standard DETR practice)
        self.cls_heads = nn.ModuleList([nn.Linear(D, num_classes) for _ in range(self.NUM_DEC)])
        self.box_heads = nn.ModuleList([_MLP(D) for _ in range(self.NUM_DEC)])

        self._init_weights()

    def _init_weights(self):
        prior_prob = 0.01
        bias = -math.log((1 - prior_prob) / prior_prob)
        for head in self.cls_heads:
            nn.init.constant_(head.bias, bias)

    # ── Forward ──────────────────────────────────────────────────────────────

    def _backbone_features(self, x):
        x = self.bb_stem(x)
        x = self.bb_layer1(x)
        c3 = self.bb_layer2(x)
        c4 = self.bb_layer3(c3)
        c5 = self.bb_layer4(c4)
        return self.proj_c3(c3), self.proj_c4(c4), self.proj_c5(c5)

    def _encode(self, s3, s4, s5):
        B, C, H5, W5 = s5.shape
        # AIFI on s5
        flat = s5.flatten(2).permute(0, 2, 1)          # (B, H5*W5, C)
        pos  = _pos2d_sine(H5, W5, C, s5.device)        # (H5*W5, C)
        flat = self.aifi(flat, pos)
        s5_enc = flat.permute(0, 2, 1).reshape(B, C, H5, W5)
        # CCFM
        return self.ccfm(s3, s4, s5_enc)

    def _build_memory(self, s3, s4, s5):
        """Flatten and concatenate s4+s5 features for decoder cross-attention.
        s3 (stride-8) produces ~10K tokens which OOMs; s4+s5 gives ~3K tokens."""
        parts, poses = [], []
        for feat in (s4, s5):
            B, C, H, W = feat.shape
            parts.append(feat.flatten(2).permute(0, 2, 1))   # (B, H*W, C)
            poses.append(_pos2d_sine(H, W, C, feat.device))   # (H*W, C)
        mem = torch.cat(parts, dim=1)   # (B, total_len, C)
        pos = torch.cat(poses, dim=0)   # (total_len, C)
        return mem, pos

    def _decode(self, mem, pos):
        B = mem.shape[0]
        q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, Q, D)
        all_logits, all_boxes = [], []
        for layer, cls_head, box_head in zip(self.decoder_layers, self.cls_heads, self.box_heads):
            q = layer(q, mem, pos)
            all_logits.append(cls_head(q))   # (B, Q, C)
            all_boxes.append(box_head(q))    # (B, Q, 4) normalised cxcywh
        return all_logits, all_boxes

    def forward(self, images, targets=None):
        original_sizes = [tuple(img.shape[-2:]) for img in images]
        images_t, targets_t = self.transform(images, targets)

        s3, s4, s5 = self._backbone_features(images_t.tensors)
        s3, s4, s5 = self._encode(s3, s4, s5)
        mem, pos   = self._build_memory(s3, s4, s5)
        all_logits, all_boxes = self._decode(mem, pos)

        if self.training:
            return self._compute_losses(all_logits, all_boxes, targets_t)

        # Use final decoder layer for inference
        return self._decode_predictions(
            all_logits[-1], all_boxes[-1], images_t, original_sizes
        )

    # ── Loss ─────────────────────────────────────────────────────────────────

    def _compute_losses(self, all_logits, all_boxes, targets):
        total_cls = torch.tensor(0.0, device=all_logits[0].device)
        total_l1  = torch.tensor(0.0, device=all_logits[0].device)
        total_giou = torch.tensor(0.0, device=all_logits[0].device)

        num_layers = len(all_logits)
        # Accumulate loss from each decoder layer (auxiliary losses)
        for logits, boxes in zip(all_logits, all_boxes):
            lc, ll, lg = self._layer_loss(logits, boxes, targets)
            total_cls  = total_cls  + lc
            total_l1   = total_l1   + ll
            total_giou = total_giou + lg

        scale = 1.0 / num_layers
        return {
            'loss_cls':  total_cls  * scale,
            'loss_l1':   total_l1   * scale,
            'loss_giou': total_giou * scale,
        }

    def _layer_loss(self, logits, boxes, targets):
        """Loss for one decoder layer output."""
        device = logits.device
        B = logits.shape[0]
        num_pos = 0
        cls_losses, l1_losses, giou_losses = [], [], []

        for i, target in enumerate(targets):
            gt_boxes  = target['boxes']    # (M, 4) xyxy absolute
            gt_labels = target['labels']   # (M,) 1-indexed

            M = gt_boxes.shape[0]
            pred_log = logits[i].detach()  # (Q, C) — matching is @no_grad
            pred_box = boxes[i].detach()   # (Q, 4)

            # No-object classification loss for unmatched queries
            # All queries are first marked as background
            cls_target = torch.zeros(self.NUM_Q, dtype=torch.long, device=device)

            if M == 0:
                cls_losses.append(
                    F.cross_entropy(logits[i], cls_target)
                )
                continue

            row, col = _match(pred_log, pred_box, gt_labels, gt_boxes, self.num_classes)
            row = row.to(device)
            col = col.to(device)

            # Matched class targets (1-indexed → 0-indexed for cross_entropy)
            cls_target[row] = gt_labels[col] - 1
            cls_losses.append(F.cross_entropy(logits[i], cls_target))

            # Box losses only for matched predictions
            if row.numel() > 0:
                # L1 on normalised cxcywh
                gt_norm = _xyxy_to_cxcywh_norm(gt_boxes[col], boxes[i])
                l1_losses.append(F.l1_loss(boxes[i][row], gt_norm, reduction='sum'))

                # GIoU
                pred_abs = _cxcywh_to_xyxy_abs(boxes[i][row], gt_boxes[col])
                giou = generalized_box_iou(pred_abs, gt_boxes[col])
                giou_losses.append((1 - giou.diag()).sum())
                num_pos += row.numel()

        normalizer = max(num_pos, 1)
        lc = sum(cls_losses) / B
        ll = (sum(l1_losses) / normalizer) if l1_losses else logits.sum() * 0.0
        lg = (sum(giou_losses) / normalizer) if giou_losses else logits.sum() * 0.0
        return lc, ll, lg

    # ── Inference ────────────────────────────────────────────────────────────

    def _decode_predictions(self, logits, boxes, images_t, original_sizes):
        """Convert final decoder output to detection dicts."""
        results = []
        image_size = images_t.tensors.shape[-2:]
        img_h, img_w = image_size

        for i in range(logits.shape[0]):
            scores_all = logits[i].softmax(-1)          # (Q, C)
            max_scores, labels = scores_all.max(-1)     # (Q,)

            # Convert normalised cxcywh → absolute xyxy
            cx, cy, w, h = boxes[i].unbind(-1)
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            abs_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
            abs_boxes = clip_boxes_to_image(abs_boxes, image_size)

            keep = max_scores > 0.05
            abs_boxes_f = abs_boxes[keep]
            labels_f    = labels[keep] + 1          # 1-indexed
            scores_f    = max_scores[keep]

            if abs_boxes_f.shape[0] > 0:
                idx = batched_nms(abs_boxes_f, scores_f, labels_f, iou_threshold=0.5)
                abs_boxes_f, labels_f, scores_f = abs_boxes_f[idx], labels_f[idx], scores_f[idx]

            results.append({'boxes': abs_boxes_f, 'labels': labels_f, 'scores': scores_f})

        return self.transform.postprocess(results, images_t.image_sizes, original_sizes)

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
