"""
Faster R-CNN with plain ResNet-50 backbone — no Feature Pyramid Network.

Original paper baseline: a single feature map from ResNet-50 layer4
(2048 channels, 1/32 input resolution) feeds the RPN and RoI heads directly.

Swap with fpn_rcnn.py to compare against the multi-scale FPN variant.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection import FasterRCNN as _TorchFasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class _ProjectedBackbone(nn.Module):
    """
    Wraps a plain ResNet-50 backbone and adds a 1x1 conv that projects the
    2048-channel layer4 output down to 256 channels.

    Without this, the RoI box head receives 2048x7x7 = 100K features and its
    first FC layer alone costs ~103M parameters — far more than the whole FPN
    model.  Projecting to 256 channels reduces that FC layer to ~13M params,
    making the two architectures parameter-comparable (~40M each).
    """
    def __init__(self, backbone: nn.Module, out_channels: int = 256):
        super().__init__()
        self.body = backbone
        self.proj = nn.Conv2d(2048, out_channels, kernel_size=1)
        self.out_channels = out_channels  # torchvision reads this attribute

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.body(x))


class FasterRCNN(nn.Module):
    """
    Faster R-CNN with a plain ResNet-50 backbone (no FPN).

    Args:
        num_classes: total classes including background (class 0)
        pretrained_backbone: use ImageNet-pretrained ResNet-50 weights
    """

    def __init__(
        self,
        num_classes: int,
        pretrained_backbone: bool = True,
        min_size: int = 800,
        max_size: int = 1333,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Plain ResNet-50: strip avgpool + fc, project 2048 → 256 channels.
        # The projection keeps parameter count comparable to the FPN model (~40M).
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained_backbone else None)
        resnet = nn.Sequential(*list(resnet.children())[:-2])
        backbone = _ProjectedBackbone(resnet, out_channels=256)

        # Single-level anchor generator — all anchor sizes on the one feature map
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),),
        )

        self.model = _TorchFasterRCNN(
            backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            min_size=min_size,
            max_size=max_size,
        )

    def forward(self, images, targets=None):
        return self.model(images, targets)

    @torch.no_grad()
    def predict(self, image: torch.Tensor, threshold: float = 0.5) -> dict:
        """Run inference on a single image tensor (3, H, W)."""
        self.model.eval()
        pred = self.model([image])[0]
        keep = pred["scores"] >= threshold
        return {
            "boxes": pred["boxes"][keep],
            "labels": pred["labels"][keep],
            "scores": pred["scores"][keep],
        }

    def get_param_groups(self, lr: float = 0.001) -> list:
        """Lower LR for the backbone, full LR for the detection head."""
        params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            params.append({
                "params": param,
                "lr": lr * 0.1 if "backbone" in name else lr,
            })
        return params
