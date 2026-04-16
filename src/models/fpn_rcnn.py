"""
Faster R-CNN with ResNet-50 + Feature Pyramid Network (FPN) backbone.

Multi-scale variant: FPN builds feature maps at 5 scales (P2-P6), improving
detection of objects that vary greatly in size.

Swap with faster_rcnn.py to compare against the plain backbone baseline.
"""

import torch
import torch.nn as nn
from typing import Optional
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_model(
    num_classes: int,
    pretrained: bool = True,
    pretrained_backbone: bool = True,
    min_size: int = 800,
    max_size: int = 1333,
    rpn_anchor_generator = None,
    rpn_head = None,
    roi_box_head = None,
    ) -> nn.Module:
    """
    Create a Faster R-CNN model with ResNet-50 backbone and FPN.

    Args:
        num_classes: Number of object classes (including background)
        pretrained: Whether to use pretrained weights for the full model
        pretrained_backbone: Whether to use pretrained backbone weights
        min_size: Minimum image size for inference (default: 800)
        max_size: Maximum image size for inference (default: 1333)
        rpn_anchor_generator: Custom anchor generator for RPN (optional)
        rpn_head: Custom RPN head (optional)
        roi_box_head: Custom RoI box head (optional)

    Returns:
        Faster R-CNN model (nn.Module)
    """
    # Load pretrained weights if requested
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    weights_backbone = None
    if weights is None and pretrained_backbone:
        weights_backbone = ResNet50_Weights.DEFAULT

    # Create the base model
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=weights_backbone,
        num_classes=91,  # Placeholder, will be replaced
        min_size=min_size,
        max_size=max_size,
        rpn_anchor_generator=rpn_anchor_generator,
        rpn_head=rpn_head,
        roi_box_head=roi_box_head,
    )

    # Replace the box predictor for custom number of classes
    # Get the number of input features from the existing box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features #type: ignore

    # Replace the head with a new one that has the correct number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


class FasterRCNN(nn.Module):
    """
    Wrapper class for Faster R-CNN with additional utilities.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        pretrained_backbone: Optional[bool] = None,
        min_size: int = 800,
        max_size: int = 1333,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.anchor_generator = get_anchor_generator()
        if pretrained_backbone is None:
            pretrained_backbone = pretrained
        self.model = create_model(
            num_classes=num_classes,
            pretrained=pretrained,
            pretrained_backbone=pretrained_backbone,
            min_size=min_size,
            max_size=max_size,
            rpn_anchor_generator=self.anchor_generator,
        )

    def forward(self, images, targets=None):
        """
        Forward pass.

        Args:
            images: List of tensors, each of shape (3, H, W)
            targets: List of target dicts (for training), each containing:
                - boxes: Tensor of shape (N, 4) in [x1, y1, x2, y2] format
                - labels: Tensor of shape (N,)
                - masks: Optional Tensor of shape (N, H, W)

        Returns:
            During training: dict with losses
            During inference: list of dicts with predictions
        """
        return self.model(images, targets)

    @torch.no_grad()
    def predict(self, image: torch.Tensor, threshold: float = 0.5) -> dict:
        """
        Run inference on a single image.

        Args:
            image: Tensor of shape (3, H, W), normalized and on correct device
            threshold: Confidence threshold for detections

        Returns:
            Dict with keys:
                - boxes: Tensor of shape (N, 4)
                - labels: Tensor of shape (N,)
                - scores: Tensor of shape (N,)
        """
        self.model.eval()
        predictions = self.model([image])
        pred = predictions[0]

        # Filter by confidence threshold
        keep = pred["scores"] >= threshold
        return {
            "boxes": pred["boxes"][keep],
            "labels": pred["labels"][keep],
            "scores": pred["scores"][keep],
        }

    def get_param_groups(self, lr: float = 0.001) -> list:
        """
        Get parameter groups for optimizer with different learning rates.

        Common practice: lower LR for backbone, higher LR for head.
        """
        params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # Use lower LR for backbone (feature extractor)
            if "backbone" in name:
                params.append({"params": param, "lr": lr * 0.1})
            else:
                params.append({"params": param, "lr": lr})
        return params


def get_anchor_generator(
    sizes: tuple = ((32,), (64,), (128,), (256,), (512,)),
    aspect_ratios: tuple = ((0.5, 1.0, 2.0),) * 5,
):
    """
    Create a custom anchor generator.

    Args:
        sizes: Anchor scales for each FPN level
        aspect_ratios: Aspect ratios for each FPN level

    Returns:
        AnchorGenerator module
    """
    from torchvision.models.detection.anchor_utils import AnchorGenerator

    return AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)
