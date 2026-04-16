"""
EfficientDet for grocery product detection.

Uses the effdet library -- Ross Wightman's PyTorch port of Google's EfficientDet.
  GitHub: https://github.com/rwightman/efficientdet-pytorch

Architecture (Tan, Pang, Le -- CVPR 2020, arXiv:1911.09070):
  1. EfficientNet backbone  -- extracts multi-scale features (C3, C4, C5)
  2. BiFPN                  -- weighted bidirectional feature pyramid fusion
  3. Class + box heads      -- shared across all feature levels, anchor-based

We load a pretrained EfficientNet backbone (ImageNet) and train the BiFPN
and prediction heads on the 356-class NorgesGruppen grocery dataset.
"""

from effdet import EfficientDet, get_efficientdet_config
from effdet.efficientdet import HeadNet


NUM_CLASSES = 356  # NorgesGruppen grocery dataset categories

# Compound scaling -- D0 is fastest, D7 is most accurate (Table 1 in the paper)
# D0: 512 px, B0 backbone, 64 BiFPN channels, 3 BiFPN layers  (~3.9M params)
# D3: 896 px, B3 backbone, 160 BiFPN channels, 6 BiFPN layers (~12M params)


def build_efficientdet(
    compound_coef: int = 0,
    pretrained_backbone: bool = True,
) -> EfficientDet:
    """
    Build EfficientDet-D{compound_coef} for the grocery detection task.

    The three core components from the paper are all present inside effdet:
      1. EfficientNet-B{N} backbone (ImageNet pretrained)
      2. BiFPN -- weighted bidirectional feature pyramid (Section 3 of paper)
      3. Shared class + box prediction heads

    We use a pretrained backbone and train BiFPN + heads from scratch
    on the grocery dataset.

    Args:
        compound_coef:       Scaling coefficient 0-7. Start with D0.
        pretrained_backbone: Load ImageNet weights. Strongly recommended.

    Returns:
        EfficientDet model ready for fine-tuning.
    """
    config = get_efficientdet_config(f"tf_efficientdet_d{compound_coef}")
    config.num_classes = NUM_CLASSES

    model = EfficientDet(config, pretrained_backbone=pretrained_backbone)

    # Replace the classification head -- pretrained head expects 90 COCO classes,
    # we need 356. Box regression head is class-agnostic so it stays as-is.
    model.class_net = HeadNet(config, num_outputs=config.num_classes)

    return model


def efficientdet_d0(pretrained_backbone: bool = True) -> EfficientDet:
    """D0 -- 512 px, ~3.9M params. Good for quick experiments."""
    return build_efficientdet(0, pretrained_backbone)


def efficientdet_d3(pretrained_backbone: bool = True) -> EfficientDet:
    """D3 -- 896 px, ~12M params. Good accuracy/speed trade-off."""
    return build_efficientdet(3, pretrained_backbone)
