#!/usr/bin/env python3
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = PROJECT_ROOT / "checkpoints_yolov9Augmented/weights/best.pt"
IMAGE = PROJECT_ROOT / "dataset/val/images/img_00122.jpg"
OUTPUT = PROJECT_ROOT / "results/yolov9_augmented_gradcam.png"
IMG_SIZE = 640


def gradcam(yolo: YOLO, image_path: str | Path, img_size: int = IMG_SIZE) -> np.ndarray:
    """Return a BGR Grad-CAM overlay (uint8) for `image_path` using `yolo`."""
    device = next(yolo.model.parameters()).device
    net = yolo.model.eval()

    target = net.model[15]

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    h_fwd = target.register_forward_hook(
        lambda _, __, out: activations.append(out[0] if isinstance(out, (list, tuple)) else out)
    )
    h_bwd = target.register_full_backward_hook(
        lambda _, __, grad_out: gradients.append(grad_out[0])
    )

    try:
        bgr = cv2.imread(str(image_path))
        h0, w0 = bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(bgr, (img_size, img_size)), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        x.requires_grad_(True)

        net.zero_grad()
        with torch.enable_grad():
            pred = net(x)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            pred[:, 4:, :].sigmoid().amax(dim=1).sum().backward()
    finally:
        h_fwd.remove()
        h_bwd.remove()

    act = activations[-1].detach()
    grad = gradients[-1].detach()
    cam = F.relu((grad.mean(dim=(2, 3), keepdim=True) * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(h0, w0), mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = cv2.GaussianBlur(cam, (0, 0), sigmaX=h0 / 40)
    cam = np.clip(cam / (np.percentile(cam, 99) + 1e-8), 0, 1)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 0.6, heatmap, 0.4, 0)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo = YOLO(str(WEIGHTS))
    yolo.model.to(device)
    overlay = gradcam(yolo, IMAGE)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT), overlay)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
