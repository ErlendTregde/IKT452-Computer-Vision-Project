"""
Augment the training split of a YOLO-format dataset using the 7 non-trivial
symmetries of the dihedral group D4 (flips + rotations).

Each original image produces 7 new images, expanding the training set 8×:
  flip_h        — mirror left/right
  flip_v        — mirror top/bottom
  rot90         — 90° clockwise
  rot180        — 180°
  rot270        — 270° clockwise  (= 90° counter-clockwise)
  rot90_flip_h  — 90° CW then mirror left/right
  rot90_flip_v  — 90° CW then mirror top/bottom

Bounding boxes (YOLO normalised cx cy w h) are transformed analytically so no
interpolation artefacts are introduced in the labels.

Usage:
    uv run src/augment_dataset.py                  # augments dataset/train in-place
    uv run src/augment_dataset.py --dataset path/to/dataset
"""

import argparse
from pathlib import Path

from PIL import Image


# ── Box transforms (all in normalised YOLO coords) ───────────────────────────
# Each function takes (cx, cy, bw, bh) and returns the same tuple transformed.
# For 90/270° rotations the image becomes H×W → bw and bh swap after
# renormalising to the new dimensions, but since they are already normalised
# w.r.t. their own axis, bw and bh simply exchange.

def _flip_h(cx, cy, bw, bh):
    return 1 - cx, cy, bw, bh

def _flip_v(cx, cy, bw, bh):
    return cx, 1 - cy, bw, bh

def _rot90(cx, cy, bw, bh):
    # 90° CW: (cx,cy) → (1-cy, cx); w↔h
    return 1 - cy, cx, bh, bw

def _rot180(cx, cy, bw, bh):
    return 1 - cx, 1 - cy, bw, bh

def _rot270(cx, cy, bw, bh):
    # 270° CW (= 90° CCW): (cx,cy) → (cy, 1-cx); w↔h
    return cy, 1 - cx, bh, bw

def _rot90_flip_h(cx, cy, bw, bh):
    # rot90 then flip_h: (1-cy,cx) → (1-(1-cy), cx) = (cy, cx); w↔h
    return cy, cx, bh, bw

def _rot90_flip_v(cx, cy, bw, bh):
    # rot90 then flip_v: (1-cy,cx) → (1-cy, 1-cx); w↔h
    return 1 - cy, 1 - cx, bh, bw


AUGMENTATIONS = {
    "flip_h":       (_flip_h,       lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)),
    "flip_v":       (_flip_v,       lambda img: img.transpose(Image.FLIP_TOP_BOTTOM)),
    "rot90":        (_rot90,        lambda img: img.transpose(Image.ROTATE_270)),  # PIL ROTATE_270 = 90° CW
    "rot180":       (_rot180,       lambda img: img.transpose(Image.ROTATE_180)),
    "rot270":       (_rot270,       lambda img: img.transpose(Image.ROTATE_90)),   # PIL ROTATE_90 = 90° CCW = 270° CW
    "rot90_flip_h": (_rot90_flip_h, lambda img: img.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)),
    "rot90_flip_v": (_rot90_flip_v, lambda img: img.transpose(Image.ROTATE_270).transpose(Image.FLIP_TOP_BOTTOM)),
}


# ── Label I/O ─────────────────────────────────────────────────────────────────

def read_labels(label_path: Path) -> list[tuple]:
    """Return list of (class_id, cx, cy, bw, bh)."""
    rows = []
    if label_path.exists():
        for line in label_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) == 5:
                cls = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
                rows.append((cls, cx, cy, bw, bh))
    return rows


def write_labels(label_path: Path, rows: list[tuple]) -> None:
    lines = [f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cls, cx, cy, bw, bh in rows]
    label_path.write_text("\n".join(lines) + "\n" if lines else "")


# ── Main ──────────────────────────────────────────────────────────────────────

def augment_split(images_dir: Path, labels_dir: Path) -> None:
    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and not any(aug in p.stem for aug in AUGMENTATIONS)  # skip already-augmented
    )

    print(f"Found {len(image_files)} original images in {images_dir}")
    created = 0

    for img_path in image_files:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  WARNING: could not read {img_path.name}: {e}, skipping")
            continue

        label_path = labels_dir / (img_path.stem + ".txt")
        labels = read_labels(label_path)

        for aug_name, (box_fn, img_fn) in AUGMENTATIONS.items():
            stem = f"{img_path.stem}_{aug_name}"
            out_img  = images_dir / (stem + img_path.suffix)
            out_lbl  = labels_dir / (stem + ".txt")

            if out_img.exists():
                continue  # don't overwrite existing augmentations

            aug_img = img_fn(img)
            aug_img.save(str(out_img))

            aug_labels = [(cls, *box_fn(cx, cy, bw, bh)) for cls, cx, cy, bw, bh in labels]
            write_labels(out_lbl, aug_labels)
            created += 1

    print(f"Created {created} augmented image/label pairs  "
          f"(total now: {len(image_files)} × {len(AUGMENTATIONS) + 1} = "
          f"{len(image_files) * (len(AUGMENTATIONS) + 1)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="D4 augmentation for YOLO-format datasets")
    parser.add_argument("--dataset", default="dataset", help="dataset root (default: dataset)")
    args = parser.parse_args()

    root = Path(args.dataset)
    train_imgs   = root / "train" / "images"
    train_labels = root / "train" / "labels"

    if not train_imgs.exists():
        raise FileNotFoundError(f"Training images not found at {train_imgs}")

    augment_split(train_imgs, train_labels)


if __name__ == "__main__":
    main()
