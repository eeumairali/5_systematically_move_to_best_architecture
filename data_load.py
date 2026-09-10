"""Standalone dataset loaders for the mmCows benchmark (detection + ReID), with a
quick visual sanity check when run directly.

Label format (one .txt per image, YOLO-style): "identity cx cy w h" with cx, cy, w, h
normalized to [0, 1] and identity in 1..N_IDENTITIES (mmCows has no separate "class",
every box is a cow; the identity id doubles as the label).
"""
import random
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset

from n10_funs.config import CAMERAS, IMAGE_ROOT, LABEL_ROOT

WEBOTS_ROOT = Path("benchmarks/webots_multipleCows_location_Identity")
WEBOTS_IMAGE_ROOT = WEBOTS_ROOT / "images"
WEBOTS_LABEL_ROOT = WEBOTS_ROOT / "labels"
WEBOTS_CLASSES_FILE = WEBOTS_ROOT / "classes.txt"


def parse_label_file(label_path):
    """Read one 'identity cx cy w h' label file into a list of tuples."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        identity = int(parts[0])
        cx, cy, w, h = (float(v) for v in parts[1:5])
        boxes.append((identity, cx, cy, w, h))
    return boxes


def index_camera(camera):
    """List (image_path, label_path) pairs for one camera folder."""
    image_dir = IMAGE_ROOT / camera
    label_dir = LABEL_ROOT / camera
    pairs = []
    for label_path in sorted(label_dir.glob("*.txt")):
        image_path = image_dir / (label_path.stem + ".jpg")
        if image_path.exists():
            pairs.append((image_path, label_path))
    return pairs


class MmCowsDetectionDataset(Dataset):
    """Detection dataset: each item is (PIL image, list of (identity, cx, cy, w, h))."""

    def __init__(self, cameras=CAMERAS):
        self.samples = []
        for camera in cameras:
            self.samples.extend(index_camera(camera))
        if not self.samples:
            raise RuntimeError(f"No labelled frames found under {IMAGE_ROOT} / {LABEL_ROOT}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        boxes = parse_label_file(label_path)
        return image, boxes


class MmCowsReIDDataset(Dataset):
    """ReID dataset: crops each labelled box out of its frame and returns (crop, identity - 1)."""

    def __init__(self, cameras=CAMERAS, crop_size=(128, 128)):
        self.crop_size = crop_size
        self.entries = []  # (img_path, box)
        for camera in cameras:
            for img_path, label_path in index_camera(camera):
                for box in parse_label_file(label_path):
                    self.entries.append((img_path, box))
        if not self.entries:
            raise RuntimeError(f"No labelled boxes found under {IMAGE_ROOT} / {LABEL_ROOT}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, (identity, cx, cy, w, h) = self.entries[idx]
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size
        x1 = max(0, int((cx - w / 2) * img_w))
        y1 = max(0, int((cy - h / 2) * img_h))
        x2 = min(img_w, int((cx + w / 2) * img_w))
        y2 = min(img_h, int((cy + h / 2) * img_h))
        crop = image.crop((x1, y1, x2, y2)).resize(self.crop_size)
        return crop, identity - 1  # 1..N -> 0..N-1


def load_class_names(classes_file):
    """Read a YOLO-style classes.txt (one name per line, line index == class id)."""
    return [line.strip() for line in classes_file.read_text().splitlines() if line.strip()]


def index_flat(image_dir, label_dir, image_ext=".jpg"):
    """List (image_path, label_path) pairs for a flat images/ + labels/ folder pair
    (no per-camera subfolders, e.g. the webots benchmark)."""
    pairs = []
    for label_path in sorted(label_dir.glob("*.txt")):
        image_path = image_dir / (label_path.stem + image_ext)
        if image_path.exists():
            pairs.append((image_path, label_path))
    return pairs


class WebotsDetectionDataset(Dataset):
    """Detection dataset for the webots multi-cow/buffalo benchmark: each item is
    (PIL image, list of (identity, cx, cy, w, h)). identity is already 0-indexed,
    matching the line numbers in classes.txt."""

    def __init__(self, image_root=WEBOTS_IMAGE_ROOT, label_root=WEBOTS_LABEL_ROOT):
        self.samples = index_flat(image_root, label_root)
        self.class_names = load_class_names(WEBOTS_CLASSES_FILE) if WEBOTS_CLASSES_FILE.exists() else None
        if not self.samples:
            raise RuntimeError(f"No labelled frames found under {image_root} / {label_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        boxes = parse_label_file(label_path)
        return image, boxes


class WebotsReIDDataset(Dataset):
    """ReID dataset for the webots benchmark: crops each labelled box out of its frame
    and returns (crop, identity), identity already 0-indexed."""

    def __init__(self, image_root=WEBOTS_IMAGE_ROOT, label_root=WEBOTS_LABEL_ROOT, crop_size=(128, 128)):
        self.crop_size = crop_size
        self.class_names = load_class_names(WEBOTS_CLASSES_FILE) if WEBOTS_CLASSES_FILE.exists() else None
        self.entries = []  # (img_path, box)
        for img_path, label_path in index_flat(image_root, label_root):
            for box in parse_label_file(label_path):
                self.entries.append((img_path, box))
        if not self.entries:
            raise RuntimeError(f"No labelled boxes found under {image_root} / {label_root}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, (identity, cx, cy, w, h) = self.entries[idx]
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size
        x1 = max(0, int((cx - w / 2) * img_w))
        y1 = max(0, int((cy - h / 2) * img_h))
        x2 = min(img_w, int((cx + w / 2) * img_w))
        y2 = min(img_h, int((cy + h / 2) * img_h))
        crop = image.crop((x1, y1, x2, y2)).resize(self.crop_size)
        return crop, identity


def plot_detection_sample(image, boxes, ax=None, title=None, class_names=None):
    """Draw an image with its YOLO-normalized boxes, colored by identity."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    img_w, img_h = image.size
    ax.imshow(image)
    cmap = plt.colormaps.get_cmap("tab20")
    for identity, cx, cy, w, h in boxes:
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        rect = patches.Rectangle(
            (x1, y1), w * img_w, h * img_h,
            linewidth=2, edgecolor=cmap(identity % 20), facecolor="none",
        )
        ax.add_patch(rect)
        label = class_names[identity] if class_names else f"C{identity:02d}"
        ax.text(x1, max(y1 - 4, 0), label, color=cmap(identity % 20), fontsize=8, weight="bold")
    ax.set_title(title or f"{len(boxes)} boxes")
    ax.axis("off")
    return ax


def plot_reid_grid(reid_dataset, n=12, class_names=None):
    """Show a grid of random ReID crops with their identity label."""
    class_names = class_names if class_names is not None else getattr(reid_dataset, "class_names", None)
    idxs = random.sample(range(len(reid_dataset)), min(n, len(reid_dataset)))
    cols = 4
    rows = (len(idxs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten() if rows * cols > 1 else [axes]
    for ax, idx in zip(axes, idxs):
        crop, identity = reid_dataset[idx]
        ax.imshow(crop)
        ax.set_title(class_names[identity] if class_names else f"C{identity + 1:02d}")
        ax.axis("off")
    for ax in axes[len(idxs):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def demo(det_ds, reid_ds, name):
    print(f"[{name}] detection dataset: {len(det_ds)} frames")
    print(f"[{name}] reid dataset:      {len(reid_ds)} crops")

    image, boxes = det_ds[random.randrange(len(det_ds))]
    class_names = getattr(det_ds, "class_names", None)
    plot_detection_sample(image, boxes, title=f"{name}: sample frame, {len(boxes)} boxes", class_names=class_names)
    plt.show()

    plot_reid_grid(reid_ds, n=12)
    plt.show()


if __name__ == "__main__":
    demo(MmCowsDetectionDataset(), MmCowsReIDDataset(), name="mmCows")
    demo(WebotsDetectionDataset(), WebotsReIDDataset(), name="webots")
