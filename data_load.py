"""Dataset and CenterNet-style target preparation for the JDE pipeline."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def read_classes(data_root):
    path = Path(data_root) / "classes.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def parse_label_file(label_path):
    boxes = []
    for line in Path(label_path).read_text().splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        identity, cx, cy, width, height = values[:5]
        boxes.append((int(identity), float(cx), float(cy), float(width), float(height)))
    return boxes


def index_split(data_root, split):
    """Return labelled image pairs, honoring a YOLO split file when present."""
    data_root = Path(data_root)
    image_root, label_root = data_root / "images", data_root / "labels"
    split_file = data_root / f"{split}.txt"
    if split_file.exists():
        names = [Path(line.strip()).name for line in split_file.read_text().splitlines() if line.strip()]
    else:
        names = sorted(path.name for path in image_root.glob("*.jpg"))
    return [
        (image_root / name, label_root / f"{Path(name).stem}.txt")
        for name in names
        if (image_root / name).exists() and (label_root / f"{Path(name).stem}.txt").exists()
    ]


def _draw_gaussian(heatmap, center_x, center_y, radius):
    radius = max(0, int(radius))
    sigma = max(radius / 3.0, 0.5)
    height, width = heatmap.shape
    left, right = min(center_x, radius), min(width - center_x, radius + 1)
    top, bottom = min(center_y, radius), min(height - center_y, radius + 1)
    if left + right <= 0 or top + bottom <= 0:
        return
    ys = torch.arange(-top, bottom).float().view(-1, 1)
    xs = torch.arange(-left, right).float().view(1, -1)
    gaussian = torch.exp(-(xs.square() + ys.square()) / (2 * sigma * sigma))
    region = heatmap[center_y - top:center_y + bottom, center_x - left:center_x + right]
    torch.maximum(region, gaussian, out=region)


def _gaussian_radius(height, width, overlap=0.7):
    a1, b1, c1 = 1.0, height + width, width * height * (1 - overlap) / (1 + overlap)
    a2, b2, c2 = 4.0, 2 * (height + width), (1 - overlap) * width * height
    a3, b3, c3 = 4 * overlap, -2 * overlap * (height + width), (overlap - 1) * width * height
    roots = [
        (b1 - (b1 * b1 - 4 * a1 * c1) ** 0.5) / (2 * a1),
        (b2 - (b2 * b2 - 4 * a2 * c2) ** 0.5) / (2 * a2),
        (b3 + (b3 * b3 - 4 * a3 * c3) ** 0.5) / (2 * a3),
    ]
    return max(0.0, min(roots))


def make_targets(boxes, image_size, stride, max_identities):
    image_width, image_height = image_size
    feature_width, feature_height = image_width // stride, image_height // stride
    heatmap = torch.zeros(1, feature_height, feature_width)
    centers, sizes, offsets, identities = [], [], [], []
    for identity, cx, cy, width, height in boxes:
        identity = max(0, min(identity, max_identities - 1))
        feature_x, feature_y = cx * image_width / stride, cy * image_height / stride
        center_x, center_y = int(feature_x), int(feature_y)
        if not (0 <= center_x < feature_width and 0 <= center_y < feature_height):
            continue
        radius = _gaussian_radius(height * image_height / stride, width * image_width / stride)
        _draw_gaussian(heatmap[0], center_x, center_y, radius)
        centers.append((center_x, center_y))
        sizes.append((width * image_width / stride, height * image_height / stride))
        offsets.append((feature_x - center_x, feature_y - center_y))
        identities.append(identity)
    count = max(len(centers), 1)
    return {
        "hm": heatmap,
        "centers": torch.tensor(centers or [(0, 0)], dtype=torch.long),
        "wh": torch.tensor(sizes or [(0.0, 0.0)], dtype=torch.float32),
        "reg": torch.tensor(offsets or [(0.0, 0.0)], dtype=torch.float32),
        "ids": torch.tensor(identities or [0], dtype=torch.long),
        "mask": torch.tensor([True] * len(centers) + [False] * (count - len(centers)), dtype=torch.bool),
    }


class WebotsJDEDataset(Dataset):
    """Webots images with ``identity cx cy width height`` YOLO labels."""

    def __init__(self, data_root, split, image_size=(640, 400), stride=4, max_identities=None):
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.stride = stride
        self.class_names = read_classes(self.data_root)
        self.max_identities = max_identities or max(len(self.class_names), 1)
        self.samples = [(image, parse_label_file(label)) for image, label in index_split(self.data_root, split)]
        if not self.samples:
            raise RuntimeError(f"No labelled images found for split '{split}' under {self.data_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, boxes = self.samples[index]
        image = Image.open(image_path).convert("RGB").resize(self.image_size)
        tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - MEAN) / STD
        return tensor, make_targets(boxes, self.image_size, self.stride, self.max_identities)


def jde_collate(batch):
    images, targets = zip(*batch)
    max_boxes = max(target["centers"].shape[0] for target in targets)
    result = {"hm": torch.stack([target["hm"] for target in targets])}
    for key, dtype in (("centers", torch.long), ("wh", torch.float32), ("reg", torch.float32), ("ids", torch.long), ("mask", torch.bool)):
        padded = torch.zeros(len(targets), max_boxes, *targets[0][key].shape[1:], dtype=dtype)
        for row, target in enumerate(targets):
            length = target[key].shape[0]
            padded[row, :length] = target[key]
        result[key] = padded
    return torch.stack(images), result
