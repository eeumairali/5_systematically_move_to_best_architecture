"""Sequential detector -> predicted crops -> ReID training and inference.

Stage 1 trains only object detection. Stage 2 receives crops made from Stage 1's
predicted boxes and trains an identity classifier. Inference reports detector,
crop/decode, ReID, and total two-stage latency separately.

Run this file directly with CLI flags - see ``build_parser()`` for all options,
or the demo command at the bottom of this file.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch import nn
from torch.utils.data import DataLoader, Dataset

from n10_funs.model import JointBackbone

ImageFile.LOAD_TRUNCATED_IMAGES = True

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

CROP_SIZE = 128


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def read_classes(data_root):
    path = Path(data_root) / "classes.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def read_identities(data_root):
    """Map raw per-animal identity ids (identities.txt, e.g. cow1..15, buffalo1..15) to contiguous 0-based indices."""
    path = Path(data_root) / "identities.txt"
    if not path.exists():
        return {}
    mapping = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        mapping[int(parts[0])] = len(mapping)
    return mapping


def parse_label_file(label_path, identity_map=None):
    """Labels are ``class cx cy width height identity`` (identity is the animal's raw id, not the class)."""
    boxes = []
    for line in Path(label_path).read_text().splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        if len(values) >= 6:
            _class_id, cx, cy, width, height, raw_identity = values[:6]
        else:
            raw_identity, cx, cy, width, height = values[:5]
        identity = int(raw_identity)
        if identity_map:
            identity = identity_map.get(identity, identity)
        boxes.append((identity, float(cx), float(cy), float(width), float(height)))
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
        self.identity_map = read_identities(self.data_root)
        self.max_identities = max_identities or (len(self.identity_map) if self.identity_map else max(len(self.class_names), 1))
        self.samples = [(image, parse_label_file(label, self.identity_map)) for image, label in index_split(self.data_root, split)]
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


# ---------------------------------------------------------------------------
# Stage 1: detector
# ---------------------------------------------------------------------------

def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def head(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, out_channels, 1))


class Detector(nn.Module):
    """Stage 1 detector: it has no identity head and never consumes identity labels."""

    def __init__(self, base_channels=32):
        super().__init__()
        self.backbone = JointBackbone(base_ch=base_channels)
        channels = self.backbone.out_channels
        self.hm_head = head(channels, 1)
        self.wh_head = head(channels, 2)
        self.reg_head = head(channels, 2)
        self.hm_head[-1].bias.data.fill_(-2.19)

    def forward(self, images):
        features = self.backbone(images)
        return {
            "hm": torch.sigmoid(self.hm_head(features)),
            "wh": self.wh_head(features),
            "reg": self.reg_head(features),
        }


def detection_loss(outputs, targets):
    prediction = outputs["hm"][:, 0].clamp(1e-6, 1 - 1e-6)
    target = targets["hm"][:, 0]
    positive = (target == 1).float()
    negative = (target < 1).float()
    heatmap = (-positive * (1 - prediction).pow(2) * prediction.log() - negative * (1 - target).pow(4) * prediction.pow(2) * (1 - prediction).log()).sum() / positive.sum().clamp(min=1)

    batch_size, _, height, width = outputs["reg"].shape
    indices = (targets["centers"][..., 1] * width + targets["centers"][..., 0]).clamp(0, height * width - 1)
    gathered = {}
    for name in ("reg", "wh"):
        flat = outputs[name].permute(0, 2, 3, 1).reshape(batch_size, height * width, 2)
        gathered[name] = torch.gather(flat, 1, indices.unsqueeze(-1).expand(-1, -1, 2))
    mask = targets["mask"]
    if mask.any():
        offset = F.l1_loss(gathered["reg"][mask], targets["reg"][mask])
        size = F.l1_loss(gathered["wh"][mask], targets["wh"][mask])
    else:
        offset = outputs["reg"].sum() * 0
        size = outputs["wh"].sum() * 0
    total = heatmap + offset + 0.1 * size
    return total, {"total": float(total.detach()), "heatmap": float(heatmap.detach()), "offset": float(offset.detach()), "size": float(size.detach())}


def box_iou(first, second):
    def corners(box):
        cx, cy, width, height = box
        return cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2

    ax1, ay1, ax2, ay2 = corners(first)
    bx1, by1, bx2, by2 = corners(second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return intersection / max(area_a + area_b - intersection, 1e-9)


@torch.no_grad()
def decode(outputs, image_size, stride, score_threshold=0.15, top_k=50):
    scores = outputs["hm"][0, 0]
    pooled = F.max_pool2d(scores[None, None], 3, stride=1, padding=1)[0, 0]
    ys, xs = torch.where((scores == pooled) & (scores >= score_threshold))
    if not len(xs):
        return []
    scores_at_peaks = scores[ys, xs]
    keep = torch.topk(scores_at_peaks, min(top_k, len(scores_at_peaks))).indices
    detections = []
    for index in keep:
        x, y = xs[index].item(), ys[index].item()
        offset_x, offset_y = outputs["reg"][0, :, y, x].tolist()
        width, height = outputs["wh"][0, :, y, x].abs().tolist()
        detections.append((float(scores_at_peaks[index]), (x + offset_x) * stride, (y + offset_y) * stride, max(1.0, width * stride), max(1.0, height * stride)))
    return detections


def fit_detector(model, train_loader, val_loader, device, epochs, learning_rate, output_dir):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    output_dir.mkdir(parents=True, exist_ok=True)
    history, best_val, best_state = [], float("inf"), None
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        running, batches = {key: 0.0 for key in ("total", "heatmap", "offset", "size")}, 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = {key: value.to(device) for key, value in targets.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, values = detection_loss(model(images), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for key in running:
                running[key] += values[key]
            batches += 1
        train = {key: value / max(batches, 1) for key, value in running.items()}
        model.eval()
        val_total, val_batches = 0.0, 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = {key: value.to(device) for key, value in targets.items()}
                val_total += detection_loss(model(images), targets)[1]["total"]
                val_batches += 1
        val_total /= max(val_batches, 1)
        record = {"epoch": epoch, "train_total": train["total"], "train_heatmap": train["heatmap"], "train_offset": train["offset"], "train_size": train["size"], "val_total": val_total, "seconds": time.perf_counter() - started, "lr": optimizer.param_groups[0]["lr"]}
        history.append(record)
        if val_total < best_val:
            best_val = val_total
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"[Detection] epoch {epoch:03d}/{epochs:03d} | train={record['train_total']:.4f} hm={record['train_heatmap']:.4f} off={record['train_offset']:.4f} size={record['train_size']:.4f} | val={val_total:.4f} | time={record['seconds']:.1f}s")
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), output_dir / "detector.pt")
    with (output_dir / "detector_history.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    return history


def image_tensor(image_path, image_size):
    image = Image.open(image_path).convert("RGB").resize(image_size)
    raw = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255.0
    return raw, (raw - MEAN) / STD


def _save_crop(tensor, path):
    """Write a 0-1 range CHW crop tensor to disk as a JPEG for visual inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (tensor.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(array).save(path)


def predicted_crops(detector, dataset, device, image_size, stride, score_threshold, match_iou, save_dir=None):
    """Create ReID examples from detector predictions matched to GT identities for supervision.

    If ``save_dir`` is given, every matched crop is also written there as a JPEG so you can
    visually confirm the detector is actually finding the animals it claims to.
    """
    detector.eval()
    crops, labels = [], []
    with torch.no_grad():
        for image_path, boxes in dataset.samples:
            raw, normalized = image_tensor(image_path, image_size)
            outputs = detector(normalized.unsqueeze(0).to(device))
            detections = decode(outputs, image_size, stride, score_threshold)
            ground_truth = [(cx * image_size[0], cy * image_size[1], width * image_size[0], height * image_size[1], identity) for identity, cx, cy, width, height in boxes]
            used = set()
            for score, px, py, width, height in detections:
                matches = [(box_iou((px, py, width, height), gt[:4]), index) for index, gt in enumerate(ground_truth) if index not in used]
                if not matches:
                    continue
                overlap, index = max(matches)
                if overlap < match_iou:
                    continue
                used.add(index)
                x1, y1 = max(0, int(px - width / 2)), max(0, int(py - height / 2))
                x2, y2 = min(image_size[0], int(px + width / 2)), min(image_size[1], int(py + height / 2))
                if x2 > x1 and y2 > y1:
                    identity = ground_truth[index][4]
                    crop = F.interpolate(raw[:, y1:y2, x1:x2][None], size=(CROP_SIZE, CROP_SIZE), mode="bilinear", align_corners=False)[0]
                    crops.append(crop)
                    labels.append(identity)
                    if save_dir is not None:
                        _save_crop(crop, save_dir / f"{image_path.stem}_id{identity}_score{score:.2f}_iou{overlap:.2f}.jpg")
    return crops, labels


def ground_truth_crops(dataset, image_size, save_dir=None):
    """Bootstrap Stage 2 when a very short detector run has no usable predictions."""
    crops, labels = [], []
    for image_path, boxes in dataset.samples:
        raw, _normalized = image_tensor(image_path, image_size)
        for identity, cx, cy, width, height in boxes:
            x1, y1 = max(0, int((cx - width / 2) * image_size[0])), max(0, int((cy - height / 2) * image_size[1]))
            x2, y2 = min(image_size[0], int((cx + width / 2) * image_size[0])), min(image_size[1], int((cy + height / 2) * image_size[1]))
            if x2 > x1 and y2 > y1:
                crop = F.interpolate(raw[:, y1:y2, x1:x2][None], size=(CROP_SIZE, CROP_SIZE), mode="bilinear", align_corners=False)[0]
                crops.append(crop)
                labels.append(identity)
                if save_dir is not None:
                    _save_crop(crop, save_dir / f"{image_path.stem}_id{identity}.jpg")
    return crops, labels


class CropDataset(Dataset):
    def __init__(self, crops, labels):
        self.crops, self.labels = crops, labels

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, index):
        return (self.crops[index] - MEAN) / STD, self.labels[index]


class ReIDModel(nn.Module):
    def __init__(self, n_identities, embedding_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embedding = nn.Linear(128, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, n_identities)

    def forward(self, images):
        embedding = F.normalize(self.embedding(self.features(images).flatten(1)), dim=1)
        return self.classifier(embedding), embedding


def fit_reid(model, train_loader, val_loader, device, epochs, learning_rate, output_dir):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history, best_accuracy, best_state = [], -1.0, None
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        loss_sum, correct, count, batches = 0.0, 0, 0, 0
        for crops, labels in train_loader:
            crops, labels = crops.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(crops)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            correct += int((logits.argmax(1) == labels).sum())
            count += labels.numel()
            batches += 1
        model.eval()
        val_correct = val_count = 0
        with torch.no_grad():
            for crops, labels in val_loader:
                logits, _ = model(crops.to(device))
                val_correct += int((logits.argmax(1) == labels.to(device)).sum())
                val_count += labels.numel()
        val_accuracy = 100.0 * val_correct / max(val_count, 1)
        record = {"epoch": epoch, "loss": loss_sum / max(batches, 1), "train_accuracy": 100.0 * correct / max(count, 1), "val_accuracy": val_accuracy, "seconds": time.perf_counter() - started}
        history.append(record)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"[ReID] epoch {epoch:03d}/{epochs:03d} | loss={record['loss']:.4f} | train_acc={record['train_accuracy']:.2f}% | val_acc={val_accuracy:.2f}% | time={record['seconds']:.1f}s")
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), output_dir / "reid.pt")
    with (output_dir / "reid_history.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    return history


@torch.no_grad()
def two_stage_inference(detector, reid_model, dataset, device, image_size, stride, score_threshold):
    detector.eval()
    reid_model.eval()
    totals = {"detector": 0.0, "decode_crop": 0.0, "reid": 0.0, "total": 0.0}
    frames, detections = 0, 0
    for image_path, _boxes in dataset.samples:
        raw, normalized = image_tensor(image_path, image_size)
        start_total = time.perf_counter()
        sync(device)
        detector_start = time.perf_counter()
        outputs = detector(normalized.unsqueeze(0).to(device))
        sync(device)
        totals["detector"] += time.perf_counter() - detector_start
        crop_start = time.perf_counter()
        decoded = decode(outputs, image_size, stride, score_threshold)
        crops = []
        for _score, px, py, width, height in decoded:
            x1, y1 = max(0, int(px - width / 2)), max(0, int(py - height / 2))
            x2, y2 = min(image_size[0], int(px + width / 2)), min(image_size[1], int(py + height / 2))
            if x2 > x1 and y2 > y1:
                crops.append(F.interpolate(raw[:, y1:y2, x1:x2][None], size=(CROP_SIZE, CROP_SIZE), mode="bilinear", align_corners=False)[0])
        totals["decode_crop"] += time.perf_counter() - crop_start
        reid_start = time.perf_counter()
        if crops:
            sync(device)
            reid_model(((torch.stack(crops) - MEAN) / STD).to(device))
            sync(device)
        totals["reid"] += time.perf_counter() - reid_start
        totals["total"] += time.perf_counter() - start_total
        frames += 1
        detections += len(crops)
    per_frame = {key: value / max(frames, 1) for key, value in totals.items()}
    print(f"[Inference] frames={frames} detections={detections} | detector={per_frame['detector'] * 1000:.2f} ms/frame | decode+crop={per_frame['decode_crop'] * 1000:.2f} ms/frame | reid={per_frame['reid'] * 1000:.2f} ms/frame | total two-stage={per_frame['total'] * 1000:.2f} ms/frame")
    return {"frames": frames, "detections": detections, **per_frame}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description="Train and benchmark sequential detection then ReID.")
    parser.add_argument("--data-root", type=Path, default=Path("benchmarks/dataset-mmreplica"), help="Dataset root (2 classes: cow, buffalo; 15 identities each = 30 total).")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/mmreplica-sequential"))
    parser.add_argument("--detector-epochs", type=int, default=10)
    parser.add_argument("--reid-epochs", type=int, default=10)
    parser.add_argument("--detector-model", type=Path, default=None, help="Load detector weights (state_dict) before training, e.g. to skip Stage 1 or fine-tune it.")
    parser.add_argument("--reid-model", type=Path, default=None, help="Load ReID weights (state_dict) before training, e.g. to fine-tune Stage 2.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.15)
    parser.add_argument("--match-iou", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--save-crops-dir", type=Path, default=None, help="If set, save Stage-2 crops here as JPEGs to visually confirm the detector is finding the animals.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main(args=None):
    config = build_parser().parse_args(args)
    device = torch.device(config.device)
    image_size = (config.width, config.height)
    class_names = read_classes(config.data_root)
    identity_map = read_identities(config.data_root)
    n_identities = len(identity_map) if identity_map else max(len(class_names), 1)

    train_dataset = WebotsJDEDataset(config.data_root, "train", image_size, config.stride, n_identities)
    val_dataset = WebotsJDEDataset(config.data_root, "val", image_size, config.stride, n_identities)
    test_dataset = WebotsJDEDataset(config.data_root, "test", image_size, config.stride, n_identities)

    loader_args = {"batch_size": config.batch_size, "num_workers": config.workers, "collate_fn": jde_collate}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    detector = Detector().to(device)
    if config.detector_model is not None:
        detector.load_state_dict(torch.load(config.detector_model, map_location=device))
        print(f"Loaded detector weights from {config.detector_model}")
    stage_start = time.perf_counter()
    fit_detector(detector, train_loader, val_loader, device, config.detector_epochs, config.learning_rate, config.output_dir)
    detection_training_seconds = time.perf_counter() - stage_start
    print(f"[Stage 1 complete] detection training time={detection_training_seconds:.1f}s")

    crop_start = time.perf_counter()
    train_crop_dir = config.save_crops_dir / "train_predicted" if config.save_crops_dir else None
    val_crop_dir = config.save_crops_dir / "val_predicted" if config.save_crops_dir else None
    train_crops, train_labels = predicted_crops(detector, train_dataset, device, image_size, config.stride, config.score_threshold, config.match_iou, save_dir=train_crop_dir)
    val_crops, val_labels = predicted_crops(detector, val_dataset, device, image_size, config.stride, config.score_threshold, config.match_iou, save_dir=val_crop_dir)
    crop_seconds = time.perf_counter() - crop_start
    print(f"[Crop generation] train_crops={len(train_crops)} val_crops={len(val_crops)} time={crop_seconds:.1f}s")
    if config.save_crops_dir:
        print(f"[Crop generation] saved crop images under {config.save_crops_dir}")
    if not train_crops or not val_crops:
        print("[Crop generation] detector produced no matched crops; bootstrapping ReID with ground-truth crops.")
        train_crops, train_labels = ground_truth_crops(train_dataset, image_size, save_dir=(config.save_crops_dir / "train_ground_truth" if config.save_crops_dir else None))
        val_crops, val_labels = ground_truth_crops(val_dataset, image_size, save_dir=(config.save_crops_dir / "val_ground_truth" if config.save_crops_dir else None))
        if not train_crops or not val_crops:
            raise RuntimeError("No crops available for ReID training.")

    crop_train_loader = DataLoader(CropDataset(train_crops, train_labels), batch_size=config.batch_size, shuffle=True, num_workers=config.workers)
    crop_val_loader = DataLoader(CropDataset(val_crops, val_labels), batch_size=config.batch_size, shuffle=False, num_workers=config.workers)
    reid = ReIDModel(n_identities).to(device)
    if config.reid_model is not None:
        reid.load_state_dict(torch.load(config.reid_model, map_location=device))
        print(f"Loaded ReID weights from {config.reid_model}")
    stage_start = time.perf_counter()
    fit_reid(reid, crop_train_loader, crop_val_loader, device, config.reid_epochs, config.learning_rate, config.output_dir)
    reid_training_seconds = time.perf_counter() - stage_start
    print(f"[Stage 2 complete] ReID training time={reid_training_seconds:.1f}s")

    timing = two_stage_inference(detector, reid, test_dataset, device, image_size, config.stride, config.score_threshold)
    print(f"[Total training] two stages={detection_training_seconds + crop_seconds + reid_training_seconds:.1f}s (detection + crop generation + ReID)")
    torch.save({"detector": detector.state_dict(), "reid": reid.state_dict(), "timing": timing, "n_identities": n_identities}, config.output_dir / "pipeline.pt")


if __name__ == "__main__":
    main()


# Demo:
# python sequence.py --data-root benchmarks/dataset-mmreplica --detector-epochs 30 --reid-epochs 30 --batch-size 8 --learning-rate 1e-4 --output-dir runs/mmreplica-sequential --device cuda
# Fine-tune from prior weights:
# python sequence.py --data-root benchmarks/dataset-mmreplica --detector-model runs/mmreplica-sequential/detector.pt --reid-model runs/mmreplica-sequential/reid.pt --output-dir runs/mmreplica-sequential-finetune
# Save Stage-2 crops as JPEGs to visually confirm detection quality:
# python sequence.py --data-root benchmarks/dataset-mmreplica --output-dir runs/mmreplica-sequential --save-crops-dir runs/mmreplica-sequential/crops
