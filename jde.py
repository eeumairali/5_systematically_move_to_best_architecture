"""Joint detection + ReID (JDE) pipeline: dataset loading, model, and training.

Run this file directly with CLI flags - see ``build_parser()`` for all options,
or the demo command at the bottom of this file.
"""

import argparse
import csv
import os
import tempfile
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
# Model, losses, and training loop
# ---------------------------------------------------------------------------

def _head(in_channels, hidden_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, hidden_channels, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(hidden_channels, out_channels, 1))


class JDEModel(nn.Module):
    """Shared backbone with heatmap, box-regression, and identity heads."""

    def __init__(self, n_identities, reid_dim=128, base_channels=32):
        super().__init__()
        self.backbone = JointBackbone(base_ch=base_channels)
        channels = self.backbone.out_channels
        self.hm_head = _head(channels, 64, 1)
        self.wh_head = _head(channels, 64, 2)
        self.reg_head = _head(channels, 64, 2)
        self.id_head = _head(channels, 64, reid_dim)
        self.identity_classifier = nn.Linear(reid_dim, n_identities)
        self.log_vars = nn.Parameter(torch.zeros(3))
        self.hm_head[-1].bias.data.fill_(-2.19)

    def forward(self, images):
        features = self.backbone(images)
        return {"hm": torch.sigmoid(self.hm_head(features)), "wh": self.wh_head(features), "reg": self.reg_head(features), "id": self.id_head(features)}

    @staticmethod
    def gather(feature_map, centers):
        _, channels, _, width = feature_map.shape
        flattened = feature_map.permute(0, 2, 3, 1).reshape(feature_map.shape[0], -1, channels)
        indices = centers[..., 1] * width + centers[..., 0]
        return torch.gather(flattened, 1, indices.unsqueeze(-1).expand(-1, -1, channels))


def focal_loss(prediction, target):
    prediction = prediction[:, 0].clamp(1e-6, 1 - 1e-6)
    positive = (target[:, 0] == 1).float()
    negative = (target[:, 0] < 1).float()
    positive_loss = -positive * (1 - prediction).pow(2) * prediction.log()
    negative_loss = -negative * (1 - target[:, 0]).pow(4) * prediction.pow(2) * (1 - prediction).log()
    return (positive_loss.sum() + negative_loss.sum()) / positive.sum().clamp(min=1)


def joint_loss(model, outputs, targets):
    heatmap_loss = focal_loss(outputs["hm"], targets["hm"])
    batch_size, _, height, width = outputs["reg"].shape
    flat_reg = outputs["reg"].permute(0, 2, 3, 1).reshape(batch_size, height * width, 2)
    flat_wh = outputs["wh"].permute(0, 2, 3, 1).reshape(batch_size, height * width, 2)
    indices = (targets["centers"][..., 1] * width + targets["centers"][..., 0]).clamp(0, height * width - 1)
    gathered_reg = torch.gather(flat_reg, 1, indices.unsqueeze(-1).expand(-1, -1, 2))
    gathered_wh = torch.gather(flat_wh, 1, indices.unsqueeze(-1).expand(-1, -1, 2))
    mask = targets["mask"]
    offset_loss = F.l1_loss(gathered_reg[mask], targets["reg"][mask]) if mask.any() else outputs["reg"].sum() * 0
    size_loss = F.l1_loss(gathered_wh[mask], targets["wh"][mask]) if mask.any() else outputs["wh"].sum() * 0
    embeddings = model.gather(outputs["id"], targets["centers"])[mask]
    identity_loss = F.cross_entropy(model.identity_classifier(embeddings), targets["ids"][mask]) if embeddings.numel() else outputs["id"].sum() * 0
    detection_loss = offset_loss + 0.1 * size_loss
    components = torch.stack([heatmap_loss, detection_loss, identity_loss])
    total = (torch.exp(-model.log_vars) * components + model.log_vars).mean()
    values = {"total": float(total.detach()), "heatmap": float(heatmap_loss.detach()), "offset": float(offset_loss.detach()), "size": float(size_loss.detach()), "detection": float(detection_loss.detach()), "identity": float(identity_loss.detach())}
    return total, values


def _move_targets(targets, device):
    return {key: value.to(device) for key, value in targets.items()}


def _atomic_torch_save(value, path):
    """Write a checkpoint through a temporary file so a crash cannot leave a partial file."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as file:
        temporary_path = Path(file.name)
    try:
        torch.save(value, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def benchmark_inference(model, dataset, device):
    """Single-image forward-pass latency: one call gets heatmap + offset + size + the full
    ReID embedding map, since that is the whole point of the joint architecture (contrast with
    the sequential pipeline's separate detector-forward + crop/decode + ReID-forward)."""
    model.eval()
    total_seconds = 0.0
    for index in range(len(dataset)):
        image, _targets = dataset[index]
        image = image.unsqueeze(0).to(device)
        _sync(device)
        started = time.perf_counter()
        model(image)
        _sync(device)
        total_seconds += time.perf_counter() - started
    frames = len(dataset)
    ms_per_frame = (total_seconds / max(frames, 1)) * 1000.0
    fps = 1000.0 / ms_per_frame if ms_per_frame > 0 else float("inf")
    print(f"[Inference] frames={frames} | joint forward-pass={ms_per_frame:.2f} ms/frame | {fps:.1f} FPS")
    return {"frames": frames, "ms_per_frame": ms_per_frame, "fps": fps}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    totals = {key: 0.0 for key in ("total", "heatmap", "offset", "size", "detection", "identity")}
    correct, count, batches = 0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), _move_targets(targets, device)
        outputs = model(images)
        _, values = joint_loss(model, outputs, targets)
        for key in totals:
            totals[key] += values[key]
        embeddings = model.gather(outputs["id"], targets["centers"])[targets["mask"]]
        if embeddings.numel():
            predictions = model.identity_classifier(embeddings).argmax(1)
            correct += int((predictions == targets["ids"][targets["mask"]]).sum())
            count += int(targets["mask"].sum())
        batches += 1
    for key in totals:
        totals[key] /= max(batches, 1)
    totals["identity_accuracy"] = 100.0 * correct / max(count, 1)
    return totals


def fit(model, train_loader, val_loader, device, epochs, learning_rate, output_dir, grad_clip=5.0, patience=5, resume=None):
    if patience < 1:
        raise ValueError("patience must be at least 1")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    history_path = output_dir / "history.csv"
    best_path = output_dir / "best.pt"
    history = []
    best_val_total = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = None
    start_epoch = 1
    if resume is not None:
        resume_path = output_dir / "latest.pt" if resume is True else Path(resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if history_path.exists():
            with history_path.open(newline="") as file:
                history = list(csv.DictReader(file))
        if best_path.exists():
            best_checkpoint = torch.load(best_path, map_location="cpu")
            best_state = best_checkpoint["model"]
            best_epoch = int(best_checkpoint["epoch"])
            best_val_total = float(best_checkpoint["metrics"]["val_total"])
            epochs_without_improvement = max(0, int(checkpoint["epoch"]) - best_epoch)
        else:
            best_epoch = int(checkpoint["epoch"])
            best_val_total = float(checkpoint["metrics"]["val_total"])
            epochs_without_improvement = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            _atomic_torch_save(checkpoint, best_path)
        print(f"Resuming from epoch {checkpoint['epoch']}; next epoch is {start_epoch}.")
    for epoch in range(start_epoch, epochs + 1):
        started = time.perf_counter()
        model.train()
        running = {key: 0.0 for key in ("total", "heatmap", "offset", "size", "detection", "identity")}
        correct, count, batches = 0, 0, 0
        for images, targets in train_loader:
            images, targets = images.to(device), _move_targets(targets, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss, values = joint_loss(model, outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            for key in running:
                running[key] += values[key]
            embeddings = model.gather(outputs["id"], targets["centers"])[targets["mask"]]
            if embeddings.numel():
                correct += int((model.identity_classifier(embeddings).argmax(1) == targets["ids"][targets["mask"]]).sum())
                count += int(targets["mask"].sum())
            batches += 1
        train_metrics = {key: value / max(batches, 1) for key, value in running.items()}
        train_metrics["identity_accuracy"] = 100.0 * correct / max(count, 1)
        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.perf_counter() - started
        record = {"epoch": epoch, "train_total": train_metrics["total"], "train_heatmap": train_metrics["heatmap"], "train_detection": train_metrics["detection"], "train_offset": train_metrics["offset"], "train_size": train_metrics["size"], "train_identity": train_metrics["identity"], "train_id_accuracy": train_metrics["identity_accuracy"], "val_total": val_metrics["total"], "val_heatmap": val_metrics["heatmap"], "val_detection": val_metrics["detection"], "val_identity": val_metrics["identity"], "val_id_accuracy": val_metrics["identity_accuracy"], "learning_rate": optimizer.param_groups[0]["lr"], "seconds": elapsed}
        history.append(record)
        with history_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=record.keys())
            writer.writeheader()
            writer.writerows(history)
            file.flush()
            os.fsync(file.fileno())
        scheduler.step()
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "metrics": record}
        _atomic_torch_save(checkpoint, output_dir / "latest.pt")
        if record["val_total"] < best_val_total:
            best_val_total = record["val_total"]
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            _atomic_torch_save(checkpoint, best_path)
            improvement = " | best"
        else:
            epochs_without_improvement += 1
            improvement = f" | no improvement {epochs_without_improvement}/{patience}"
        print(f"Epoch {epoch:03d}/{epochs:03d} | train total={record['train_total']:.4f} hm={record['train_heatmap']:.4f} det={record['train_detection']:.4f} id={record['train_identity']:.4f} acc={record['train_id_accuracy']:.2f}% | val total={record['val_total']:.4f} hm={record['val_heatmap']:.4f} det={record['val_detection']:.4f} id={record['val_identity']:.4f} acc={record['val_id_accuracy']:.2f}% | lr={record['learning_rate']:.2e} time={elapsed:.1f}s")
        print(f"best_val={best_val_total:.4f} (epoch {best_epoch}){improvement}")
        print("=" * 120)
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}: validation loss did not improve for {patience} epochs.")
            break
    if best_state is None:
        raise RuntimeError("No epoch completed; no best checkpoint is available.")
    model.load_state_dict(best_state)
    _atomic_torch_save(model.state_dict(), output_dir / "model.pt")
    print(f"Restored best checkpoint from epoch {best_epoch} (val_total={best_val_total:.4f}).")
    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description="Train a CenterNet/FairMOT-inspired JDE model.")
    parser.add_argument("--data-root", type=Path, default=Path("benchmarks/dataset-mmreplica"), help="Dataset root (2 classes: cow, buffalo; 15 identities each = 30 total).")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--resume", nargs="?", const=True, default=None, help="Resume from output-dir/latest.pt, or provide a checkpoint path.")
    parser.add_argument("--model", type=Path, default=None, help="Load model weights (state_dict) before training, e.g. to fine-tune from a prior run's model.pt.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/mmreplica-jde"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main(args=None):
    config = build_parser().parse_args(args)
    if not config.data_root.exists():
        raise FileNotFoundError(f"Dataset does not exist: {config.data_root}")
    device = torch.device(config.device)
    class_names = read_classes(config.data_root)
    identity_map = read_identities(config.data_root)
    n_identities = len(identity_map) if identity_map else max(len(class_names), 1)
    image_size = (config.width, config.height)

    train_dataset = WebotsJDEDataset(config.data_root, "train", image_size, config.stride, n_identities)
    val_split = "val" if (config.data_root / "val.txt").exists() else "test"
    val_dataset = WebotsJDEDataset(config.data_root, val_split, image_size, config.stride, n_identities)

    loader_kwargs = {"batch_size": config.batch_size, "num_workers": config.workers, "collate_fn": jde_collate}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = JDEModel(n_identities=n_identities).to(device)
    if config.model is not None:
        model.load_state_dict(torch.load(config.model, map_location=device))
        print(f"Loaded model weights from {config.model}")

    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Dataset: {config.data_root} | train={len(train_dataset)} | val={len(val_dataset)}")
    print(f"Identities: {n_identities} | parameters: {parameters:,} | device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    fit(model, train_loader, val_loader, device, config.epochs, config.learning_rate, config.output_dir, patience=config.patience, resume=config.resume)
    benchmark_inference(model, val_dataset, device)


if __name__ == "__main__":
    main()


# Demo:
# python jde.py --data-root benchmarks/dataset-mmreplica --epochs 30 --batch-size 8 --learning-rate 1e-4 --output-dir runs/mmreplica-jde --device cuda
# Resume a run:
# python jde.py --data-root benchmarks/dataset-mmreplica --output-dir runs/mmreplica-jde --resume --epochs 60
# Fine-tune from a prior model:
# python jde.py --data-root benchmarks/dataset-mmreplica --model runs/mmreplica-jde/model.pt --output-dir runs/mmreplica-jde-finetune
