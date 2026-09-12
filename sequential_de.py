"""Sequential detector -> predicted crops -> ReID training and inference.

Stage 1 trains only object detection. Stage 2 receives crops made from Stage 1's
predicted boxes and trains an identity classifier. Inference reports detector,
crop/decode, ReID, and total two-stage latency separately.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from data_load import MEAN, STD, WebotsJDEDataset, jde_collate, read_classes
from n10_funs.model import JointBackbone


RUN_ROOT = Path("runs/sequential_de")
CROP_SIZE = 128


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


def predicted_crops(detector, dataset, device, image_size, stride, score_threshold, match_iou):
    """Create ReID examples from detector predictions matched to GT identities for supervision."""
    detector.eval()
    crops, labels = [], []
    with torch.no_grad():
        for image_path, boxes in dataset.samples:
            raw, normalized = image_tensor(image_path, image_size)
            outputs = detector(normalized.unsqueeze(0).to(device))
            detections = decode(outputs, image_size, stride, score_threshold)
            ground_truth = [(cx * image_size[0], cy * image_size[1], width * image_size[0], height * image_size[1], identity) for identity, cx, cy, width, height in boxes]
            used = set()
            for _score, px, py, width, height in detections:
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
                    crops.append(F.interpolate(raw[:, y1:y2, x1:x2][None], size=(CROP_SIZE, CROP_SIZE), mode="bilinear", align_corners=False)[0])
                    labels.append(ground_truth[index][4])
    return crops, labels


def ground_truth_crops(dataset, image_size):
    """Bootstrap Stage 2 when a very short detector run has no usable predictions."""
    crops, labels = [], []
    for image_path, boxes in dataset.samples:
        raw, _normalized = image_tensor(image_path, image_size)
        for identity, cx, cy, width, height in boxes:
            x1, y1 = max(0, int((cx - width / 2) * image_size[0])), max(0, int((cy - height / 2) * image_size[1]))
            x2, y2 = min(image_size[0], int((cx + width / 2) * image_size[0])), min(image_size[1], int((cy + height / 2) * image_size[1]))
            if x2 > x1 and y2 > y1:
                crops.append(F.interpolate(raw[:, y1:y2, x1:x2][None], size=(CROP_SIZE, CROP_SIZE), mode="bilinear", align_corners=False)[0])
                labels.append(identity)
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


def build_parser():
    parser = argparse.ArgumentParser(description="Train and benchmark sequential detection then ReID.")
    parser.add_argument("--data-root", type=Path, default=Path("benchmarks/webots_mcid-Small500"))
    parser.add_argument("--output-dir", type=Path, default=RUN_ROOT)
    parser.add_argument("--detector-epochs", type=int, default=10)
    parser.add_argument("--reid-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.15)
    parser.add_argument("--match-iou", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main(args=None):
    config = build_parser().parse_args(args)
    device = torch.device(config.device)
    image_size = (config.width, config.height)
    class_names = read_classes(config.data_root)
    n_identities = max(len(class_names), 1)
    train_dataset = WebotsJDEDataset(config.data_root, "train", image_size, config.stride, n_identities)
    val_dataset = WebotsJDEDataset(config.data_root, "val", image_size, config.stride, n_identities)
    test_dataset = WebotsJDEDataset(config.data_root, "test", image_size, config.stride, n_identities)
    loader_args = {"batch_size": config.batch_size, "num_workers": config.workers, "collate_fn": jde_collate}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    detector = Detector().to(device)
    stage_start = time.perf_counter()
    fit_detector(detector, train_loader, val_loader, device, config.detector_epochs, config.learning_rate, config.output_dir)
    detection_training_seconds = time.perf_counter() - stage_start
    print(f"[Stage 1 complete] detection training time={detection_training_seconds:.1f}s")

    crop_start = time.perf_counter()
    train_crops, train_labels = predicted_crops(detector, train_dataset, device, image_size, config.stride, config.score_threshold, config.match_iou)
    val_crops, val_labels = predicted_crops(detector, val_dataset, device, image_size, config.stride, config.score_threshold, config.match_iou)
    crop_seconds = time.perf_counter() - crop_start
    print(f"[Crop generation] train_crops={len(train_crops)} val_crops={len(val_crops)} time={crop_seconds:.1f}s")
    if not train_crops or not val_crops:
        print("[Crop generation] detector produced no matched crops; bootstrapping ReID with ground-truth crops.")
        train_crops, train_labels = ground_truth_crops(train_dataset, image_size)
        val_crops, val_labels = ground_truth_crops(val_dataset, image_size)
        if not train_crops or not val_crops:
            raise RuntimeError("No crops available for ReID training.")
    crop_train_loader = DataLoader(CropDataset(train_crops, train_labels), batch_size=config.batch_size, shuffle=True, num_workers=config.workers)
    crop_val_loader = DataLoader(CropDataset(val_crops, val_labels), batch_size=config.batch_size, shuffle=False, num_workers=config.workers)
    reid = ReIDModel(n_identities).to(device)
    stage_start = time.perf_counter()
    fit_reid(reid, crop_train_loader, crop_val_loader, device, config.reid_epochs, config.learning_rate, config.output_dir)
    reid_training_seconds = time.perf_counter() - stage_start
    print(f"[Stage 2 complete] ReID training time={reid_training_seconds:.1f}s")
    timing = two_stage_inference(detector, reid, test_dataset, device, image_size, config.stride, config.score_threshold)
    print(f"[Total training] two stages={detection_training_seconds + crop_seconds + reid_training_seconds:.1f}s (detection + crop generation + ReID)")
    torch.save({"detector": detector.state_dict(), "reid": reid.state_dict(), "timing": timing, "n_identities": n_identities}, config.output_dir / "pipeline.pt")


if __name__ == "__main__":
    main()


# python sequential_de.py --detector-epochs 10 --reid-epochs 10 --batch-size 8
# python sequential_de.py --detector-epochs 30 --reid-epochs 30 --batch-size 8 --data-root benchmarks/webots_mcid