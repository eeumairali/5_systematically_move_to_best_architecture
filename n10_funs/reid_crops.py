"""n10b Stage 2: crop generation from a trained detector's own predictions, the crop-identity
classifier, and its training loop + end-to-end sequential-pipeline timing."""
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from n10_funs.config import DEVICE, IMG_H, IMG_W, N_IDENTITIES
from n10_funs.dataset import IMAGENET_MEAN, IMAGENET_STD
from n10_funs.decode import box_iou, decode_detections
from n10_funs.model import ConvBNReLU, ResBlock

RUN_DIR = Path("runs") / "n10b_sequential"

CROP_H, CROP_W = 128, 128  # cow crops resized to a fixed square for the Stage-2 classifier


@torch.no_grad()
def generate_crops(model, samples, iou_thresh=0.5, score_thresh=0.3):
    """This is the crop-generation step that makes the pipeline genuinely SEQUENTIAL: crops come
    from the trained detector's OWN predicted boxes (decode_detections), not from ground-truth
    boxes. A predicted box is only kept as a labelled training/validation crop for Stage 2 if it
    matches a GT box at IoU >= 0.5 -- that match supplies the identity label (the detector itself
    has no notion of identity). Any detector mistake (missed cow, double-detected cow, imprecise
    box) propagates into what Stage 2 sees, exactly like a real two-stage tracker pipeline."""
    model.eval()
    crops, labels = [], []
    for img_path, boxes in samples:
        img = Image.open(img_path).convert("RGB").resize((IMG_W, IMG_H))
        arr = np.array(img).astype("float32") / 255.0
        raw = torch.from_numpy(arr).permute(2, 0, 1)          # (3, H, W) in [0,1], for cropping
        x = ((raw - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0).to(DEVICE)  # normalized, for the model

        outputs = model(x)
        dets = decode_detections(outputs["classification"][0, 0], outputs["offset"][0], outputs["size"][0],
                                  score_thresh=score_thresh)

        gt_boxes = [(cx * IMG_W, cy * IMG_H, bw * IMG_W, bh * IMG_H, cid) for cid, cx, cy, bw, bh in boxes]
        matched_gt = set()
        for score, px, py, w, h, fx, fy in dets:
            best_iou, best_j = 0.0, -1
            for j, (gx, gy, gw, gh, _cid) in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                iou = box_iou((px, py, w, h), (gx, gy, gw, gh))
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou < iou_thresh:
                continue  # unmatched prediction: no identity label available, drop it
            matched_gt.add(best_j)
            cid = gt_boxes[best_j][4]

            x1 = int(max(0, px - w / 2)); y1 = int(max(0, py - h / 2))
            x2 = int(min(IMG_W, px + w / 2)); y2 = int(min(IMG_H, py + h / 2))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = raw[:, y1:y2, x1:x2]
            crop = F.interpolate(crop.unsqueeze(0), size=(CROP_H, CROP_W), mode="bilinear",
                                  align_corners=False).squeeze(0)
            crops.append(crop)
            labels.append(cid - 1)  # 1..16 -> 0..15
    return crops, labels


class CropDataset(Dataset):
    """Wraps pre-generated (crop, identity) pairs. Crops are already normalized-space RGB
    tensors of shape (3, CROP_H, CROP_W); no further transform is needed here."""

    def __init__(self, crops, labels):
        self.crops, self.labels = crops, labels

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        crop = (self.crops[idx] - IMAGENET_MEAN) / IMAGENET_STD
        return crop, self.labels[idx]


class CropIdentityClassifier(nn.Module):
    """Stage 2: a small standalone CNN that has never seen a full frame or a detection score --
    only cropped cow images and identity labels. Built from the same ConvBNReLU/ResBlock blocks
    as the detector's backbone, just narrower, since the input is a small fixed-size crop rather
    than a full frame."""

    def __init__(self, n_identities=N_IDENTITIES, base_ch=32):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4
        self.stem = ConvBNReLU(3, c1, k=7, stride=2)      # 128 -> 64
        self.stage1 = ResBlock(c1, c2, stride=2)           # 64 -> 32
        self.stage2 = ResBlock(c2, c3, stride=2)           # 32 -> 16
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c3, n_identities)

    def forward(self, x):
        feat = self.stage2(self.stage1(self.stem(x)))
        return self.fc(self.pool(feat).flatten(1))


def fit_reid_crops(model, train_loader, val_loader, epochs=25, lr=1e-4):
    """Stage 2 training loop: plain cross-entropy over crops, no detection loss anywhere here."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    log_path = RUN_DIR / "reid_history.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    history = []
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats(DEVICE)
        model.train()
        running_loss, correct, total, n_batches = 0.0, 0, 0, 0
        for crops, ids in train_loader:
            crops, ids = crops.to(DEVICE), ids.to(DEVICE)
            logits = model(crops)
            loss = F.cross_entropy(logits, ids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (logits.argmax(1) == ids).sum().item()
            total += ids.numel()
            n_batches += 1

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for crops, ids in val_loader:
                crops, ids = crops.to(DEVICE), ids.to(DEVICE)
                val_correct += (model(crops).argmax(1) == ids).sum().item()
                val_total += ids.numel()

        gpu_mem_mb = torch.cuda.max_memory_allocated(DEVICE) / 1e6 if DEVICE.type == "cuda" else 0.0
        rec = {
            "epoch": epoch, "loss": running_loss / max(n_batches, 1),
            "train_acc": 100 * correct / max(total, 1),
            "val_acc": 100 * val_correct / max(val_total, 1),
            "epoch_seconds": time.perf_counter() - epoch_start,
            "gpu_mem_mb": gpu_mem_mb,
        }
        history.append(rec)
        with open(log_path, "a", newline="") as history_file:
            writer = csv.DictWriter(history_file, fieldnames=list(rec.keys()))
            if history_file.tell() == 0:
                writer.writeheader()
            writer.writerow(rec)

        gpu_mem_text = f"{rec['gpu_mem_mb']:>7.0f}MB" if DEVICE.type == "cuda" else "   no GPU"
        print(f"epoch {epoch:>3}/{epochs:<3} | loss {rec['loss']:>7.3f} | "
              f"train_acc {rec['train_acc']:>6.1f}% | val_acc {rec['val_acc']:>6.1f}% | "
              f"time {rec['epoch_seconds']:>6.1f}s | gpu {gpu_mem_text}")
    return history


@torch.no_grad()
def sequential_pipeline_timing(detector, reid_classifier, samples, score_thresh=0.3):
    """End-to-end per-frame timing for the SEQUENTIAL pipeline: detector forward -> decode ->
    crop/resize each box -> reid forward on the batch of crops. This is what "sequential" costs
    that a single joint forward pass (n10) does not: two separate models and a crop step between
    them, run one frame at a time exactly as a deployed two-stage tracker would."""
    detector.eval(); reid_classifier.eval()
    total_seconds, n_frames, n_boxes = 0.0, 0, 0
    for img_path, boxes in samples:
        img = Image.open(img_path).convert("RGB").resize((IMG_W, IMG_H))
        arr = np.array(img).astype("float32") / 255.0
        raw = torch.from_numpy(arr).permute(2, 0, 1)
        x = ((raw - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0).to(DEVICE)

        start = time.perf_counter()
        outputs = detector(x)
        dets = decode_detections(outputs["classification"][0, 0], outputs["offset"][0], outputs["size"][0],
                                  score_thresh=score_thresh)
        crops = []
        for score, px, py, w, h, fx, fy in dets:
            x1 = int(max(0, px - w / 2)); y1 = int(max(0, py - h / 2))
            x2 = int(min(IMG_W, px + w / 2)); y2 = int(min(IMG_H, py + h / 2))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = raw[:, y1:y2, x1:x2]
            crop = F.interpolate(crop.unsqueeze(0), size=(CROP_H, CROP_W), mode="bilinear",
                                  align_corners=False).squeeze(0)
            crops.append((crop - IMAGENET_MEAN) / IMAGENET_STD)
        if crops:
            reid_classifier(torch.stack(crops).to(DEVICE))
        total_seconds += time.perf_counter() - start
        n_frames += 1
        n_boxes += len(crops)

    return total_seconds / max(n_frames, 1), n_frames, n_boxes


def save_sequential_checkpoint(detector, reid_classifier, detector_history, reid_history):
    from n10_funs.config import CAMERAS, DATASET_NAME, DATA_ROOT, FRAME_STRIDE, N_CLASSES, STRIDE
    from n10_funs.config import IMG_H as _IMG_H, IMG_W as _IMG_W

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = RUN_DIR / "pipeline.pt"
    torch.save({
        "detector_state": detector.state_dict(),
        "reid_state": reid_classifier.state_dict(),
        "detector_history": detector_history,
        "reid_history": reid_history,
        "config": {
            "dataset": DATASET_NAME, "data_root": str(DATA_ROOT), "cameras": list(CAMERAS),
            "img_w": _IMG_W, "img_h": _IMG_H, "stride": STRIDE, "crop_h": CROP_H, "crop_w": CROP_W,
            "n_classes": N_CLASSES, "n_identities": N_IDENTITIES, "frame_stride": FRAME_STRIDE,
        },
    }, checkpoint_path)
    print(f"saved sequential detector + reid classifier + histories + config to {checkpoint_path}")
    return checkpoint_path
