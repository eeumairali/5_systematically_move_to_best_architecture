"""n10b Stage 1: detector-only model (same backbone/heads as the joint model, minus re-ID) and
its training loop. No identity supervision anywhere in this file -- that is the entire point of
"sequential": detection is trained in isolation."""
import csv
import time
from pathlib import Path

import torch
from torch import nn

from n10_funs.config import DEVICE, N_CLASSES
from n10_funs.losses import detection_l1_loss, focal_classification_loss
from n10_funs.model import JointBackbone, make_head
from n10_funs.train import evaluate_detection

RUN_DIR = Path("runs") / "n10b_sequential"


class DetectorOnlyModel(nn.Module):
    """Stage 1 of the sequential pipeline: the same backbone/head shapes as the joint model,
    but WITHOUT a re-ID head. This is the true detector-only model -- nothing here has ever seen
    an identity label, so its weights cannot leak identity information into detection."""

    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        self.backbone = JointBackbone(base_ch=32)
        c = self.backbone.out_channels
        self.classification_head = make_head(c, 64, n_classes)
        self.offset_head = make_head(c, 64, 2)
        self.size_head = make_head(c, 64, 2)

    def forward(self, x):
        feat = self.backbone(x)
        return {
            "classification": torch.sigmoid(self.classification_head(feat)),
            "offset": self.offset_head(feat),
            "size": self.size_head(feat),
        }


def detector_loss(outputs, heatmap_gt, offset_gt, size_gt, centers, mask):
    """Classification + detection terms only (Eq. 1 + Eq. 2). No re-ID term exists at this
    stage -- that is the entire point of "sequential": detection is trained in isolation."""
    l_classification = focal_classification_loss(outputs["classification"], heatmap_gt)
    l_off, l_size = detection_l1_loss(outputs["offset"], outputs["size"], offset_gt, size_gt, centers, mask)
    total = l_classification + l_off + l_size
    return total, {"classification": l_classification.item(), "off": l_off.item(), "size": l_size.item(),
                    "total": total.item()}


def fit_detector(model, train_loader, val_loader, epochs=25, lr=1e-4, lr_drop_epoch=15):
    """Stage 1 training loop: detector only, no identity supervision anywhere in this function."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    log_path = RUN_DIR / "detector_history.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    history = []
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats(DEVICE)
        if epoch == lr_drop_epoch:
            for group in optimizer.param_groups:
                group["lr"] *= 0.1

        model.train()
        running = {"classification": 0.0, "off": 0.0, "size": 0.0, "total": 0.0}
        n_batches = 0
        for x, hm, off, sz, ctr, ids, mask in train_loader:
            x, hm = x.to(DEVICE), hm.to(DEVICE)
            off, sz, ctr, mask = off.to(DEVICE), sz.to(DEVICE), ctr.to(DEVICE), mask.to(DEVICE)

            outputs = model(x)
            loss, parts = detector_loss(outputs, hm, off, sz, ctr, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for key in running:
                running[key] += parts[key]
            n_batches += 1

        for key in running:
            running[key] /= max(n_batches, 1)

        precision, recall, tp, fp, fn = evaluate_detection(model, val_loader)
        epoch_seconds = time.perf_counter() - epoch_start
        gpu_mem_mb = torch.cuda.max_memory_allocated(DEVICE) / 1e6 if DEVICE.type == "cuda" else 0.0
        rec = {
            "epoch": epoch, "total": running["total"], "classification": running["classification"],
            "detection": running["off"] + running["size"], "off": running["off"], "size": running["size"],
            "val_precision": 100 * precision, "val_recall": 100 * recall,
            "epoch_seconds": epoch_seconds, "lr": optimizer.param_groups[0]["lr"],
            "gpu_mem_mb": gpu_mem_mb,
        }
        history.append(rec)
        with open(log_path, "a", newline="") as history_file:
            writer = csv.DictWriter(history_file, fieldnames=list(rec.keys()))
            if history_file.tell() == 0:
                writer.writeheader()
            writer.writerow(rec)

        gpu_mem_text = f"{rec['gpu_mem_mb']:>7.0f}MB" if DEVICE.type == "cuda" else "   no GPU"
        print(
            f"epoch {epoch:>3}/{epochs:<3} | total {rec['total']:>8.3f} | cls {rec['classification']:>8.3f} | "
            f"det {rec['detection']:>8.3f} | val_precision {rec['val_precision']:>6.1f}% | "
            f"val_recall {rec['val_recall']:>6.1f}% | time {rec['epoch_seconds']:>7.1f}s | lr {rec['lr']:.1e} | "
            f"gpu {gpu_mem_text}"
        )
    return history
