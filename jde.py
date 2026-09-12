"""JDE model, joint losses, validation, and detailed epoch training."""

import csv
import os
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from n10_funs.model import JointBackbone


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
