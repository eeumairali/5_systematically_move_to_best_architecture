"""Training loop, closed-set identity accuracy, and detection precision/recall evaluation."""
import csv
import time
from pathlib import Path

import torch

from n10_funs.config import DEVICE, N_IDENTITIES, STRIDE
from n10_funs.decode import box_iou, decode_detections
from n10_funs.losses import joint_loss


@torch.no_grad()
def evaluate_identity_head(model, loader):
    """Closed-set accuracy of the re-ID head at GT centers.
    FLAGGED DEVIATION from the paper: FairMOT evaluates re-ID with OPEN-set retrieval (TPR@FAR,
    IDF1 -- see n8's notebook for that style of evaluation) because MOT test identities are not
    seen in training. mmCows' 16 cows are a fixed, closed herd -- every cow in val was also seen
    (with different images) in train -- so plain classification accuracy is the meaningful, honest
    metric here, not a retrieval metric that would be testing something we didn't set up for."""
    model.eval()
    correct_id, total = 0, 0
    for x, hm, off, sz, ctr, ids, mask in loader:
        x, ctr, ids, mask = x.to(DEVICE), ctr.to(DEVICE), ids.to(DEVICE), mask.to(DEVICE)
        outputs = model(x)
        reid_embed = model.gather_embeddings(outputs["reid_map"], ctr, mask)
        if reid_embed.numel() == 0:
            continue
        id_pred = model.identity_fc(reid_embed).argmax(1)
        correct_id += (id_pred == ids[mask]).sum().item()
        total += mask.sum().item()
    if total == 0:
        return 0.0
    return 100 * correct_id / total


def fit(model, train_loader, val_loader, epochs=25, lr=1e-4, lr_drop_epoch=15):
    """Train the model and return one clearly labelled metrics record per epoch."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    log_path = Path("runs") / "n10_joint_det_reid" / "history.csv"
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
        running = {"classification": 0.0, "off": 0.0, "size": 0.0, "identity": 0.0, "total": 0.0}
        train_correct_id, train_total_id, n_batches = 0, 0, 0
        for x, hm, off, sz, ctr, ids, mask in train_loader:
            x, hm = x.to(DEVICE), hm.to(DEVICE)
            off, sz, ctr = off.to(DEVICE), sz.to(DEVICE), ctr.to(DEVICE)
            ids, mask = ids.to(DEVICE), mask.to(DEVICE)

            outputs = model(x)
            loss, parts = joint_loss(model, outputs, hm, off, sz, ctr, ids, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for key in running:
                running[key] += parts[key]
            with torch.no_grad():
                embeddings = model.gather_embeddings(outputs["reid_map"], ctr, mask)
                if embeddings.numel():
                    predicted_ids = model.identity_fc(embeddings).argmax(1)
                    train_correct_id += (predicted_ids == ids[mask]).sum().item()
                    train_total_id += mask.sum().item()
            n_batches += 1

        for key in running:
            running[key] /= max(n_batches, 1)

        val_identity_acc = evaluate_identity_head(model, val_loader)
        epoch_seconds = time.perf_counter() - epoch_start
        gpu_mem_mb = torch.cuda.max_memory_allocated(DEVICE) / 1e6 if DEVICE.type == "cuda" else 0.0
        rec = {
            "epoch": epoch,
            "total": running["total"],
            "classification": running["classification"],
            "detection": running["off"] + running["size"],
            "off": running["off"],
            "size": running["size"],
            "identity": running["identity"],
            "train_identity_acc": 100 * train_correct_id / max(train_total_id, 1),
            "val_identity_acc": val_identity_acc,
            "epoch_seconds": epoch_seconds,
            "lr": optimizer.param_groups[0]["lr"],
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
            f"epoch {epoch:>3}/{epochs:<3} | "
            f"total {rec['total']:>8.3f} | cls {rec['classification']:>8.3f} | "
            f"det {rec['detection']:>8.3f} | re-id {rec['identity']:>8.3f} | "
            f"train_acc {rec['train_identity_acc']:>6.1f}% | val_acc {rec['val_identity_acc']:>6.1f}% | "
            f"time {rec['epoch_seconds']:>7.1f}s | lr {rec['lr']:.1e} | gpu {gpu_mem_text}"
        )
    return history


@torch.no_grad()
def evaluate_detection(model, loader, iou_thresh=0.5, score_thresh=0.3):
    model.eval()
    tp, fp, fn = 0, 0, 0
    for x, hm, off, sz, ctr, ids, mask in loader:
        x = x.to(DEVICE)
        outputs = model(x)
        for b in range(x.shape[0]):
            dets = decode_detections(outputs["classification"][b, 0], outputs["offset"][b],
                                      outputs["size"][b], score_thresh=score_thresh)
            gt_boxes = []
            for i in range(mask.shape[1]):
                if not mask[b, i]:
                    continue
                fx, fy = ctr[b, i].tolist()
                w, h = sz[b, i].tolist()
                gt_boxes.append((fx * STRIDE, fy * STRIDE, w * STRIDE, h * STRIDE))
            matched_gt = set()
            for score, px, py, w, h, fx, fy in dets:
                best_iou, best_j = 0.0, -1
                for j, gb in enumerate(gt_boxes):
                    if j in matched_gt:
                        continue
                    iou = box_iou((px, py, w, h), gb)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= iou_thresh:
                    tp += 1
                    matched_gt.add(best_j)
                else:
                    fp += 1
            fn += len(gt_boxes) - len(matched_gt)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return precision, recall, tp, fp, fn
