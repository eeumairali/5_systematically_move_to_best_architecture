"""Matplotlib plotting helpers: training curves, GT overlay, and qualitative predictions."""
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from n10_funs.config import DEVICE, IMG_H, IMG_W, N_IDENTITIES
from n10_funs.dataset import IMAGENET_MEAN, IMAGENET_STD, unnormalize
from n10_funs.decode import decode_detections


def plot_batch_targets(x0, hm0, ctr0, sz0, id0, stride):
    fig, axes = plt.subplots(3, 1, figsize=(8, 15))
    axes[0].imshow(unnormalize(x0[0])); axes[0].set_title("input frame (640x400)"); axes[0].axis("off")
    axes[1].imshow(hm0[0], cmap="hot"); axes[1].set_title("GT classification heatmap target (Eq. 1), 160x100")
    axes[2].imshow(unnormalize(x0[0]))
    n_boxes = int((sz0[0].abs().sum(dim=1) > 0).sum().item())
    for i in range(n_boxes):
        fx, fy = ctr0[0, i].tolist()
        w, h = sz0[0, i].tolist()
        px, py = fx * stride, fy * stride
        rect = plt.Rectangle((px - w * stride / 2, py - h * stride / 2), w * stride, h * stride,
                              fill=False, edgecolor="lime", linewidth=1.5)
        axes[2].add_patch(rect)
        axes[2].text(px, py, f"C{id0[0, i].item() + 1:02d}", color="yellow", fontsize=7)
    axes[2].set_title("GT boxes + identity"); axes[2].axis("off")
    plt.tight_layout(); plt.show()


def plot_history(history):
    epochs = [record["epoch"] for record in history]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(epochs, [record["total"] for record in history], label="total", color="C3")
    axes[0].plot(epochs, [record["classification"] for record in history], label="classification", color="C0")
    axes[0].set_title("training losses (lower is better)"); axes[0].legend(); axes[0].set_xlabel("epoch")
    axes[1].plot(epochs, [record["detection"] for record in history], label="detection", color="C1")
    axes[1].plot(epochs, [record["identity"] for record in history], label="re-ID", color="C2")
    axes[1].set_title("detection / re-ID losses"); axes[1].legend(); axes[1].set_xlabel("epoch")
    axes[2].plot(epochs, [record["train_identity_acc"] for record in history], label="train accuracy", color="C4")
    axes[2].plot(epochs, [record["val_identity_acc"] for record in history], label="validation accuracy", color="C2")
    axes[2].axhline(100 / N_IDENTITIES, ls="--", color="C2", alpha=0.5, label="random identity")
    axes[2].set_title("re-ID identity accuracy (higher is better)")
    axes[2].legend(); axes[2].set_xlabel("epoch"); axes[2].set_ylabel("percent")
    plt.tight_layout(); plt.show()


@torch.no_grad()
def predict_and_draw(model, dataset_samples, n_show=3):
    """Qualitative check: decoded detections + predicted identity on val frames."""
    model.eval()
    fig, axes = plt.subplots(1, n_show, figsize=(6 * n_show, 6))
    if n_show == 1:
        axes = [axes]
    for a, (img_path, boxes) in zip(axes, dataset_samples[:n_show]):
        img = Image.open(img_path).convert("RGB").resize((IMG_W, IMG_H))
        arr = np.array(img).astype("float32") / 255.0
        x = (torch.from_numpy(arr).permute(2, 0, 1) - IMAGENET_MEAN) / IMAGENET_STD
        outputs = model(x.unsqueeze(0).to(DEVICE))
        dets = decode_detections(outputs["classification"][0, 0], outputs["offset"][0], outputs["size"][0],
                                  score_thresh=0.3)
        a.imshow(arr)
        for score, px, py, w, h, fx, fy in dets:
            reid_vec = outputs["reid_map"][0, :, fy, fx].unsqueeze(0)
            pred_id = model.identity_fc(reid_vec).argmax(1).item() + 1
            a.add_patch(plt.Rectangle((px - w / 2, py - h / 2), w, h, fill=False, edgecolor="lime", linewidth=1.5))
            a.text(px, py - h / 2, f"C{pred_id:02d} {score:.2f}", color="yellow", fontsize=7)
        a.set_title(img_path.name); a.axis("off")
    plt.tight_layout(); plt.show()
