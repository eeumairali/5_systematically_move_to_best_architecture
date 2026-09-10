"""Gaussian heatmap targets + the joint detection/re-ID Dataset and collate function."""
import math

import torch
from PIL import Image
from torch.utils.data import Dataset

from n10_funs.config import FEAT_H, FEAT_W, IMG_H, IMG_W, STRIDE

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def gaussian_radius(box_h, box_w, min_overlap=0.7):
    """How wide a Gaussian bump around an object center still guarantees an IoU >= min_overlap
    with the true box, for *some* placement of a same-size predicted box.

    Standard CornerNet (Law & Deng 2018) radius formula, which FairMOT explicitly reuses for its
    heatmap target (Sec. 4.2.1 references CornerNet/CenterNet for the detection branch). It solves
    three quadratics for the three ways a predicted box of the same size can overlap the GT box
    (fully inside, fully outside, or straddling a corner) and takes the smallest resulting radius,
    i.e. the strictest constraint."""
    a1, b1, c1 = 1, (box_h + box_w), box_w * box_h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 - math.sqrt(max(b1 ** 2 - 4 * a1 * c1, 0))) / 2

    a2, b2, c2 = 4, 2 * (box_h + box_w), (1 - min_overlap) * box_w * box_h
    r2 = (b2 - math.sqrt(max(b2 ** 2 - 4 * a2 * c2, 0))) / 2

    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (box_h + box_w), (min_overlap - 1) * box_w * box_h
    r3 = (b3 + math.sqrt(max(b3 ** 2 - 4 * a3 * c3, 0))) / 2

    return max(0.0, min(r1, r2, r3))


def draw_gaussian(heatmap, cx, cy, radius):
    """Splat exp(-(dx^2+dy^2)/(2*sigma^2)) onto heatmap around (cx, cy), clipped to the image,
    keeping the maximum where two objects' Gaussians overlap (Eq. 1's M_xy is itself a sum, but
    taking the max is the standard, numerically nicer implementation used by CenterNet/FairMOT
    code releases and gives the same effective target after the >1 values are clamped)."""
    radius = max(0, int(radius))
    sigma = max(radius / 3.0, 0.5)
    h, w = heatmap.shape
    left, right = min(cx, radius), min(w - cx, radius + 1)
    top, bottom = min(cy, radius), min(h - cy, radius + 1)
    if left + right <= 0 or top + bottom <= 0:
        return
    ys = torch.arange(-top, bottom, dtype=torch.float32).view(-1, 1)
    xs = torch.arange(-left, right, dtype=torch.float32).view(1, -1)
    gauss = torch.exp(-(xs ** 2 + ys ** 2) / (2 * sigma ** 2))
    region = heatmap[cy - top:cy + bottom, cx - left:cx + right]
    torch.max(region, gauss, out=region)


def unnormalize(img):
    return (img * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1).permute(1, 2, 0).numpy()


class MmCowsJointDataset(Dataset):
    """Produces one training tensor bundle per frame: the resized image, plus the dense
    supervision maps for the classification head (heatmap) and, at each GT center, the box
    (offset, size) for the detection head and the identity for the re-ID head."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, boxes = self.samples[idx]
        img = Image.open(img_path).convert("RGB").resize((IMG_W, IMG_H))
        img = torch.from_numpy(
            __import__("numpy").array(img)
        ).permute(2, 0, 1).float() / 255.0
        # ImageNet normalization (matches n7/n8's convention in this repo)
        img = (img - IMAGENET_MEAN) / IMAGENET_STD

        heatmap = torch.zeros(FEAT_H, FEAT_W)   # classification target: 1 = cow center, Eq. 1
        max_boxes = len(boxes)
        offset_gt = torch.zeros(max_boxes, 2)   # detection target
        size_gt = torch.zeros(max_boxes, 2)     # detection target
        center_int = torch.zeros(max_boxes, 2, dtype=torch.long)  # (feat_x, feat_y) per box
        identity_gt = torch.zeros(max_boxes, dtype=torch.long)     # re-ID target
        mask = torch.zeros(max_boxes, dtype=torch.bool)

        for i, (cid, cx, cy, bw, bh) in enumerate(boxes):
            # normalized -> pixel -> stride-4 feature-map coordinates (paper Sec. 4.2, before Eq.1)
            px, py = cx * IMG_W, cy * IMG_H
            pw, ph = bw * IMG_W, bh * IMG_H
            fx, fy = px / STRIDE, py / STRIDE
            fx_int, fy_int = int(fx), int(fy)
            if not (0 <= fx_int < FEAT_W and 0 <= fy_int < FEAT_H):
                continue  # center fell outside the feature map after resize/rounding; drop it

            radius = gaussian_radius(ph / STRIDE, pw / STRIDE)
            draw_gaussian(heatmap, fx_int, fy_int, radius)

            offset_gt[i] = torch.tensor([fx - fx_int, fy - fy_int])       # sub-pixel offset, Eq.2
            size_gt[i] = torch.tensor([pw / STRIDE, ph / STRIDE])         # box size in feature units
            center_int[i] = torch.tensor([fx_int, fy_int])
            identity_gt[i] = cid - 1        # ids in files are 1..16 -> 0..15 for CrossEntropyLoss
            mask[i] = True

        return img, heatmap, offset_gt, size_gt, center_int, identity_gt, mask


def collate_joint(batch):
    """Frames have a variable number of boxes, so pad the per-box tensors to the batch max
    (same pattern n6/n7 use for variable-length YOLO targets) and carry a boolean mask."""
    imgs, heatmaps, offsets, sizes, centers, ids, masks = zip(*batch)
    max_n = max(m.numel() for m in masks)
    max_n = max(max_n, 1)

    def pad(tensors, fill_shape):
        out = torch.zeros(len(tensors), max_n, *fill_shape) if fill_shape else \
              torch.zeros(len(tensors), max_n, dtype=tensors[0].dtype)
        for i, t in enumerate(tensors):
            out[i, :t.shape[0]] = t
        return out

    return (
        torch.stack(imgs),
        torch.stack(heatmaps),
        pad(offsets, (2,)),
        pad(sizes, (2,)),
        pad(centers, (2,)).long(),
        pad(ids, ()).long(),
        pad(masks, ()).bool(),
    )
