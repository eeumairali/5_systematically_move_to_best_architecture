"""Turning dense head outputs back into boxes, and box-IoU for evaluation."""
import torch
import torch.nn.functional as F

from n10_funs.config import STRIDE


@torch.no_grad()
def decode_detections(class_score, offset, size, k=40, score_thresh=0.3):
    """Turn the network's dense classification/offset/size maps back into a list of boxes,
    following FairMOT Sec. 4.5.1: 3x3 max-pool NMS on the classification (heatmap) score (keep a
    pixel only if it is its own 3x3 neighborhood's maximum), take the top-k surviving peaks, then
    read off offset + size there. class_score: (H, W) sigmoid scores in [0,1], already reduced to
    a single foreground channel (N_CLASSES=1 for mmCows). offset, size: (2, H, W). Returns a list
    of (score, px, py, pw, ph, fx, fy) in *pixel* coordinates (already multiplied by STRIDE)."""
    H, W = class_score.shape
    pooled = F.max_pool2d(class_score.unsqueeze(0).unsqueeze(0), 3, stride=1, padding=1)[0, 0]
    peaks = (pooled == class_score) & (class_score > score_thresh)
    ys, xs = torch.where(peaks)
    scores = class_score[ys, xs]
    if len(scores) == 0:
        return []
    topk = torch.topk(scores, min(k, len(scores))).indices
    results = []
    for i in topk:
        fx, fy = xs[i].item(), ys[i].item()
        ox, oy = offset[0, fy, fx].item(), offset[1, fy, fx].item()
        w, h = size[0, fy, fx].item(), size[1, fy, fx].item()
        px, py = (fx + ox) * STRIDE, (fy + oy) * STRIDE
        results.append((scores[i].item(), px, py, w * STRIDE, h * STRIDE, fx, fy))
    return results


def box_iou(box_a, box_b):
    """IoU of two (cx, cy, w, h) boxes in the same units. Used only for evaluation, not training
    (FairMOT/CenterNet never regress IoU directly -- detection is trained via the classification
    heatmap + L1 losses)."""
    ax1, ay1, ax2, ay2 = box_a[0] - box_a[2] / 2, box_a[1] - box_a[3] / 2, box_a[0] + box_a[2] / 2, box_a[1] + box_a[3] / 2
    bx1, by1, bx2, by2 = box_b[0] - box_b[2] / 2, box_b[1] - box_b[3] / 2, box_b[0] + box_b[2] / 2, box_b[1] + box_b[3] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)
