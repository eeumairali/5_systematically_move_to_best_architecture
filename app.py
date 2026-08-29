"""
Tiger Vision Lab -- run any trained checkpoint from runs/ on an uploaded photo.

Classification checkpoints (n1, n2) guess which of the 107 ATRW tiger identities
is in the photo. Detection checkpoints (n5, n6, n7) draw a box around the tiger.
Model architectures below mirror the corresponding notebook exactly, so the saved
state_dicts load without modification.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms

APP_DIR = Path(__file__).resolve().parent
RUNS_DIR = APP_DIR / "runs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

GRID_SIZES_3SCALE = [13, 26, 52]


# =============================================================================
# Architectures (one class per notebook, unchanged from training code)
# =============================================================================

class LeNet5(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5), nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5), nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 120, kernel_size=5), nn.Tanh(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(120, 84), nn.Tanh(),
            nn.Linear(84, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class AlexNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4), nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2), nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ConvBlockV1(nn.Module):
    """conv -> BN -> leaky ReLU(0.1) -> [optional 2x2 max-pool]."""

    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))


class FastYOLO(nn.Module):
    """YOLOv1 ("Fast YOLO"): single 7x7 grid, 2 boxes/cell, 1 class, sigmoid head."""

    def __init__(self, s=7, b=2, c=1):
        super().__init__()
        self.s, self.b, self.c = s, b, c
        self.features = nn.Sequential(
            ConvBlockV1(3, 16), ConvBlockV1(16, 32), ConvBlockV1(32, 64),
            ConvBlockV1(64, 128), ConvBlockV1(128, 256), ConvBlockV1(256, 512),
            ConvBlockV1(512, 1024, pool=False),
            ConvBlockV1(1024, 1024, pool=False),
            ConvBlockV1(1024, 1024, pool=False),
        )
        out_dim = s * s * (b * 5 + c)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * s * s, 1024), nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, out_dim),
        )

    def forward(self, x):
        x = self.head(self.features(x))
        x = torch.sigmoid(x)
        return x.view(-1, self.s, self.s, self.b * 5 + self.c)


class ConvBN(nn.Module):
    """Darknet building block: conv -> BN -> leaky ReLU(0.1)."""

    def __init__(self, in_ch, out_ch, k=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResUnit(nn.Module):
    """1x1 reduce -> 3x3 expand -> add input (Darknet-53 shortcut block)."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBN(channels, channels // 2, k=1)
        self.conv2 = ConvBN(channels // 2, channels, k=3)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


def res_stage(channels, n):
    return nn.Sequential(*[ResUnit(channels) for _ in range(n)])


def head_block(in_ch, mid_ch):
    return nn.Sequential(
        ConvBN(in_ch, mid_ch, k=1),
        ConvBN(mid_ch, mid_ch * 2, k=3),
        ConvBN(mid_ch * 2, mid_ch, k=1),
    )


class MiniDarknet53(nn.Module):
    """Darknet-53 stride/channel progression, fewer residual repeats per stage."""

    def __init__(self):
        super().__init__()
        self.stem = ConvBN(3, 32, k=3)
        self.down1 = ConvBN(32, 64, k=3, stride=2)
        self.stage1 = res_stage(64, 1)
        self.down2 = ConvBN(64, 128, k=3, stride=2)
        self.stage2 = res_stage(128, 1)
        self.down3 = ConvBN(128, 256, k=3, stride=2)
        self.stage3 = res_stage(256, 2)
        self.down4 = ConvBN(256, 512, k=3, stride=2)
        self.stage4 = res_stage(512, 2)
        self.down5 = ConvBN(512, 1024, k=3, stride=2)
        self.stage5 = res_stage(1024, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(self.down1(x))
        x = self.stage2(self.down2(x))
        route_52 = self.stage3(self.down3(x))
        route_26 = self.stage4(self.down4(route_52))
        route_13 = self.stage5(self.down5(route_26))
        return route_52, route_26, route_13


class YOLOv3Mini(nn.Module):
    """3 detection heads (13/26/52 grids), 3 anchors each, 1 class."""

    def __init__(self, a=3, c=1):
        super().__init__()
        self.a, self.c = a, c
        out_ch = a * (5 + c)
        self.backbone = MiniDarknet53()

        self.head13 = head_block(1024, 512)
        self.pred13 = nn.Sequential(ConvBN(512, 1024, k=3), nn.Conv2d(1024, out_ch, 1))

        self.reduce13 = ConvBN(512, 256, k=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.head26 = head_block(256 + 512, 256)
        self.pred26 = nn.Sequential(ConvBN(256, 512, k=3), nn.Conv2d(512, out_ch, 1))

        self.reduce26 = ConvBN(256, 128, k=1)
        self.head52 = head_block(128 + 256, 128)
        self.pred52 = nn.Sequential(ConvBN(128, 256, k=3), nn.Conv2d(256, out_ch, 1))

    def forward(self, x):
        route_52, route_26, route_13 = self.backbone(x)

        h13 = self.head13(route_13)
        out13 = self.pred13(h13)

        up13 = self.upsample(self.reduce13(h13))
        h26 = self.head26(torch.cat([up13, route_26], dim=1))
        out26 = self.pred26(h26)

        up26 = self.upsample(self.reduce26(h26))
        h52 = self.head52(torch.cat([up26, route_52], dim=1))
        out52 = self.pred52(h52)

        def reshape(t, g):
            n = t.shape[0]
            return t.permute(0, 2, 3, 1).reshape(n, g, g, self.a, 5 + self.c)

        return [reshape(out13, 13), reshape(out26, 26), reshape(out52, 52)]


class MiniBackboneNeck(nn.Module):
    """Shared backbone + FPN neck feeding the YOLO26-mini detection heads."""

    def __init__(self):
        super().__init__()
        self.stem = ConvBN(3, 32, k=3)
        self.down1 = ConvBN(32, 64, k=3, stride=2)
        self.stage1 = res_stage(64, 1)
        self.down2 = ConvBN(64, 128, k=3, stride=2)
        self.stage2 = res_stage(128, 1)
        self.down3 = ConvBN(128, 256, k=3, stride=2)
        self.stage3 = res_stage(256, 2)
        self.down4 = ConvBN(256, 512, k=3, stride=2)
        self.stage4 = res_stage(512, 2)
        self.down5 = ConvBN(512, 1024, k=3, stride=2)
        self.stage5 = res_stage(1024, 1)

        self.head13 = nn.Sequential(ConvBN(1024, 512, k=1), ConvBN(512, 1024, k=3), ConvBN(1024, 512, k=1))
        self.reduce13 = ConvBN(512, 256, k=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.head26 = nn.Sequential(ConvBN(256 + 512, 256, k=1), ConvBN(256, 512, k=3), ConvBN(512, 256, k=1))
        self.reduce26 = ConvBN(256, 128, k=1)
        self.head52 = nn.Sequential(ConvBN(128 + 256, 128, k=1), ConvBN(128, 256, k=3), ConvBN(256, 128, k=1))

        self.out_channels = [512, 256, 128]

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(self.down1(x))
        x = self.stage2(self.down2(x))
        route_52 = self.stage3(self.down3(x))
        route_26 = self.stage4(self.down4(route_52))
        route_13 = self.stage5(self.down5(route_26))

        f13 = self.head13(route_13)
        up13 = self.upsample(self.reduce13(f13))
        f26 = self.head26(torch.cat([up13, route_26], dim=1))
        up26 = self.upsample(self.reduce26(f26))
        f52 = self.head52(torch.cat([up26, route_52], dim=1))
        return [f13, f26, f52]


def dual_head(in_ch, c):
    return nn.Sequential(ConvBN(in_ch, in_ch // 2, k=3), nn.Conv2d(in_ch // 2, 4 + c, 1))


class MiniYOLO26(nn.Module):
    """Anchor-free, dual one-to-many / one-to-one heads. o2o is NMS-free at inference."""

    def __init__(self, c=1):
        super().__init__()
        self.c = c
        self.backbone_neck = MiniBackboneNeck()
        ch = self.backbone_neck.out_channels
        self.o2m_heads = nn.ModuleList([dual_head(ci, c) for ci in ch])
        self.o2o_heads = nn.ModuleList([dual_head(ci, c) for ci in ch])

    def forward(self, x):
        feats = self.backbone_neck(x)

        def run(heads):
            outs = []
            for f, head, g in zip(feats, heads, GRID_SIZES_3SCALE):
                o = head(f)
                n = o.shape[0]
                outs.append(o.permute(0, 2, 3, 1).reshape(n, g * g, 4 + self.c))
            return torch.cat(outs, dim=1)

        return run(self.o2m_heads), run(self.o2o_heads)


# =============================================================================
# Registry: which checkpoint uses which architecture, input size, preprocessing
# =============================================================================

def _n_classes(state, key):
    return state[key].shape[0]


RUN_SPECS = {
    "n1_LeNet5_atrw": dict(
        label="LeNet-5 -- tiger identity classifier",
        task="classification",
        img_size=32,
        norm_mean=[0.5, 0.5, 0.5], norm_std=[0.5, 0.5, 0.5],
        build=lambda state: LeNet5(n_classes=_n_classes(state, "classifier.3.weight")),
    ),
    "n2_AlexNet": dict(
        label="AlexNet -- tiger identity classifier",
        task="classification",
        img_size=227,
        norm_mean=IMAGENET_MEAN, norm_std=IMAGENET_STD,
        build=lambda state: AlexNet(n_classes=_n_classes(state, "classifier.7.weight")),
    ),
    "n5_yolov1": dict(
        label="YOLOv1 (Fast YOLO) -- tiger detector",
        task="detection",
        img_size=448,
        norm_mean=IMAGENET_MEAN, norm_std=IMAGENET_STD,
        classes=["Tiger"],
        build=lambda state: FastYOLO(s=7, b=2, c=1),
    ),
    "n6_yolov3": dict(
        label="YOLOv3-mini -- tiger detector, multi-scale",
        task="detection",
        img_size=416,
        norm_mean=IMAGENET_MEAN, norm_std=IMAGENET_STD,
        classes=["Tiger"],
        build=lambda state: YOLOv3Mini(a=3, c=1),
    ),
    "n7_yolo26": dict(
        label="YOLO26-mini -- tiger detector, anchor-free / NMS-free",
        task="detection",
        img_size=416,
        norm_mean=IMAGENET_MEAN, norm_std=IMAGENET_STD,
        classes=["Tiger"],
        build=lambda state: MiniYOLO26(c=1),
    ),
}


def discover_runs():
    """Only list checkpoints that actually exist on disk yet."""
    available = {}
    if not RUNS_DIR.exists():
        return available
    for run_dir in sorted(RUNS_DIR.iterdir()):
        spec = RUN_SPECS.get(run_dir.name)
        ckpt = run_dir / "model.pt"
        if spec and ckpt.exists():
            metrics_path = run_dir / "metrics.json"
            metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
            available[run_dir.name] = {**spec, "checkpoint": ckpt, "metrics": metrics}
    return available


@st.cache_resource(show_spinner="Loading model...")
def load_model(run_name):
    spec = RUN_SPECS[run_name]
    checkpoint = torch.load(RUNS_DIR / run_name / "model.pt", map_location=DEVICE, weights_only=False)
    state = checkpoint["model_state"]
    model = spec["build"](state)
    model.load_state_dict(state)
    model.eval().to(DEVICE)
    extra = {}
    if "anchors" in checkpoint:
        extra["anchors"] = torch.tensor(np.asarray(checkpoint["anchors"]), dtype=torch.float32)
    return model, extra


# =============================================================================
# Pre/post-processing shared by every model
# =============================================================================

def preprocess(image, img_size, mean, std):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return tfm(image.convert("RGB")).unsqueeze(0)


def box_iou(box1, boxes2):
    x1 = torch.maximum(box1[:, 0], boxes2[:, 0])
    y1 = torch.maximum(box1[:, 1], boxes2[:, 1])
    x2 = torch.minimum(box1[:, 2], boxes2[:, 2])
    y2 = torch.minimum(box1[:, 3], boxes2[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - inter
    return inter / union.clamp(min=1e-9)


def non_max_suppression(boxes, scores, iou_thresh=0.5):
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        ious = box_iou(boxes[i].unsqueeze(0), boxes[rest])
        order = rest[ious <= iou_thresh]
    return keep


@torch.no_grad()
def run_classification(model, x):
    logits = model(x.to(DEVICE))
    return torch.softmax(logits, dim=1)[0].cpu()


@torch.no_grad()
def decode_yolov1(model, x, img_size, conf_thresh):
    preds = model(x.to(DEVICE))[0].cpu()  # (S,S,B*5+C)
    s, b = model.s, model.b
    cell = img_size / s
    rows = torch.arange(s).view(s, 1).expand(s, s)
    cols = torch.arange(s).view(1, s).expand(s, s)
    pred_boxes = preds[..., :b * 5].view(s, s, b, 5)
    class_prob = preds[..., b * 5:]

    boxes_all, scores_all = [], []
    for k in range(b):
        px, py, pw, ph, conf = pred_boxes[..., k, :].unbind(-1)
        score = conf.clamp(min=0) * class_prob[..., 0].clamp(min=0)
        keep = score > conf_thresh
        if not keep.any():
            continue
        xc = (cols[keep] + px[keep]) * cell
        yc = (rows[keep] + py[keep]) * cell
        bw = pw[keep].clamp(min=0) * img_size
        bh = ph[keep].clamp(min=0) * img_size
        corners = torch.stack([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2], dim=-1)
        boxes_all.append(corners)
        scores_all.append(score[keep])

    if not boxes_all:
        return torch.empty(0, 4), torch.empty(0)
    return torch.cat(boxes_all), torch.cat(scores_all)


@torch.no_grad()
def decode_yolov3(model, x, anchors, img_size, conf_thresh):
    preds = model(x.to(DEVICE))  # list of 3, each (1,g,g,A,5+C)
    a = model.a
    boxes_all, scores_all = [], []
    for scale_idx, raw in enumerate(preds):
        raw = raw[0].cpu()  # (g,g,A,5+C)
        g = raw.shape[0]
        stride = img_size / g
        rows = torch.arange(g).view(g, 1, 1).expand(g, g, a)
        cols = torch.arange(g).view(1, g, 1).expand(g, g, a)
        anc = anchors[scale_idx * a:(scale_idx + 1) * a]

        tx, ty, tw, th, tobj = raw[..., 0], raw[..., 1], raw[..., 2], raw[..., 3], raw[..., 4]
        class_logit = raw[..., 5:]

        bx = (torch.sigmoid(tx) + cols) * stride
        by = (torch.sigmoid(ty) + rows) * stride
        bw = anc[:, 0] * torch.exp(tw.clamp(max=6))
        bh = anc[:, 1] * torch.exp(th.clamp(max=6))
        boxes = torch.stack([bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2], dim=-1)
        score = torch.sigmoid(tobj) * torch.sigmoid(class_logit[..., 0])

        keep = score > conf_thresh
        if keep.any():
            boxes_all.append(boxes[keep])
            scores_all.append(score[keep])

    if not boxes_all:
        return torch.empty(0, 4), torch.empty(0)
    return torch.cat(boxes_all), torch.cat(scores_all)


@torch.no_grad()
def decode_yolo26(model, x, img_size, conf_thresh, top_k=50):
    _, o2o_raw = model(x.to(DEVICE))  # (1, N_ANCHORS, 4+C), NMS-free head
    raw = o2o_raw[0].cpu()

    strides = [img_size // g for g in GRID_SIZES_3SCALE]
    pts = []
    for g, stride in zip(GRID_SIZES_3SCALE, strides):
        ys, xs = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
        cx = (xs.float() + 0.5) * stride
        cy = (ys.float() + 0.5) * stride
        pts.append(torch.stack([cx, cy], dim=-1).reshape(-1, 2))
    all_points = torch.cat(pts, dim=0)

    l, t, r, b = raw[..., 0], raw[..., 1], raw[..., 2], raw[..., 3]
    l, t, r, b = F.softplus(l), F.softplus(t), F.softplus(r), F.softplus(b)
    cx, cy = all_points[:, 0], all_points[:, 1]
    boxes = torch.stack([cx - l, cy - t, cx + r, cy + b], dim=-1)
    scores = torch.sigmoid(raw[..., 4])

    keep = scores > conf_thresh
    boxes, scores = boxes[keep], scores[keep]
    if scores.numel() > top_k:
        top = scores.topk(top_k).indices
        boxes, scores = boxes[top], scores[top]
    return boxes, scores


def draw_boxes(image, boxes, scores, label, img_size):
    scale_x = image.width / img_size
    scale_y = image.height / img_size
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for box, score in zip(boxes, scores):
        x0, y0, x1, y1 = box.tolist()
        x0, y0, x1, y1 = x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y
        draw.rectangle([x0, y0, x1, y1], outline="#00e08a", width=3)
        tag = f"{label} {score:.2f}"
        text_y = y0 - 16 if y0 - 16 > 0 else y0 + 2
        draw.rectangle([x0, text_y, x0 + 8 * len(tag), text_y + 14], fill="#00e08a")
        draw.text((x0 + 2, text_y), tag, fill="#00251a")
    return annotated


# =============================================================================
# UI
# =============================================================================

st.set_page_config(page_title="Tiger Vision Lab", page_icon="🐯", layout="wide")

st.title("🐯 Tiger Vision Lab")
st.caption(
    "Pick a trained checkpoint from `runs/`, upload a tiger photo, and run it. "
    "Classifiers (LeNet-5, AlexNet) guess the tiger's identity; detectors (YOLOv1/v3/26) draw a box."
)

runs = discover_runs()

if not runs:
    st.error("No trained checkpoints found under `runs/`. Train a model in one of the notebooks first.")
    st.stop()

with st.sidebar:
    st.header("Model")
    run_name = st.selectbox(
        "Checkpoint",
        options=list(runs.keys()),
        format_func=lambda k: runs[k]["label"],
    )
    spec = runs[run_name]
    metrics = spec["metrics"]

    if spec["task"] == "classification" and "held_out_accuracy" in metrics:
        st.metric("Held-out accuracy", f"{metrics['held_out_accuracy']:.1f}%")
    elif spec["task"] == "detection" and "held_out_ap50" in metrics:
        st.metric("Held-out AP@0.5", f"{metrics['held_out_ap50']:.3f}")

    st.caption(f"Task: {spec['task']} · input {spec['img_size']}x{spec['img_size']}")

    conf_thresh, iou_thresh = 0.2, 0.5
    if spec["task"] == "detection":
        st.header("Detection settings")
        conf_thresh = st.slider("Confidence threshold", 0.05, 0.95, 0.20, 0.05)
        if run_name == "n7_yolo26":
            st.caption("YOLO26's o2o head is trained to be NMS-free -- no suppression is applied.")
        else:
            iou_thresh = st.slider("NMS IoU threshold", 0.05, 0.95, 0.50, 0.05)

uploaded = st.file_uploader("Upload or drag & drop a tiger photo", type=["jpg", "jpeg", "png", "bmp"])

col1, col2 = st.columns(2)

image = None
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    with col1:
        st.image(image, caption="Input image", use_container_width=True)

run_clicked = st.button("Run inference", type="primary", disabled=image is None)

if run_clicked and image is not None:
    model, extra = load_model(run_name)
    x = preprocess(image, spec["img_size"], spec["norm_mean"], spec["norm_std"])

    if spec["task"] == "classification":
        probs = run_classification(model, x)
        k = min(5, probs.numel())
        top = torch.topk(probs, k=k)
        with col2:
            st.subheader("Prediction")
            best_idx = int(top.indices[0])
            best_conf = float(top.values[0]) * 100
            st.success(f"Predicted class index **#{best_idx}** -- {best_conf:.1f}% confidence")
            st.caption(
                f"Indices are the {probs.numel()} ATRW tiger identities in class-index order "
                "(not the original dataset tiger IDs)."
            )
            chart_df = pd.DataFrame(
                {"probability": [float(v) for v in top.values]},
                index=[f"class #{int(i)}" for i in top.indices],
            )
            st.bar_chart(chart_df)
    else:
        if run_name == "n5_yolov1":
            boxes, scores = decode_yolov1(model, x, spec["img_size"], conf_thresh)
            keep = non_max_suppression(boxes, scores, iou_thresh) if boxes.numel() else []
            boxes, scores = boxes[keep], scores[keep]
        elif run_name == "n6_yolov3":
            boxes, scores = decode_yolov3(model, x, extra["anchors"], spec["img_size"], conf_thresh)
            keep = non_max_suppression(boxes, scores, iou_thresh) if boxes.numel() else []
            boxes, scores = boxes[keep], scores[keep]
        else:  # n7_yolo26
            boxes, scores = decode_yolo26(model, x, spec["img_size"], conf_thresh)

        with col2:
            st.subheader(f"Detections ({len(scores)})")
            if len(scores) == 0:
                st.info("No tigers detected above the confidence threshold. Try lowering it in the sidebar.")
            else:
                label = spec["classes"][0]
                annotated = draw_boxes(image, boxes, scores, label, spec["img_size"])
                st.image(annotated, caption="Detections", use_container_width=True)
                df = pd.DataFrame({
                    "label": [label] * len(scores),
                    "confidence": [round(float(s), 3) for s in scores],
                    "x0": [round(float(b[0]), 1) for b in boxes],
                    "y0": [round(float(b[1]), 1) for b in boxes],
                    "x1": [round(float(b[2]), 1) for b in boxes],
                    "y1": [round(float(b[3]), 1) for b in boxes],
                }).sort_values("confidence", ascending=False).reset_index(drop=True)
                st.dataframe(df, use_container_width=True)
