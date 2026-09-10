"""Shared backbone + three named heads (classification, detection, re-ID)."""
import torch
from torch import nn
import torch.nn.functional as F

from n10_funs.config import DEVICE, IMG_H, IMG_W, N_CLASSES, N_IDENTITIES


def count_params(model):
    """Running tally for the series: n2 AlexNet=61M, n3 VGG16=138M, n4 ResNet-18=11M,
    n8 OSNet=2.2M. This network is a small custom encoder-decoder, not a paper baseline with a
    published parameter count to match -- so there is no target number here, just a sanity check
    that it is small enough to train in a notebook."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_shapes(model, x=None):
    """it x is not none, make zeros tensor of h, w then take out model output, print the name and shape of each output tensor"""
    x = x if x is not None else torch.zeros(1, 3, IMG_H, IMG_W).to(DEVICE)
    out = model(x)
    for name, t in out.items():
        print(f"{name:10s} {tuple(t.shape)}")


class ConvBNReLU(nn.Module):
    """conv -> BN -> ReLU. The one repeated building block of both the encoder and decoder."""

    def __init__(self, in_ch, out_ch, k=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ResBlock(nn.Module):
    """Standard pre-ResNet residual block (He et al. 2016), used to build the downsampling
    stages. Nothing FairMOT-specific here -- any competent backbone stage works; DLA-34 itself is
    built from very similar residual/aggregation blocks."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, out_ch, stride=stride)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)
        out = self.conv1(x)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class FuseUp(nn.Module):
    """One decoder step of the IDA-Up-style path: upsample the deeper (lower-resolution, more
    semantic) feature map to match the shallower one, concatenate, then fuse with a conv.
    SIMPLIFICATION vs paper: FairMOT's up-sampling convolutions are deformable; these are plain
    3x3 convs (see the architecture-overview markdown cell in the notebook for why)."""

    def __init__(self, deep_ch, shallow_ch, out_ch):
        super().__init__()
        self.reduce = ConvBNReLU(deep_ch, out_ch, k=1)
        self.fuse = ConvBNReLU(out_ch + shallow_ch, out_ch, k=3)

    def forward(self, deep, shallow):
        deep = self.reduce(deep)
        deep = F.interpolate(deep, size=shallow.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([deep, shallow], dim=1))


class JointBackbone(nn.Module):
    """Encoder (stride 4/8/16/32) + decoder fusing back down to stride 4. Output: a single
    (B, C, H/4, W/4) feature map shared by every head, exactly as in FairMOT Fig. 1."""

    def __init__(self, base_ch=32):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        # Stem: two stride-2 convs -> stride 4 directly (DLA-34 similarly reaches /4 immediately).
        self.stem = nn.Sequential(ConvBNReLU(3, c1, k=7, stride=2), ConvBNReLU(c1, c1, stride=2))
        self.stage8 = ResBlock(c1, c2, stride=2)     # /8
        self.stage16 = ResBlock(c2, c3, stride=2)    # /16
        self.stage32 = ResBlock(c3, c4, stride=2)    # /32

        self.up16 = FuseUp(c4, c3, c3)   # /32 -> fuse with /16
        self.up8 = FuseUp(c3, c2, c2)    # -> fuse with /8
        self.up4 = FuseUp(c2, c1, c1)    # -> fuse with /4 (stem output)

        self.out_channels = c1

    def forward(self, x):
        f4 = self.stem(x)
        f8 = self.stage8(f4)
        f16 = self.stage16(f8)
        f32 = self.stage32(f16)
        d16 = self.up16(f32, f16)
        d8 = self.up8(d16, f8)
        d4 = self.up4(d8, f4)
        return d4  # (B, out_channels, H/4, W/4)


def make_head(in_ch, mid_ch, out_ch):
    """Every head in FairMOT is '3x3 conv (256 ch) -> 1x1 conv -> target' (Sec. 4.2.2/4.3). We
    keep the same two-layer shape, just with a smaller mid channel count to match the smaller
    backbone."""
    return nn.Sequential(
        nn.Conv2d(in_ch, mid_ch, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(mid_ch, out_ch, 1))


class JointDetReIDModel(nn.Module):
    """Three named heads on one shared backbone, per the user's requirement:

      classification (N_CLASSES, H/4, W/4)  foreground/objectness score, Eq. 1
                                              (N_CLASSES=1 for mmCows: cow vs. background)
      detection: offset (2, H/4, W/4)        sub-pixel center correction, Eq. 2
      detection: size   (2, H/4, W/4)        box width/height in feature units, Eq. 2
      reid      (128, H/4, W/4)              identity embedding map, Sec. 4.3 (dim=128, the
                                              paper's own choice after their Sec. 5.3.4 ablation --
                                              Table 6 shows LOW-dimensional re-ID features are
                                              better for the joint task, not worse)

    The re-ID head outputs an embedding *map* rather than direct class scores, matching the
    paper's design: the actual identity classification happens by gathering the embedding at the
    object's center and passing it through one shared FC+softmax (identity_fc below), not a
    per-pixel classifier. This is what lets the embedding double as a tracking feature at
    inference time (Sec. 4.5.2) while still being trained with a plain classification loss."""

    def __init__(self, n_classes=N_CLASSES, n_identities=N_IDENTITIES, reid_dim=128):
        super().__init__()
        self.backbone = JointBackbone(base_ch=32)
        c = self.backbone.out_channels
        self.classification_head = make_head(c, 64, n_classes)
        self.offset_head = make_head(c, 64, 2)
        self.size_head = make_head(c, 64, 2)
        self.reid_head = make_head(c, 64, reid_dim)

        # Sec. 4.3.1: "extract the re-ID feature vector ... use a fully connected layer and a
        # softmax to map it to a class distribution".
        self.identity_fc = nn.Linear(reid_dim, n_identities)

        # Eq. 4/5: one learnable log-uncertainty per task. The paper groups heatmap+box under one
        # "detection" weight and gives re-ID its own (2 terms total, Eq. 5). Since the user wants
        # 3 *named* heads rather than 2, this unbundles that into 3 separate learnable weights
        # (classification, detection, re-ID) -- a reorganization of Eq. 5, not a literal copy.
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        feat = self.backbone(x)
        return {
            "classification": torch.sigmoid(self.classification_head(feat)),
            "offset": self.offset_head(feat),
            "size": self.size_head(feat),
            "reid_map": self.reid_head(feat),
        }

    def gather_embeddings(self, feat_map, centers, mask):
        """Pick out the embedding vector at each GT center. feat_map: (B, C, H, W).
        centers: (B, N, 2) feature-map (x, y) ints. mask: (B, N) bool, which slots are real boxes.
        Returns (n_valid, C) embeddings, matched 1:1 with the flattened valid rows of centers."""
        B, C, H, W = feat_map.shape
        flat = feat_map.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, H*W, C)
        idx = (centers[..., 1] * W + centers[..., 0]).clamp(0, H * W - 1)  # (B, N) -> flat index
        gathered = torch.gather(flat, 1, idx.unsqueeze(-1).expand(-1, -1, C))  # (B, N, C)
        return gathered[mask]
