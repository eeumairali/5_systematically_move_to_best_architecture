import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class BackBone:
    """backbone for JDE model"""
    def __init__(self):
        self.backbone = None
    def make_backbone(self):
        raise NotImplementedError("make_backbone method should be implemented in subclass")


class Architecture(BackBone):
    """main architecture for JDE model"""
    def __init__(self):
        super(Architecture, self).__init__()
        self.make_backbone()

    def make_neck(self):
        raise NotImplementedError("make_neck method should be implemented in subclass")
    def make_head_detection(self):
        raise NotImplementedError("make_head_detection method should be implemented in subclass")
    def make_head_embedding(self):
        raise NotImplementedError("make_head_embedding method should be implemented in subclass")


class JDEModel(Architecture, nn.Module):
    """
    JDE / FairMOT-style model: shared backbone + neck producing a single
    stride-4 feature map, with three CenterNet-style heads:
      - hm  : object center heatmap (num_classes)
      - wh  : box width/height regression (2)
      - reg : sub-pixel center offset (2)
    plus a ReID embedding head (id) used for joint detection + tracking.
    """

    def __init__(self, num_classes=1, reid_dim=128, head_conv=256):
        self.num_classes = num_classes
        self.reid_dim = reid_dim
        self.head_conv = head_conv
        nn.Module.__init__(self)
        Architecture.__init__(self)
        self.make_neck()
        self.make_head_detection()
        self.make_head_embedding()

    # ---------------- backbone ----------------
    def make_backbone(self):
        resnet = torchvision.models.resnet34(weights=None)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1   # stride 4,  64ch
        self.layer2 = resnet.layer2   # stride 8,  128ch
        self.layer3 = resnet.layer3   # stride 16, 256ch
        self.layer4 = resnet.layer4   # stride 32, 512ch
        self.backbone = [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]

    # ---------------- neck (FPN-style upsampling fusion down to stride 4) ----------------
    def make_neck(self):
        def upsample_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.lat4 = upsample_block(512, 256)
        self.lat3 = upsample_block(256, 256)
        self.lat2 = upsample_block(128, 256)
        self.lat1 = upsample_block(64, 256)

        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)

        self.neck_out_channels = 256

    def _fuse_neck(self, c1, c2, c3, c4):
        p4 = self.lat4(c4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p3 = self.smooth3(p3)
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        p2 = self.smooth2(p2)
        p1 = self.lat1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="nearest")
        p1 = self.smooth1(p1)
        return p1  # stride 4 feature map

    # ---------------- heads ----------------
    def _make_head(self, out_channels):
        head = nn.Sequential(
            nn.Conv2d(self.neck_out_channels, self.head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_conv, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )
        return head

    def make_head_detection(self):
        self.hm = self._make_head(self.num_classes)
        self.wh = self._make_head(2)
        self.reg = self._make_head(2)
        # heatmap bias init so training starts from low foreground probability
        self.hm[-1].bias.data.fill_(-2.19)

    def make_head_embedding(self):
        self.id_head = self._make_head(self.reid_dim)
        # learnable per-task uncertainty weights, as in FairMOT's loss balancing
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))

    # ---------------- forward / train / infer ----------------
    def forward(self, x):
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        feat = self._fuse_neck(c1, c2, c3, c4)

        out = {
            "hm": torch.sigmoid(self.hm(feat)),
            "wh": self.wh(feat),
            "reg": self.reg(feat),
            "id": self.id_head(feat),
        }
        return out

    def compute_loss(self, outputs, targets, det_loss_fn, id_loss_fn):
        """
        det_loss_fn(outputs, targets) -> scalar detection loss (focal + wh/reg L1)
        id_loss_fn(outputs, targets)  -> scalar ReID classification/embedding loss
        Combines them with FairMOT's uncertainty-based weighting.
        """
        det_loss = det_loss_fn(outputs, targets)
        id_loss = id_loss_fn(outputs, targets)
        loss = (
            torch.exp(-self.s_det) * det_loss
            + torch.exp(-self.s_id) * id_loss
            + (self.s_det + self.s_id)
        ) * 0.5
        return loss, {"det_loss": det_loss.detach(), "id_loss": id_loss.detach()}

    def backpropogation(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    @torch.no_grad()
    def inference(self, x):
        self.eval()
        return self.forward(x)


if __name__ == "__main__":
    model = JDEModel(num_classes=1, reid_dim=128, head_conv=256)
    print(model)
    x = torch.randn(1, 3, 512, 512)
    outputs = model(x)
    for k, v in outputs.items():
        print(k, v.shape) 