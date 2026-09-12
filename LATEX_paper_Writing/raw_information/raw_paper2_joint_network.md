# Paper 2 — Joint Detection + Re-ID Network (JDE): Architecture Raw Notes

This document is raw source material for the **architecture / methodology**
section of the paper describing the *joint* (single-network, multi-head)
detection + identity model — as opposed to the *sequential* two-stage
detector-then-ReID pipeline (`sequence.py`), which is documented separately.
It corresponds to the implementation in `jde.py` (model class `JDEModel`),
which is itself a slimmed-down re-implementation of `n10_funs/model.py`'s
`JointDetReIDModel` — the two share the same backbone and the same design
lineage from **FairMOT** (Zhang et al., 2021, *"FairMOT: On the Fairness of
Detection and Re-Identification in Multiple Object Tracking"*) and, one
level deeper, **CenterNet** (Zhou et al., 2019, *"Objects as Points"*) for
the anchor-free heatmap detection formulation.

**Update policy**: keep this in sync with `jde.py` / `n10_funs/model.py` —
if the backbone, heads, or loss weighting change, update the relevant
section below and add a Change Log entry.

---

## 1. Proposed Architecture

### 1.1 One-sentence summary
A single shared convolutional encoder-decoder backbone (`JointBackbone`)
produces one stride-4 feature map from the input image; four lightweight
task-specific heads read off that same feature map to jointly predict
**object center heatmaps**, **sub-pixel center offsets**, **box sizes**, and
**per-object identity embeddings** — trained end-to-end with a single
homoscedastic-uncertainty-weighted multi-task loss.

### 1.2 Why "joint" (as opposed to sequential)
The defining architectural claim (matching FairMOT's own motivation) is that
detection and re-identification **share** the backbone's features instead of
running as two independent networks (detector → crop → separate ReID CNN,
as in `sequence.py`). The hypothesis under test in this "systematically move
to best architecture" project is that sharing representations is more
parameter- and compute-efficient, and avoids compounding errors from a hard
detector→ReID hand-off — at the cost of a trickier joint optimization
(competing gradients from very different tasks pulling on the same
features, which is exactly what the uncertainty-weighted loss in §3.5 is
there to manage).

### 1.3 Full data flow (as implemented)

```
Input image                              (B, 3, 400, 640)   [H, W = 400, 640]
        │
        ▼
┌───────────────────────────┐
│   JointBackbone (shared)  │
│  stem → stage8 → stage16  │            output: (B, 32, 100, 160)
│  → stage32 → up16 → up8   │            i.e. stride 4 relative to input
│  → up4                    │
└───────────────────────────┘
        │  (shared stride-4 feature map, 32 channels)
        ├────────────┬────────────┬────────────┐
        ▼            ▼            ▼            ▼
   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
   │ hm_head │  │ wh_head │  │reg_head │  │ id_head │
   │ 3x3→1x1 │  │ 3x3→1x1 │  │ 3x3→1x1 │  │ 3x3→1x1 │
   │ 32→64→1 │  │32→64→2  │  │32→64→2  │  │32→64→128│
   └─────────┘  └─────────┘  └─────────┘  └─────────┘
        │            │            │            │
     sigmoid       (raw)        (raw)      gather @ GT
        │            │            │        center (train)
        ▼            ▼            ▼            ▼
   heatmap        box w,h      sub-pixel   128-d embedding
  (B,1,100,160)  (B,2,100,160) offset      per object
                               (B,2,100,160)     │
                                                  ▼
                                       identity_classifier
                                         Linear(128→30)
                                                  │
                                                  ▼
                                          identity logits
```

### 1.4 Backbone — `JointBackbone` — layer-by-layer

`base_ch = 32` → channel plan `c1=32, c2=64, c3=128, c4=256`.

| Stage | Op | In→Out ch | Stride (cumulative) | Spatial size (640×400 input) |
|---|---|---|---|---|
| Stem | `ConvBNReLU(3→32, k=7, s=2)` → `ConvBNReLU(32→32, k=3, s=2)` | 3→32 | /4 | 160×100 |
| stage8 | `ResBlock(32→64, s=2)` | 32→64 | /8 | 80×50 |
| stage16 | `ResBlock(64→128, s=2)` | 64→128 | /16 | 40×25 |
| stage32 | `ResBlock(128→256, s=2)` | 128→256 | /32 | 20×13 (rounded) |
| up16 | `FuseUp(256, 128, 128)`: reduce f32 128→128 (1×1), bilinear-upsample to f16's size, concat (256ch), fuse 3×3 → 128 | 256+128→128 | back to /16 | 40×25 |
| up8 | `FuseUp(128, 64, 64)` | 128+64→64 | back to /8 | 80×50 |
| up4 | `FuseUp(64, 32, 32)` | 64+32→32 | back to /4 | 160×100 |

Output: `d4`, shape `(B, 32, 100, 160)` — this is `out_channels = c1 = 32`,
the single tensor every head reads from.

`ResBlock` = standard pre-activation-style residual block (He et al., 2016):
`conv-BN-ReLU → conv-BN`, added to a (possibly 1×1-conv-projected) shortcut,
then ReLU. `FuseUp` = one IDA-Up-style decoder step (Yu et al., 2018, *Deep
Layer Aggregation*, as adopted by CenterNet/FairMOT's DLA-34 decoder):
1×1-reduce the deeper/lower-resolution map, bilinear-upsample it to match
the shallower skip connection, concatenate, then fuse with a 3×3 conv.

**Deliberate simplification vs. the FairMOT paper**: FairMOT's decoder uses
*deformable* convolutions in its up-sampling path (on a full DLA-34 trunk);
this implementation uses plain 3×3 convolutions and a much shallower/narrower
trunk (`base_ch=32` vs. DLA-34's much larger channel counts). This is a
capacity/compute trade-off appropriate for a small (1500-image, 30-identity)
dataset — see §4.1 for the explicit backbone justification.

### 1.5 Heads — `_head(in, hidden, out)` = `Conv3x3(in→hidden) → ReLU → Conv1x1(hidden→out)`

| Head | Output channels | Activation | Meaning |
|---|---|---|---|
| `hm_head` | 1 | sigmoid | per-pixel object-center probability (class-agnostic — one heatmap, not one per species class) |
| `wh_head` | 2 | linear | box (width, height) in **feature-map units** at that pixel |
| `reg_head` | 2 | linear | sub-pixel (Δx, Δy) correction between the true center and its rounded-down feature-grid cell |
| `id_head` | 128 | linear | dense re-ID embedding map — one 128-d vector per feature-map pixel |

Every head shares the exact same two-layer shape (`3×3 conv → ReLU → 1×1
conv`), mid-channel width 64 — a deliberately uniform, minimal head design
so head capacity differences don't confound the comparison between tasks.

`hm_head`'s final-layer bias is initialized to **−2.19** (`log(0.01/0.99)`,
the standard CenterNet/RetinaNet focal-loss trick) so the network starts by
predicting a low foreground probability (~1%) everywhere, matching the fact
that the vast majority of pixels are background — this avoids the enormous
initial loss and gradient instability that would come from starting near
p=0.5 with a heavily imbalanced pixel-classification target.

### 1.6 From embedding map to identity — `JDEModel.gather` + `identity_classifier`

The `id_head` produces a *dense* embedding at every pixel, but identity
*labels* only exist at the (small number of) true object centers per image.
`JDEModel.gather(feature_map, centers)` flattens the map to `(B, H·W, C)`
and indexes out exactly the embedding vectors at the ground-truth center
locations (`torch.gather`), giving `(B, N, 128)`. Those are the only
embeddings that ever see a gradient from the identity loss. A single shared
`nn.Linear(128, 30)` (`identity_classifier`) then maps each gathered
embedding to identity logits over the 30 known cow/buffalo individuals.

This indirection (embedding map → gather-at-center → shared FC classifier)
is the architectural trick that lets the *same* embedding double as (a) a
trainable-by-cross-entropy identity classifier output during training and
(b) a plain feature vector usable for embedding-distance-based re-ID /
tracking at inference time, without needing two different heads for those
two uses.

### 1.7 Multi-task uncertainty weighting — `log_vars`

`JDEModel.log_vars` is a free `nn.Parameter(torch.zeros(3))` — one learnable
scalar per task group (heatmap, detection, identity). See §3.5 for the
exact loss formula. This is what actually balances the three very
differently-scaled losses (a bounded [0,1] pixel probability loss, an
unbounded L1 regression loss, and a 30-way cross-entropy) without manual
loss-weight tuning.

---

## 2. Figure-Making Notes (for whoever draws the architecture diagram)

Target: one FairMOT-Fig.-1-style diagram, roughly this layout, left to right:

1. **Leftmost box**: input image icon, labelled `640×400×3`.
2. **Encoder row** (going right, each box smaller/darker to show downsampling):
   `Stem (/4)` → `Stage8 (/8)` → `Stage16 (/16)` → `Stage32 (/32)`. Draw
   these as a shrinking staircase of feature-map "slabs" (a classic
   CNN-diagram convention: a thin 3D box whose footprint shrinks and whose
   thickness/channel-count grows at each stage — 32 → 64 → 128 → 256).
3. **Decoder row**, drawn as a mirrored staircase going back *up and left*
   underneath or above the encoder row: `up16` (/32→/16) → `up8` (/16→/8)
   → `up4` (/8→/4). At each decoder box, draw a **dashed skip-connection
   arrow** coming from the *matching* encoder stage (stage16→up16,
   stage8→up8, stem→up4) meeting an arrow from the previous (deeper)
   decoder stage — label the merge point "concat".
4. Encoder and decoder staircases together should visually read as a
   **U-shape** (classic encoder-decoder / U-Net-style silhouette), consistent
   with how FairMOT's own Fig. 1 draws the DLA-34 encoder-decoder.
5. **After the U**, one single box labelled "shared feature map,
   32×100×160" — this is the fan-out point.
6. From that box, **four parallel arrows** fan out to four small head boxes
   drawn side by side, each internally labelled `3×3 conv → 1×1 conv`:
   - "Heatmap head" → output labelled `1×100×160`, tag it "sigmoid".
   - "Offset head" → `2×100×160`.
   - "Size head" → `2×100×160`.
   - "Re-ID head" → `128×100×160`.
   Color-suggestion: give the first three heads one color family (e.g.
   blues, "detection branch") and the Re-ID head a distinct color (e.g.
   orange, "identity branch") — this is the single most important visual
   cue since it communicates the paper's core "shared trunk, split
   objective" claim at a glance.
7. Below the heatmap/offset/size heads, draw a small "Detection decode"
   box (dashed border, since `jde.py` doesn't implement this step —
   mark it optional / used at inference in the sequential baseline) merging
   into "predicted boxes".
8. Below the Re-ID head, draw: `Re-ID head output` → small box "Gather at
   GT center (training only)" → `128-d embedding` → `Linear(128→30)` →
   "Identity logits".
9. **Bottom band**: all four losses (heatmap focal loss, offset L1, size
   L1, identity cross-entropy) feed into one "Uncertainty-weighted
   multi-task loss" box that also takes three small labelled inputs
   `s_hm, s_det, s_id` (the learnable `log_vars`), producing `L_total`.
10. Caption should explicitly note: "single backbone, four heads, one loss"
    to contrast against the two-network sequential-pipeline figure.

---

## 3. Mathematical Formulation

Let the input image be $I \in \mathbb{R}^{3 \times H \times W}$ with
$H=400, W=640$, and let $R=4$ be the network stride, so the shared feature
map is $F = \text{Backbone}(I) \in \mathbb{R}^{C \times \hat H \times \hat W}$
with $\hat H = H/R = 100,\ \hat W = W/R = 160,\ C=32$.

### 3.1 Ground-truth heatmap construction
For each labelled box $k$ with normalized center $(c_x^k, c_y^k)$ and
normalized size $(w^k, h^k)$, the feature-grid center is
$$\tilde p^k = \left(\frac{c_x^k \cdot W}{R}, \frac{c_y^k \cdot H}{R}\right), \qquad p^k = \lfloor \tilde p^k \rfloor$$
A 2-D Gaussian is splatted onto the ground-truth heatmap $Y \in [0,1]^{\hat H \times \hat W}$ around $p^k$:
$$Y_{xy} = \exp\!\left(-\frac{(x-p^k_x)^2 + (y-p^k_y)^2}{2\sigma_k^2}\right), \qquad \sigma_k = \max(r_k/3,\ 0.5)$$
overlapping Gaussians for different objects are combined with an element-wise max.

The radius $r_k$ is the CornerNet/CenterNet radius (Law & Deng, 2018) — the
largest radius such that a box shifted by up to $r_k$ still has IoU
$\geq \text{overlap}$ (0.7 here) with the original box. It is the minimum of
three roots, one per corner-overlap case (both corners shrink, both grow,
one of each):
$$
\begin{aligned}
r_1 &= \frac{b_1 - \sqrt{b_1^2 - 4a_1c_1}}{2a_1}, & a_1&=1,\ b_1 = w{+}h,\ c_1 = \frac{wh(1-\text{ov})}{1+\text{ov}} \\
r_2 &= \frac{b_2 - \sqrt{b_2^2 - 4a_2c_2}}{2a_2}, & a_2&=4,\ b_2 = 2(w{+}h),\ c_2 = (1-\text{ov})\,wh \\
r_3 &= \frac{b_3 + \sqrt{b_3^2 - 4a_3c_3}}{2a_3}, & a_3&=4\,\text{ov},\ b_3 = -2\,\text{ov}(w{+}h),\ c_3 = (\text{ov}-1)\,wh \\
r_k &= \max(0,\ \min(r_1, r_2, r_3))
\end{aligned}
$$
with $(w,h)$ the box size in feature-map units and $\text{ov}=0.7$.

### 3.2 Heatmap loss (penalty-reduced pixel-wise focal loss, CenterNet Eq. (1))
Let $\hat Y = \sigma(\text{hm\_head}(F)) \in (0,1)^{\hat H \times \hat W}$ be the predicted heatmap.
$$
L_{hm} = \frac{-1}{N}\sum_{xy}
\begin{cases}
(1-\hat Y_{xy})^{\alpha}\,\log(\hat Y_{xy}) & \text{if } Y_{xy}=1 \\
(1-Y_{xy})^{\beta}\,\hat Y_{xy}^{\alpha}\,\log(1-\hat Y_{xy}) & \text{otherwise}
\end{cases}
$$
with $\alpha=2,\ \beta=4$ (as in CornerNet/CenterNet/FairMOT), and $N$ = number of positive (object-center) locations, clamped to $\geq 1$ so an empty image never divides by zero.

### 3.3 Box regression losses
At each positive location $p^k$, gather the predicted offset $\widehat{\text{reg}}_{p^k}$ and size $\widehat{\text{wh}}_{p^k}$ (bilinear-free — direct index gather, since targets are defined exactly on the integer feature grid), and regress against:
$$\text{reg}^k = \tilde p^k - p^k \in [0,1)^2, \qquad \text{wh}^k = \left(\frac{w^k W}{R}, \frac{h^k H}{R}\right)$$
$$L_{off} = \frac{1}{N}\sum_k \left\| \widehat{\text{reg}}_{p^k} - \text{reg}^k \right\|_1, \qquad L_{size} = \frac{1}{N}\sum_k \left\| \widehat{\text{wh}}_{p^k} - \text{wh}^k \right\|_1$$
$$L_{det} = L_{off} + \lambda_{size}\,L_{size}, \qquad \lambda_{size}=0.1$$

### 3.4 Identity loss
Gather the 128-d embedding at each positive location, $e^k = \text{id\_head}(F)_{p^k}$, and classify:
$$\hat y^k = \text{softmax}\!\big(W_{id}\,e^k + b_{id}\big) \in \mathbb{R}^{30}, \qquad L_{id} = \frac{1}{N}\sum_k \text{CrossEntropy}(\hat y^k,\ y^k)$$
where $y^k \in \{0,\dots,29\}$ is the contiguous identity index (see §4.4).

### 3.5 Multi-task uncertainty-weighted total loss
Following Kendall, Gal & Cipolla (2018, homoscedastic task-uncertainty
weighting) as used in FairMOT's Eq. (4)–(5), but **unbundled into three
terms instead of FairMOT's two** (FairMOT groups heatmap+box under one
"detection" uncertainty and gives re-ID its own; this implementation gives
heatmap, detection, and identity each their own learnable log-variance, to
match the "three named heads" framing of §1.5):
$$L_{total} = \frac{1}{3}\sum_{t \in \{hm,\ det,\ id\}} \Big( e^{-s_t}\, L_t + s_t \Big)$$
where $s_{hm}, s_{det}, s_{id}$ are the three entries of `log_vars`,
learned jointly with the network weights (no separate schedule or
optimizer). Intuitively, $e^{-s_t}$ is a task-specific loss scale that the
network can shrink for tasks it is confident about (reducing their pull
on shared features) at the cost of the $s_t$ penalty term, which prevents
the trivial solution of shrinking every scale to zero. **Because $s_t$ is
unconstrained, $L_{total}$ is not bounded below by zero** — see the
Results/Discussion document, §"Negative total loss", for why the reported
`train_total`/`val_total` going negative during training is expected
behavior of this formula, not a bug.

### 3.6 Full-image identity accuracy (evaluation only, not a loss)
$$\text{Identity Accuracy} = \frac{1}{N}\sum_k \mathbb{1}\!\left[\arg\max(\hat y^k) = y^k\right] \times 100\%$$
computed only over positive (labelled-object) locations, both at train time (running, on the training batch) and at validation time (`evaluate()`).

---

## 4. Technical Terms — What Was Chosen, and Why

### 4.1 Backbone: custom small encoder-decoder (FairMOT/DLA-34-*inspired*, not literal DLA-34)
- **Category**: anchor-free, keypoint/heatmap-based single-stage detector
  backbone with a U-Net-style encoder-decoder, in the CenterNet/FairMOT
  family (as opposed to two-stage/anchor-based detectors like Faster
  R-CNN, or anchor-based one-stage detectors like YOLOv3).
- **Why this family at all**: FairMOT's central argument (and the reason
  this project uses it as the joint-network reference) is that anchor-free,
  per-pixel-center detection is a better match for *joint* detection+ReID
  than anchor-based detection — anchors create a many-anchors-per-object
  ambiguity about *which* anchor's embedding should represent the object's
  identity, which anchor-free/single-center detection avoids by
  construction (exactly one embedding location per object).
- **Why *not* full DLA-34** (FairMOT's actual backbone): DLA-34 is a
  ~34-layer, multi-branch, iterative deep-layer-aggregation network with
  far more parameters and compute than justified by this project's
  dataset scale (1500 images, 30 identities — see the Results/Discussion
  doc for how quickly even this small network reaches >99% identity
  accuracy on this data). A custom 4-stage residual encoder with a
  3-step IDA-Up-style decoder (`base_ch=32`, ~1.7M total params measured)
  keeps the *architectural shape* (multi-resolution encoder, skip-connected
  decoder back to stride 4) that gives FairMOT its detection-ReID sharing
  property, at a scale that trains in seconds/epoch on this dataset rather
  than requiring a large pretrained-ImageNet backbone.
- **Why stride 4 output** (not stride 8 or 16, as some detectors use):
  matches CenterNet/FairMOT's choice — stride 4 is the accepted trade-off
  point between feature-map resolution (needed so nearby small objects
  don't collide into the same heatmap pixel — relevant here since many
  cows/buffalo can be visible and close together in one frame) and
  compute cost (halving stride quadruples feature-map area).
- **Why plain conv instead of deformable conv in the decoder**: FairMOT
  uses deformable convolutions in its up-sampling path specifically to let
  the network adapt its receptive field to object scale/pose. This was
  simplified to plain 3×3 convs — deformable conv adds implementation
  complexity and a CUDA-kernel dependency for a benefit that matters most
  at large scale/pose variance; with a fixed set of 30 known
  individuals photographed from a small number of camera rigs (see the
  dataset raw-info doc), the scale/pose variance the deformable kernels
  are meant to absorb is much smaller here.

### 4.2 Heads: category and why each one exists
- **Heatmap head** (category: *keypoint detection / dense binary
  classification*): reduces "where are the objects" to "is this pixel an
  object center", the anchor-free detection primitive. Single channel
  (class-agnostic) because both cow and buffalo are treated as the same
  "detect an animal" target for the detection sub-task — species
  distinction is not modeled by this network at all (there's no
  classification head for cow-vs-buffalo in `JDEModel`, unlike
  `JointDetReIDModel`'s `classification_head` in the n10 pipeline, which
  targets `N_CLASSES` foreground/background instead). Species is
  implicitly encoded only insofar as it correlates with which of the 30
  identity classes gets predicted.
- **Offset head** (category: *sub-pixel regression*): corrects for the
  precision lost by discretizing continuous centers onto an integer
  stride-4 grid. Needed because at stride 4, a naive round-to-nearest-cell
  center estimate can be off by up to 2 pixels in the original image —
  small in absolute terms but non-trivial relative to the box sizes here
  (30–150 px wide typical, see dataset raw-info doc's box statistics if
  added later).
- **Size head** (category: *direct box-size regression*, not
  anchor-relative like YOLO's log-space anchor scaling): directly predicts
  (w, h) in feature-map units at the object center, following CenterNet's
  simplification that anchor-free detectors don't need anchor priors for
  size — the center location already disambiguates "which object", so
  size is a free regression target.
- **Re-ID head** (category: *dense metric-learning embedding, trained via
  a proxy classification loss*): outputs a full embedding *map* (one
  vector per pixel) rather than one embedding per detected box, because at
  training time it's cheaper/simpler to supervise every pixel location
  densely and only gather the labelled ones, and at inference time (not
  yet implemented in `jde.py` — see the Results/Discussion doc's scope
  note) a dense map lets embeddings be read off directly from decoded box
  centers without a second forward pass or an ROI-pooling step.128-d
  embedding size follows FairMOT's own ablation finding (their Table 6)
  that *lower*-dimensional ReID features perform *better* in the joint
  setting than higher-dimensional ones (unlike ReID-only literature, where
  bigger embeddings often win) — attributed to lower-dimensional
  embeddings competing less aggressively with the detection task for
  shared backbone capacity.

### 4.3 Loss category: penalty-reduced focal loss + L1 + cross-entropy, combined via homoscedastic uncertainty weighting
- **Why focal loss for the heatmap** (not plain binary cross-entropy):
  every heatmap has orders of magnitude more background pixels than
  foreground (object-center) pixels; unweighted BCE would be dominated by
  the background gradient. The $(1-\hat Y)^\alpha$ / $\hat Y^\alpha$ terms
  down-weight easy/confident pixels (both correctly-confident foreground
  and correctly-confident background), and the extra $(1-Y)^\beta$ term
  further reduces the penalty near (but not exactly at) a positive center,
  since those nearby pixels are labelled with a soft Gaussian value, not a
  hard 0 — without it, "almost-correct" predictions near a true center
  would be penalized as harshly as predictions far from any object.
- **Why L1 (not L2/smooth-L1) for offset/size**: matches
  CenterNet/FairMOT's own choice; L1 is less sensitive to the occasional
  very large size outlier (e.g. a partially-occluded or edge-of-frame
  animal with an unusually large or small labelled box) than L2's squared
  penalty.
- **Why cross-entropy (not triplet/contrastive loss) for identity**: this
  is FairMOT's own deliberate choice, motivated by their finding that a
  plain softmax classification loss over a **closed, known** identity set
  works as well as metric-learning losses for the joint-training setting,
  while being simpler to implement and not requiring hard-negative mining.
  This is appropriate here specifically because the identity set is closed
  and small (exactly the same 30 known individuals appear in every
  split — see the caveat in the Results/Discussion document about what
  this does and doesn't prove about generalization to *unseen* individuals).
- **Why homoscedastic uncertainty weighting** (learnable `log_vars`) rather
  than fixed manual loss weights: the three losses have incompatible
  natural scales (a bounded-below focal loss, an unbounded L1 regression
  loss, and a 30-way cross-entropy) and, more importantly, their *relative
  difficulty* changes over training (e.g. identity classification starts
  very hard with 30 random-looking classes and gets easy quickly, per the
  Results doc's accuracy curve) — a fixed weight tuned for epoch 1 would be
  wrong by epoch 20. Learnable per-task log-variance lets the optimizer
  itself re-balance the three objectives every step.

### 4.4 Identity space: 30 classes, not the 2 species classes
- **Category distinction that matters**: `classes.txt` (2 entries: cow,
  buffalo) is the *species/category* label; `identities.txt` (30 entries)
  is the *individual-animal* label used for ReID (`cow1..cow15` →
  contiguous indices 0–14, `buffalo1..buffalo15` → 15–29, see
  `read_identities()`). `JDEModel`'s `identity_classifier` output size is
  **30**, driven by `identities.txt`, not 2 — this was a bug in an earlier
  version of `jde.py` (species label was mistakenly used as the ReID
  target, collapsing all 30 individuals into a 2-way species classifier;
  fixed by parsing the dataset's true 6-column label format
  `class cx cy w h identity` and mapping the raw identity id through
  `identities.txt`). Worth stating explicitly in the paper's dataset/method
  section since it's an easy point of confusion between "detection
  category" and "re-identification identity" that this exact codebase hit
  in practice.

---

## Change Log

- **(latest, 2026-09-13)** Initial version of this document: full
  architecture description, figure-drawing spec, complete loss/heatmap
  math (matching `jde.py` exactly), and backbone/head/loss design
  justification, written after the first successful 30-epoch training run
  (`runs/mmreplica-jde/`). See `raw_paper2_results_discussion_jde.md` for
  the corresponding empirical results.
