# Paper 2 — JDE (Joint Network) Results & Discussion: Raw Notes

Raw source material for the **Results** and **Discussion** sections
covering the joint detection+ReID network (`jde.py`, model `JDEModel`,
architecture described in `raw_paper2_joint_network.md`). Numbers below are
taken directly from the actual training run in `runs/mmreplica-jde/`
(`history.csv`, 30 rows, one per epoch) and the console log of that run.

**Update policy**: if `jde.py` is re-run (new hyperparameters, fixed
train/val split, more data, etc.), re-generate the numbers in this document
from the new `runs/.../history.csv` rather than editing them by hand, and
add a dated Change Log entry noting what changed and why the numbers moved.

---

## 1. Run Configuration (as executed)

| Setting | Value |
|---|---|
| Command | `python jde.py` (all defaults — no CLI flags) |
| Dataset | `benchmarks/dataset-mmreplica` (2 classes: cow, buffalo; 30 identities) |
| Images | 1500 total |
| Train samples | 1500 |
| Val samples | 1500 |
| Epochs (config) | 30 |
| Patience (early stop) | 5 |
| Batch size | 4 |
| Learning rate | 1e-4 (cosine annealed to ≈0 over 30 epochs) |
| Image size | 640×400 |
| Stride | 4 |
| Device | CUDA — NVIDIA GeForce RTX 4060 Laptop GPU |
| Model parameters | 1,738,438 (≈1.74M, trainable) |
| Output dir | `runs/mmreplica-jde/` |
| Result | Ran the full 30 epochs (early stopping never triggered — val loss kept improving every single epoch) |
| Per-epoch wall time | 25.9 s (epoch 2, fastest) → 47.8 s (epoch 19, slowest); no clear trend, consistent with shared-GPU/thermal/OS scheduling noise rather than a model-size or data effect |
| Total wall time (approx.) | ~17.5 minutes for all 30 epochs |

### 1.1 Critical caveat that must be stated before any other interpretation
**`dataset-mmreplica` currently has no `train.txt`/`val.txt`/`test.txt` split
files.** `index_split()` falls back to globbing *all* images in that case,
so `train_dataset` and `val_dataset` are built from the exact same 1500
images in the exact same (sorted-filename) order — confirmed by the run log
(`train=1500 | val=1500`, and both datasets constructed via
`WebotsJDEDataset(DATA_ROOT, "train"/"test", ...)` hitting the same glob
fallback since neither `val.txt` nor `test.txt` exists).

**This means every "val_*" metric reported below is a second measurement on
the training set, not a held-out generalization measurement.** The near-100%
val identity accuracy and the close train/val loss tracking are exactly
what you'd expect from evaluating on data the model was trained on, and
**cannot yet be used to claim the model generalizes to unseen
images/individuals.** Before these results are used as the paper's
headline numbers, a real split (e.g. `identities.txt`-stratified image-level
split, or held-out camera views) should be generated. This document's
per-epoch numbers are still legitimate evidence of *optimization behavior*
(does the joint loss converge stably, does identity accuracy on seen data
climb as expected) — just not yet of generalization.

---

## 2. Headline Numbers (final epoch, 30/30)

| Metric | Train | Val |
|---|---|---|
| Total loss | −0.2582 | −0.2672 |
| Heatmap loss | 0.0706 | 0.0626 |
| Detection loss (offset + 0.1·size) | 0.2043 | 0.2007 |
| Offset loss (raw) | 0.1258 | — (not separately logged for val) |
| Size loss (raw) | 0.7854 | — (not separately logged for val) |
| Identity (cross-entropy) loss | 0.0288 | 0.0231 |
| Identity accuracy | 99.52% | 99.65% |
| Learning rate | 2.74e-07 (annealed to ~0) | — |

Best checkpoint: **epoch 30** (the last epoch — validation total loss
improved monotonically every single epoch, `best_val` was reset 30/30
times, see §3).

---

## 3. Training Dynamics (epoch-by-epoch behavior)

### 3.1 Overall shape
All three loss components (heatmap, detection, identity) decrease
**monotonically and smoothly** across all 30 epochs, with no oscillation,
divergence, or instability at any point — e.g. identity loss:
3.05 → 2.72 (epoch 1) → 0.19 (epoch 14) → 0.023 (epoch 30); heatmap loss:
1.18 (epoch 1) → 0.070 (epoch 30). This is a strong indication that: (a)
the learning rate (1e-4, cosine-annealed) was well-chosen for this model
size/dataset — no epoch shows the loss spike typical of an LR that's
briefly too high; (b) the uncertainty-weighted multi-task loss (§3.5 of the
architecture doc) is not producing the destabilizing competing-gradients
behavior that naively-summed multi-task losses often show early in
training.

### 3.2 Three convergence phases (visible in the identity-accuracy column)
- **Epochs 1–5 (coarse phase)**: id accuracy 10.5% → 48.7% (train). The
  model is still primarily learning to detect (heatmap loss falls fastest
  here, 1.18 → 0.66) — identity classification on a not-yet-reliable
  embedding is necessarily weak.
- **Epochs 6–14 (rapid-improvement phase)**: id accuracy 59.7% → 94.8%.
  Detection is now "good enough" (heatmap loss 0.61 → 0.30) that the
  embedding head starts getting a clean, consistently-located training
  signal at true object centers, and identity accuracy climbs fastest in
  this window.
- **Epochs 15–30 (fine-tuning / saturation phase)**: id accuracy
  95.9% → 99.5%, heatmap loss continues a slow decline (0.27 → 0.07),
  learning rate has annealed to near-zero by the end. Marginal returns per
  epoch shrink steadily — e.g. the last 10 epochs (21→30) only move
  identity accuracy from 98.8% to 99.5%, a ~0.7-point gain, vs. a
  ~35-point gain in the 6-epoch window 6→12 alone.

### 3.3 Detection sub-losses did **not** saturate as fast as identity/heatmap
By epoch 30, `train_size` (0.785) is still the single largest raw loss
component in absolute terms — much larger than heatmap (0.071) or offset
(0.126) at the same epoch — even though it's down-weighted by
$\lambda_{size}=0.1$ inside `L_det`. This is worth flagging explicitly in
the discussion: **box-size regression is the slowest-converging sub-task**
here, plausibly because (a) box sizes in this dataset vary a lot with
camera distance/angle (4 different camera rigs, per the dataset raw-info
doc) giving a wide, harder-to-fit target distribution, and (b) L1 loss on
size in feature-map units doesn't get the benefit of the focal loss's
easy-example down-weighting the way the heatmap does. A follow-up
worth trying: log the *unweighted* `train_size` trend on its own axis in
the paper's convergence figure (not just folded into `train_detection`) so
this doesn't get visually hidden behind the much smaller heatmap/offset
curves.

### 3.4 No early stopping — is `patience=5` doing anything?
No. Validation total loss improved every one of the 30 epochs (`best_val`
was reset every single epoch in the console log — every line prints
`| best`, none print "no improvement"), so the `patience=5` early-stopping
gate was never exercised at all in this run. This says relatively little
about whether patience=5 is well-tuned in general — it just means 30
epochs wasn't enough for this run to plateau (or, per the caveat in §1.1,
that evaluating "validation" on the training set itself makes overfitting-
triggered early stopping structurally unlikely to ever fire). Re-running
with a genuine held-out split and/or more epochs is needed before
`patience` can be meaningfully evaluated.

### 3.5 Negative total loss — expected, not a bug
`train_total`/`val_total` cross from positive to negative around epoch
15–16 (train: 0.029 at epoch 15 → −0.010 at epoch 16) and keep decreasing
to −0.258 / −0.267 by epoch 30. **This is expected behavior of the
homoscedastic uncertainty-weighted loss** (§3.5 of the architecture doc),
not a sign that some component loss went negative — every individual
component (`heatmap`, `detection`, `identity`) stays strictly positive for
all 30 epochs, as shown in the table columns above. The combined objective
$L_{total} = \frac13\sum_t (e^{-s_t}L_t + s_t)$ has **no lower bound at
zero** once the learnable log-variances $s_t$ are allowed to become very
negative (which happens naturally as the network grows confident on each
task and the optimizer shrinks that task's effective weight) — the $s_t$
term itself then dominates and drags the sum below zero. **When writing
this up, explicitly state this so a reader doesn't mistake a negative
total loss for an error in the loss implementation** — cite Kendall,
Gal & Cipolla (2018) for why this is a known, intentional property of the
technique.

---

## 4. What This Run Does *Not* Yet Measure (scope limitations)

These are honest gaps to state in the Discussion / Limitations section,
not to hide:

1. **No held-out generalization measurement** — see §1.1. All numbers here
   are train-set (self-evaluation) numbers, whatever split label they carry
   in `history.csv`.
2. **No object-detection metric (mAP/precision/recall) at all.** `jde.py`
   only ever measures detection quality indirectly, via L1 regression loss
   on offset/size **at ground-truth center locations** — it never runs a
   `decode()`/peak-extraction/NMS step to produce actual predicted boxes
   the way `sequence.py`'s `Detector` + `decode()` does. This means we
   currently cannot state a detection accuracy number (e.g. "the joint
   model detects X% of animals") for the joint network at all — only a
   proxy regression-quality number. Adding a `decode()`-based evaluation
   pass to `jde.py` (mirroring `sequence.py`'s) is a prerequisite before
   the paper can compare joint-vs-sequential detection quality head to
   head, not just training-loss curves.
3. **No tracking-quality (MOTA/IDF1) evaluation.** The re-ID embedding is
   only ever evaluated via classification accuracy against the 30 known
   training identities (a *closed-set* metric) — there is no multi-frame
   tracking evaluation exercising the embedding as an actual re-ID/tracking
   feature (the use case that motivates having a dense embedding map at
   all, per §4.2 of the architecture doc).
4. **Closed identity set.** All 30 individuals that ever appear in the
   dataset appear in every split (trivially true right now since train=val
   are literally the same images) — this setup can only ever demonstrate
   *closed-set* re-identification (classify among 30 known individuals),
   not *open-set* re-ID (recognize/reject an individual never seen during
   training), which is the harder and more practically relevant capability
   real deployment would need.
5. **Single run, no seeds/variance.** One training run, one seed — no
   error bars, no repeated-run variance estimate. Given how smoothly this
   run converged (§3.1), variance is plausibly low, but that's an
   assumption, not a measurement.

---

## 5. Suggested Framing for the Paper's Results Section

- Lead with the convergence plot (train/val heatmap, detection, identity
  loss vs. epoch, plus identity accuracy vs. epoch) — the three-phase shape
  in §3.2 is a clean, presentable story about how the joint model's
  sub-tasks bootstrap each other (detection quality gating identity
  learnability).
- Report final-epoch numbers from §2 as the headline table, but the
  Discussion **must** carry the §1.1 caveat prominently, ideally as a
  footnote directly on the results table itself, not buried in prose —
  a reviewer who reads only the table would otherwise reasonably assume
  "val" means held out.
- Use §3.5 (negative loss) as a short methodological aside/footnote near
  wherever the loss formula is first shown — this is the kind of detail
  a reviewer unfamiliar with uncertainty weighting will otherwise flag as
  an apparent error.
- Use §4 as the explicit "Limitations" list, and frame the natural next
  experiments as: (a) real train/val/test split, (b) add `decode()`-based
  detection metrics to `jde.py`, (c) compare against the `sequence.py`
  two-stage baseline on the *same* metrics once (a) and (b) exist — this
  is, after all, exactly what the "systematically move to best
  architecture" project is set up to compare.

---

## Change Log

- **(latest, 2026-09-13)** Initial version, written from the first
  successful 30-epoch `jde.py` run (`runs/mmreplica-jde/history.csv` +
  console log). Flagged the train==val split issue as the single most
  important caveat before these numbers can be used as headline results;
  documented the three-phase convergence pattern, the slower-converging
  size-regression sub-loss, and why total loss goes negative (uncertainty
  weighting, not a bug). No detection-mAP or tracking metrics exist yet for
  the joint model — noted as required follow-up work.
