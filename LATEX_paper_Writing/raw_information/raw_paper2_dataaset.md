# Paper 2 — MMCows Data Replica: Full Change Log & Reference

This document tracks **everything** built/changed for the "paper 2" dataset
generation pipeline (the MMCows-style replica: 4 side cameras, cow/buffalo
detection + identity, day/night lighting, wandering herd). It is meant to be
the single source of truth so future sessions don't have to re-derive
decisions from chat history.

**Update policy: this file must be kept up to date.** Any future change to
the world file, the `Paper2_DataReplica_MMCows` / `HerdWander` /
`DayNightCycle` controllers, or the dataset output format must be reflected
here in the same session, in the relevant section below (add a new dated
entry under "Change Log" too). Do not skip entries even for small tweaks.

---

## 1. Scope / Target World

- **World file**: `worlds/umair_14_2 slides_one_top_camera_for_Recording.wbt`
  — this is the **only** world file this pipeline touches. Other worlds
  (e.g. `umair_10_3Top_4Side_cameras_now_record_as_much_as_possible.wbt`,
  `umair_12`, `umair_13`) are separate experiments and must not be modified
  by this work.
- Despite its filename (leftover from an earlier iteration), this world now
  has **exactly 4 cameras total** — no top camera, no supervisor recording
  camera. See §3.

---

## 2. Animals

- **15 buffalo** (`buffalo1`–`buffalo15`) and **15 cow** (`cow1`–`cow15`) —
  equal counts, by request, for balanced class distribution during training.
  - Originally the world had 15 buffalo / 35 cow; `cow16`–`cow35` were
    deleted to bring cow count down to 15.
- Each buffalo/cow `Robot` node's **`model` field** was changed from the
  shared default `"cow"` to its own unique instance name (e.g. `model
  "buffalo7"`, `model "cow3"`), matching its `name` field. This is required
  by the capture controller (§4) to read per-instance identity + class
  directly from Webots' recognition API instead of doing segmentation-image
  color matching.
- **Movement**: previously the herd was completely static (`controller
  "<none>"` on every animal, no motors). A dedicated supervisor
  (`HerdWander`, §5) now moves them directly by writing their `translation`
  field every timestep, since these Robot nodes have no wheels/motors of
  their own.

---

## 3. Cameras

4 side-view cameras total, all running the same controller
`Paper2_DataReplica_MMCows` (§4):

| Robot name | Prefix used in filenames | Origin (umair_10 camera it was copied from) |
|---|---|---|
| `FT` | `FT` | `camera_robot` — translation `50.98 36.6591 5.98227`, rotation `1 0 0 1.8326` |
| `FL` | `FL` | `camera_robot(1)` — translation `54.8639 -22.6183 7.43621`, rotation `3.470740371242349e-07 0.7933520848596725 0.608763065115395 3.14159` |
| `FR` | `FR` | `camera_robot(2)` — translation `85.183 12.8007 8.43621`, rotation `0.621514650930745 -0.621514650930745 -0.4769057321493091 2.25159` |
| `FB` | `FB` | `camera_robot(3)` — translation `10.5163 11.2371 9.43621`, rotation `0.6215158574562867 0.621514857456516 0.47690389062282057 2.25159` |

Naming history (for context, not to be repeated):
1. Initially there were only 3 side cameras (`camera_robot_2/3/4`, controller
   `CameraStream`) plus a 4th top-mounted "supervisor recording" camera
   (`camera_robot`, controller `CowMovement_Yolo`) and 3 more top cameras
   (`camera_robot_top_a/b/c`, controller `CameraStream`).
2. Renamed `camera_robot_2/3/4` → `side1/side2/side3`, added a new `side4`
   (placeholder position at the time).
3. Renamed `side1..4` → `FT/FL/FR/FB` (all camera device names were
   observed to start with `F`, e.g. the old top camera's file prefix was
   `F`; the user asked for consistent `F`-prefixed names for all 4).
4. **All top cameras and the supervisor recording camera were removed**
   (`camera_robot`, `camera_robot_top_a`, `camera_robot_top_b`,
   `camera_robot_top_c`) — the requirement settled on **4 cameras total**.
5. FT/FL/FR/FB's translation+rotation were then overwritten with the 4
   corner-camera transforms copied from `umair_10_3Top_4Side_cameras_...wbt`
   (see table above), per explicit request to reuse that world's camera
   placement/orientation instead of ad-hoc/placeholder values.

Each camera `Robot` node still has: `Camera` device named `"camera"`,
`width 640`, `height 480`, `fieldOfView 1`, `recognition Recognition
{ frameThickness 0 segmentation TRUE }` (segmentation flag is currently
unused by the controller — see §4 — but left enabled in case a future
controller variant needs it).

---

## 4. Capture controller — `Paper2_DataReplica_MMCows`

Path: `controllers/Paper2_DataReplica_MMCows/Paper2_DataReplica_MMCows.py`

### What it does
Each of the 4 camera robots runs this same controller as its own OS
process. Every simulation step it:
1. Grabs the camera image (`cam.getImage()`).
2. Computes a dHash of the frame and compares it (Hamming distance) against
   all previously-saved frames **from that same camera** (matched by
   filename prefix). If the closest match has Hamming distance
   `< SIMILAR_FRAME_HAMMING_THRESHOLD` (10), the frame is **skipped** — this
   is the "Hamming window" duplicate filter so a static/slow scene doesn't
   flood the dataset with near-identical images.
3. If the frame is novel enough, reads `cam.getRecognitionObjects()` —
   Webots' built-in per-object recognition list, which already gives pixel
   bounding boxes (`position_on_image`, `size_on_image`) and is
   occlusion-aware (objects hidden behind others are not returned). **No
   segmentation image / color-matching is used** — this was an earlier
   approach (see history below) replaced because it's simpler and more
   accurate.
4. For each recognized object, reads its `model` field string (e.g.
   `"cow7"`, `"buffalo3"`) and parses out:
   - **class name**: `cow` or `buffalo` (case-insensitive prefix match)
   - **class_id**: `0` for cow, `1` for buffalo (from `dataset-mmreplica/classes.txt`)
   - **identity_id**: global unique integer — `cow1..cow15` → `1..15`,
     `buffalo1..buffalo15` → `101..115` (offset by 100 so the two classes'
     IDs never collide).
5. Writes the image and a label file to disk (see §6 for output format).
6. The first time any camera sees a given identity, it's appended to
   `dataset-mmreplica/identities.txt` as `<identity_id> <model_name>`. This
   file is a shared source of truth across all 4 camera processes — before
   appending, the controller re-reads the file to check the ID isn't
   already there (each process seeds its own in-memory "seen" set from the
   file at startup too), preventing the duplicate-line bug described below.

### Design history / decisions made along the way
- **v1**: Used segmentation image (`getRecognitionSegmentationImage()`) +
  per-class unique colors to compute bounding boxes by pixel-mask
  color-matching, and wrote plain YOLO labels (`class_id xc yc w h`) with no
  identity — this mirrored the older `CameraStream` controller. This was the
  very first version of this controller.
- **v2 (YOLO removed)**: User said "now it should not do any yolo, i just
  need to make dataset" → stripped all class/label logic, controller only
  saved plain JPGs with Hamming-distance dedup, no `dataset/labels`
  directory at all.
- **v3 (bbox + class + identity added back, differently)**: User then asked
  for bounding box + class (cow/buffalo, 2 classes) + per-instance identity
  (cow1, cow2, ..., buffalo1, ...) in one frame, plus balanced 15/15 counts
  (see §2). Rather than reintroducing segmentation-based bbox math, switched
  to `Camera.getRecognitionObjects()` (simpler, gives real pixel bboxes
  directly, occlusion-aware) — this required setting each animal's `model`
  field to its unique name (§2) since Webots recognition objects only
  expose a `model` string, not the Robot's `name` field.
- **Output directory renamed**: originally wrote to `dataset/` (shared with
  other older controllers like `CameraStream`/`CowMovement_Yolo`). Renamed
  to **`dataset-mmreplica/`** so this pipeline's output never mixes with
  older datasets in `dataset/`.
- **Camera prefix renames**: prefix map went `S1/S2/S3/S4` (when robots were
  named `side1..4`) → `FT/FL/FR/FB` (matching final robot names, §3).
- **`identities.txt` duplicate-line bug**: because the 4 cameras are 4
  separate OS processes, each had its own in-memory "seen identities" set;
  each independently wrote the first line it saw for a given identity,
  producing up to 4 duplicate lines per identity (and more across sim
  restarts, since the in-memory set doesn't persist). Fixed by having
  `append_identity_record()` re-check the file on disk (the real shared
  state) before writing, and by seeding the in-memory set from the file at
  controller startup. The existing duplicated file was cleaned once
  (deduplicated keeping first occurrence per ID).
  - **Note on identity IDs themselves**: the ID was never actually
    camera-dependent — it's deterministically derived from the animal's own
    `model` field (`parse_instance()`), so `cow7` always gets ID `7`
    regardless of which camera sees it. Only the *logging* of first-seen
    identities to `identities.txt` had the duplicate-line bug, not the
    per-frame label IDs.

### Key constants (as of last update)
```python
CAMERA_DEVICE_NAME = "camera"
RECORD_EVERY_N_STEPS = 1
SIMILARITY_CHECK_SIZE = (160, 90)          # dHash working resolution
SIMILAR_FRAME_HAMMING_THRESHOLD = 10        # below this distance = duplicate, skip
CAMERA_PREFIX_BY_ROBOT = {"FT": "FT", "FL": "FL", "FR": "FR", "FB": "FB"}
CLASS_NAMES = ["cow", "buffalo"]            # class_id 0 and 1 respectively
```

---

## 5. Herd movement controller — `HerdWander`

Path: `controllers/HerdWander/HerdWander.py`

### Why
Originally requested because a fully static herd gives cameras no variety —
"cows and buffalos should move and should be scattered slowly so camera
could do better different images and do more generalization."

### How it works
- Runs as a **Supervisor** robot (`herd_wander_supervisor`, added near the
  top of the world file, no visible geometry — controller-only node).
- At startup, scans `supervisor.getRoot()`'s children for every `Robot`
  whose `name` field matches `^(cow|buffalo)\d+$` (regex), and keeps a
  handle to each one's `translation` field.
- Every simulation step, each animal:
  1. Has its heading angle perturbed by a small random turn
     (`MAX_TURN_RATE = 0.25` rad/s max), giving a smooth correlated random
     walk instead of jittery direction changes.
  2. Moves forward along that heading at `SPEED_MPS = 0.15` m/s (slow,
     grazing pace).
  3. Calls `node.resetPhysics()` after each direct translation write, since
     Supervisor field writes bypass the physics engine and residual
     velocity/contact state should be cleared so the rigid body doesn't
     fight the imposed motion.

### Enclosure / boundary behavior (important fix)
- **v1**: each animal was constrained to wander within
  `WANDER_RADIUS = 12.0` m of **its own starting position** (steer back
  toward its own origin once outside the radius). Problem reported by user:
  "this cows are now going out of bar[n]" — actually the deeper issue was
  that constraining each animal to its own small personal radius meant they
  never explored the wider pen, not that they left it; the user wanted the
  whole herd to be able to **fill the entire enclosure**, not idle in 30
  disjoint small circles.
- **v2 (current)**: replaced per-animal radius with **one shared enclosure
  rectangle** that all animals bounce inside:
  ```python
  ENCLOSURE_X_MIN = 24.0
  ENCLOSURE_X_MAX = 74.0
  ENCLOSURE_Y_MIN = -13.0
  ENCLOSURE_Y_MAX = 22.0
  ```
  These bounds were inset from the picket-fence perimeter found in the
  world file (`OpenFieldFence` nodes forming a rectangle roughly
  `x:[20.2, 77.9], y:[-16.3, 26.2]`). When an animal's next position would
  cross a wall, its heading is **reflected** (`heading = pi - heading` for
  an X-wall hit, `heading = -heading` for a Y-wall hit) so it turns around
  and heads back inward — described by the user as "go away, rotate and
  come back" — instead of stopping dead or teleporting. Over time this lets
  every animal's random walk cover the whole enclosure ("each cow will fill
  bar[n]").
  - **Caveat**: these bounds were inferred from the `PicketFence` node
    translations in the world file, not confirmed against the actual
    barn/pen geometry by the user. If the real enclosure differs, these 4
    constants are the only thing that needs updating.

### First deployment bug (movement didn't happen at all)
- The very first time the `herd_wander_supervisor` node was added to the
  world file via a text edit, it silently disappeared from the file before
  the user next reloaded (most likely Webots re-saved/reformatted the world
  and dropped it, or it was removed by hand after appearing to do nothing).
  Result: **zero movement**, not just slow movement, because the controller
  was never running.
- Fixed by re-adding the node, and added a startup log line
  (`[HerdWander] tracking N cow/buffalo nodes`) plus a heartbeat print every
  ~5 simulated seconds, specifically so this "is it even running" question
  can be answered by checking the Webots console instead of guessing.
- Also bumped `SPEED` (originally named confusingly, effectively ~0.04 m/s)
  up to the current `SPEED_MPS = 0.15` since the original value was too
  subtle to visually notice over a short observation window.

### Key constants (as of last update)
```python
NAME_PATTERN = re.compile(r"^(cow|buffalo)\d+$", re.IGNORECASE)
SPEED_MPS = 0.6      # bumped up from 0.15 - that was confirmed running
                     # (console showed the tracking log) but too subtle to
                     # visually notice as "moving" over a short observation
MAX_TURN_RATE = 0.4  # bumped up from 0.25 alongside the speed increase
ENCLOSURE_X_MIN, ENCLOSURE_X_MAX = 24.0, 74.0
ENCLOSURE_Y_MIN, ENCLOSURE_Y_MAX = -13.0, 22.0
```

---

## 6. Lighting controller — `DayNightCycle`

Path: `controllers/DayNightCycle/DayNightCycle.py`

### Why
User's concern: "with one side light one camera may always detect dark
texture" — a single fixed light source means whichever camera faces away
from it always sees dark/backlit animals, biasing the dataset. Requested a
full day/night cycle: sun moving east→west, plus night-time lamp lighting,
"for complete dynamic generalization" of texture/color/lighting conditions.

### How it works
Runs as a second Supervisor robot (`day_night_supervisor`, controller
`DayNightCycle`), independent from `HerdWander`. Every step it computes a
repeating cycle based on `supervisor.getTime()` (simulation time), so it's
resilient to pausing/resetting rather than relying on a manually-incremented
counter:

```python
DAY_SECONDS = 120.0
NIGHT_SECONDS = 60.0
CYCLE_SECONDS = DAY_SECONDS + NIGHT_SECONDS   # 180s full cycle
```

- **During the day segment** (`t < DAY_SECONDS`, phase `p = t/DAY_SECONDS`
  going 0→1 from sunrise to sunset):
  - `elevation_factor = max(0, sin(p * pi))` — 0 at sunrise/sunset, 1 at
    solar noon.
  - Sun direction vector: `[-cos(p*pi), 0, -max(sin(p*pi), 0.02)]` — sweeps
    from a low grazing angle out of the east (`p=0`), to straight down at
    noon (`p=0.5`), to a low grazing angle out of the west (`p=1`).
  - Sun color interpolates from warm orange `[1.0, 0.55, 0.28]` (low
    elevation, i.e. dawn/dusk) to neutral white `[1.0, 0.97, 0.9]` (high
    elevation, i.e. noon), scaled by `elevation_factor`.
  - Sun intensity: `SUN_MAX_INTENSITY (3.5) * elevation_factor`.
- **During the night segment** (`t >= DAY_SECONDS`): `elevation_factor = 0`
  for the whole night (sun off, direction frozen at last sunset value since
  it doesn't matter with zero intensity).
- **Sky brightness** (`DEF SKY` / `DEF SKY_LIGHT` `luminosity` fields):
  interpolated between `SKY_NIGHT_LUMINOSITY = 0.05` (not pure black, so
  there's still faint ambient visibility) and `SKY_DAY_LUMINOSITY = 1.0`,
  driven by the same `elevation_factor`.
- **Night lamps** (`DEF NIGHT_LAMP_1..4`, `PointLight` nodes): intensity
  `LAMP_MAX_INTENSITY (4.0) * (1 - elevation_factor)` — fully off at noon,
  ramping up through dusk, fully on all night, ramping back down through
  dawn. Warm color `[1, 0.85, 0.6]`.

### World file additions this required
- `TexturedBackground {}` → `DEF SKY TexturedBackground { luminosity 1 }`
- `TexturedBackgroundLight { castShadows FALSE }` → `DEF SKY_LIGHT
  TexturedBackgroundLight { castShadows FALSE luminosity 1 }`
- New `DEF SUN DirectionalLight { direction -1 0 -0.2 intensity 3.5 color 1
  0.97 0.9 castShadows TRUE }`
- 4 new `PointLight` nodes, `DEF NIGHT_LAMP_1..4`, placed at the 4 corners
  of the same enclosure rectangle `HerdWander` uses (`(24,-13)`, `(74,-13)`,
  `(24,22)`, `(74,22)`), all at height `z=7`, `intensity 0` by default,
  `attenuation 1 0 0.01`, `radius 45`, warm color `1 0.85 0.6`,
  `castShadows TRUE`.
- New supervisor Robot node `day_night_supervisor` running `DayNightCycle`.

### Caveats / things not yet verified
- Lamp post positions reuse the `HerdWander` enclosure corners as a
  reasonable guess, not confirmed against real barn/fence geometry — same
  caveat as §5.
- `TexturedBackground`/`TexturedBackgroundLight` are Cyberbotics external
  PROTOs; their `luminosity` field is assumed exposed (standard for these
  PROTOs as of R2025a) — if a future Webots version changes this interface,
  `sky_luminosity_field`/`sky_light_luminosity_field` could come back
  `None` and the controller already guards for that (prints a warning,
  skips animating that field) but won't error.
- Day/night sun arc is a simplified geometric model (not astronomically
  accurate — it doesn't model latitude/season), it's deliberately just
  "good enough to visually change lighting direction/color/intensity over
  time."

---

## 7. Dataset output format (`dataset-mmreplica/`)

```
dataset-mmreplica/
├── classes.txt        # 2 lines: "cow" then "buffalo" (class_id = line index)
├── identities.txt      # "<identity_id> <model_name>" per known animal, one line per ID, no duplicates
├── images/
│   ├── FT000001.jpg    # <prefix><6-digit index>.jpg, prefix = FT/FL/FR/FB
│   ├── FL000001.jpg
│   ├── FR000001.jpg
│   └── FB000001.jpg
└── labels/
    ├── FT000001.txt     # same base name as its image
    └── ...
```

### Label file format
One line per detected animal in that frame:

```
<class_id> <xc> <yc> <w> <h> <identity_id>
```

- `class_id`: `0` = cow, `1` = buffalo (see `classes.txt`)
- `xc, yc, w, h`: standard YOLO-normalized bounding box (center x/y, width,
  height, all in `[0, 1]` relative to image width/height)
- `identity_id`: extra 6th column (**not** standard YOLO format) — global
  per-instance ID: `1..15` = `cow1..cow15`, `101..115` = `buffalo1..buffalo15`.
  Consumers that expect strict 5-column YOLO labels will need to either
  ignore or strip this column.
- A frame with zero detections still gets an (empty) label file written.

### `identities.txt` format
```
<identity_id> <model_name>
```
e.g. `7 cow7`, `103 buffalo3`. One line per unique identity ever observed by
any of the 4 cameras across the whole run (append-only, deduplicated by ID).

### Final dataset snapshot (as of 2026-09-13)

Actual measured contents of `dataset-mmreplica/` at time of writing —
**update this snapshot whenever a new recording session changes these
numbers.**

| Metric | Value |
|---|---|
| Total images | 1500 |
| Total label files | 1500 (1:1 with images, none missing) |
| Images per camera | FT: 276, FL: 406, FR: 418, FB: 400 |
| Image resolution | 640×480 RGB JPEG (matches `Camera { width 640 height 480 }`) |
| Total on-disk size | ~183 MB |
| `classes.txt` | 2 lines: `cow`, `buffalo` |
| `identities.txt` | 30 lines, one per animal, IDs `1–15` (cow1–15) and `101–115` (buffalo1–15) — all valid, no corruption (see Change Log fix) |
| Total detection lines (all label files combined) | 29,642 |
| Empty label files (zero detections in frame) | 0 — every saved frame had at least one visible animal |
| Avg detections per image | ~19.76 (expected — 30 animals total, wide-FOV corner cameras, so most of the herd is visible in most frames) |
| Max detections in a single image | 29 (i.e. nearly the whole herd visible in one frame at least once) |
| Class balance in labels | cow: 15,042 detections (class 0), buffalo: 14,600 detections (class 1) — balanced, consistent with the 15/15 animal split |
| Identity coverage | all 30 identities appear in the labels (761–1185 detections per individual) — no animal is missing/under-represented to the point of being unusable |

**Interpretation / notes:**
- The per-camera image counts (276–418) differ because the Hamming-distance
  duplicate filter (§4) is independent per camera — a camera with less
  herd motion in its field of view accumulates near-duplicate frames faster
  and therefore saves fewer novel ones (`FT` clearly saved the fewest,
  suggesting less visual change happened in its view across the session
  relative to FL/FR/FB).
- Zero empty label files plus a high average detection count per frame
  means the dataset is **detection-rich but not occlusion/hard-negative
  rich** — there's no frame with 0 or 1 animals to teach a model what "mostly
  empty pasture" or "single isolated animal" looks like. If the paper's
  goal includes robustness to sparse scenes, consider whether that matters
  before treating this as a final training set.
- The near-even min/max detections-per-identity (761 vs 1185) means no
  single animal is drastically under-represented, which is good for the
  overfitting concern already flagged in the Change Log — the imbalance
  that does exist is mild (~1.5x), not severe.

### Note on the older `dataset/` directory
`dataset/` (no `-mmreplica` suffix) is a **separate, older** dataset
directory used by other controllers (`CameraStream`, `CowMovement_Yolo`,
etc. — not part of this pipeline). `dataset/classes.txt` was recreated at
one point (`cow`, `buffalo`) since it had been deleted, but that directory
is not written to by `Paper2_DataReplica_MMCows` and is out of scope for
this document except as a note to avoid confusing the two.

---

## 8. Open items / things worth double-checking

- [ ] Confirm the real barn/fence enclosure coordinates against
      `ENCLOSURE_X_MIN/MAX`, `ENCLOSURE_Y_MIN/MAX` in `HerdWander.py` (used
      also for night lamp placement in the world file) — currently inferred
      from `PicketFence` node translations, not confirmed by the user.
- [ ] Confirm `DAY_SECONDS`/`NIGHT_SECONDS` (120s/60s) match the intended
      recording session length — if a full recording run is much
      longer/shorter than a few cycles, consider retuning so the dataset
      gets a good spread of lighting conditions without over- or
      under-sampling any one condition.
- [ ] `HerdWander` and `DayNightCycle` both run as separate Supervisor
      robots with no visible geometry — confirmed this is fine in Webots
      (multiple supervisors can coexist), but worth confirming no
      performance issue with many supervisor field writes per step if the
      herd/lamp count grows.

---

## Change Log

Newest first. Add a new dated entry here whenever this pipeline changes.

- **(latest, 2026-09-13)** Inspected the actual `dataset-mmreplica/` output
  and added a "Final dataset snapshot" table to §7: 1500 images / 1500
  labels (276/406/418/400 per FT/FL/FR/FB), 640×480 RGB JPEGs, ~183 MB
  total, 29,642 detection lines, 0 empty label files, balanced cow/buffalo
  class counts (15,042 / 14,600), all 30 identities represented
  (761–1185 detections each, ~1.5x spread — mild, not severe). Noted two
  things worth knowing before treating this as a final training set: (1)
  every saved frame has at least one animal — there are no sparse/empty
  frames, so the dataset doesn't teach a model what an empty or
  near-empty pasture looks like; (2) `FT` saved noticeably fewer frames
  than the other 3 cameras, consistent with less visual change happening
  in its field of view (so the per-camera dedup filter kept rejecting it as
  near-duplicate more often).
- Herd was confirmed running (console tracking log present) but
  the walking speed (0.15 m/s) was imperceptible over a short observation
  window — bumped `HerdWander.SPEED_MPS` to `0.6` and `MAX_TURN_RATE` to
  `0.4` for clearly visible movement.
- Fixed a crash-causing race condition: the 4 camera processes
  appended to `identities.txt` concurrently with no locking, which produced
  an interleaved/corrupted line (`alo14` instead of `114 buffalo14`).
  Loading that corrupted file at startup (`int(line.split()[0])`) crashed
  every `Paper2_DataReplica_MMCows` process with `ValueError` *before*
  `cam.enable()` was ever called — which is why all 4 camera views appeared
  fully black (the controllers were dead on startup, not a lighting issue).
  Fixed by: (1) a simple cross-process file lock
  (`identities.lock`, exclusive-create with stale-lock takeover after 2s)
  around every read-check-append of `identities.txt`; (2)
  `load_known_identity_ids()` now skips malformed lines instead of raising,
  so a damaged file can never crash the controller again; (3) repaired the
  existing corrupted file (30 valid identities recovered, garbage lines and
  stray `\r` dropped).
  - Confirmed "resume from where it left off" was already correct and
    needed no change: `get_next_capture_index()` scans existing filenames
    per camera prefix and continues from `highest_index + 1`, and
    `next_available_base()` skips any index that already has a file on
    disk — a restarted sim never overwrites previous captures.
  - **Overfitting note (flagged by user, worth keeping in mind for future
    tuning):** with only 15 unique cow + 15 unique buffalo identities,
    an identity-recognition model trained on this dataset risks overfitting
    to these exact 30 individuals/textures rather than generalizing to
    unseen cattle. Mitigations already in place (day/night lighting,
    herd wandering, 4 viewpoints) help variety per-identity, but if this
    becomes a real training set (not just a pipeline demo), consider
    adding more distinct 3D models/textures rather than only recording more
    frames of the same 30 animals.
-  Documented the entire pipeline in this file for the first
  time, consolidating all decisions made across the conversation: 4-camera
  layout copied from `umair_10`, 15/15 cow/buffalo balance, bbox+class+ID
  labeling via `getRecognitionObjects()`, `dataset-mmreplica/` output,
  `HerdWander` enclosure-bounce movement, `DayNightCycle` lighting.
- Added `DayNightCycle` supervisor controller + `DEF SUN`/`DEF
  SKY`/`DEF SKY_LIGHT`/`DEF NIGHT_LAMP_1..4` nodes to the world for dynamic
  day/night lighting.
- Fixed `HerdWander` to bounce animals within one shared enclosure rectangle
  instead of a per-animal personal radius, so the whole herd can roam the
  full pen.
- Fixed `identities.txt` duplicate-line bug (cross-process re-check before
  append) and cleaned the existing duplicated file.
- Re-added the `herd_wander_supervisor` node after it disappeared from the
  world file on first attempt (movement wasn't happening at all); added
  startup/heartbeat logging and increased wander speed.
- Added `HerdWander` supervisor controller so the (motor-less) cow/buffalo
  Robots slowly scatter instead of sitting static.
- Copied FT/FL/FR/FB translation+rotation from `umair_10`'s 4 corner
  cameras, replacing earlier placeholder/self-set positions.
- Removed the top-mounted supervisor recording camera (`camera_robot`,
  `CowMovement_Yolo`) and the 3 top cameras (`camera_robot_top_a/b/c`,
  `CameraStream`) — settled on 4 cameras total.
- Renamed side cameras `side1..4` → `FT/FL/FR/FB`; updated
  `CAMERA_PREFIX_BY_ROBOT` in the controller to match.
- Renamed output directory from `dataset/` to `dataset-mmreplica/` in
  `Paper2_DataReplica_MMCows.py`.
- Reworked bounding-box/class/identity extraction to use
  `Camera.getRecognitionObjects()` instead of segmentation-image color
  matching; set each animal's `model` field to its unique instance name to
  support this; trimmed `cow16..cow35` to balance 15 cow / 15 buffalo.
- Stripped all YOLO/segmentation/label logic per request ("i just need to
  make dataset") — controller briefly saved plain images only.
- Created `Paper2_DataReplica_MMCows` controller (v1: segmentation-based
  YOLO labels, no identity) and renamed `camera_robot_2/3/4` →
  `side1/side2/side3`, added new `side4` — first version of the 4-side-
  camera dataset pipeline.
