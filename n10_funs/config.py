"""Dataset/model constants shared across the n10 joint detection+re-ID pipeline."""
import random
from pathlib import Path

import torch
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # a few mmCows jpgs are partially written; don't crash on them

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True

torch.manual_seed(0)
random.seed(0)

# This pipeline is intentionally scoped to the identity-labelled mmCows benchmark.
# four cameras for different angles images; image names are same under different cameras

DATASET_NAME = "mmCows"
DATA_ROOT = Path("benchmarks/mmCows/visual_data/visual_data")
IMAGE_ROOT = DATA_ROOT / "images" / "0725"
LABEL_ROOT = DATA_ROOT / "labels" / "combined" / "0725"
CAMERAS = ("cam_1", "cam_2", "cam_3", "cam_4")

N_CLASSES = 1            # mmCows has exactly one object category ("cow")
N_IDENTITIES = 16        # cows C01..C16 -> label ids 1..16 in the .txt files
FRAME_STRIDE = 3         # keep every 3rd frame per camera to limit runtime
MAX_SAMPLES = 500        # cap the combined train+validation frames; use None for the full set
IMG_W, IMG_H = 640, 400  # multiple of 4, same 8:5 aspect ratio as native frames
STRIDE = 4               # output feature map is stride 4, as in FairMOT/CenterNet
FEAT_W, FEAT_H = IMG_W // STRIDE, IMG_H // STRIDE

BATCH = 8


def check_dataset():
    assert DATA_ROOT.exists(), f"Missing {DATASET_NAME} dataset: {DATA_ROOT}"
    assert all((IMAGE_ROOT / camera).exists() and (LABEL_ROOT / camera).exists() for camera in CAMERAS), \
        f"Expected image and combined-label folders for all cameras under {DATA_ROOT}"
