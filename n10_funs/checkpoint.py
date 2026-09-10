"""Atomic checkpoint save (write to .tmp, verify, then replace) with model + history + config."""
from pathlib import Path

import torch

from n10_funs.config import CAMERAS, DATASET_NAME, DATA_ROOT, FRAME_STRIDE, IMG_H, IMG_W, N_CLASSES, \
    N_IDENTITIES, STRIDE

RUN_DIR = Path("runs") / "n10_joint_det_reid"


def save_checkpoint(model, history):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = RUN_DIR / "model.pt"
    temp_checkpoint_path = RUN_DIR / "model.pt.tmp"

    checkpoint = {
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "history": history,
        "config": {
            "dataset": DATASET_NAME, "data_root": str(DATA_ROOT), "cameras": list(CAMERAS),
            "img_w": IMG_W, "img_h": IMG_H, "stride": STRIDE,
            "n_classes": N_CLASSES, "n_identities": N_IDENTITIES,
            "frame_stride": FRAME_STRIDE,
        },
    }

    torch.save(checkpoint, temp_checkpoint_path)
    try:
        verified = torch.load(temp_checkpoint_path, map_location="cpu", weights_only=False)
        if "model_state" not in verified or not verified["model_state"] or "history" not in verified:
            raise RuntimeError("checkpoint verification failed: missing model state or history")
        temp_checkpoint_path.replace(checkpoint_path)
    finally:
        if temp_checkpoint_path.exists():
            temp_checkpoint_path.unlink()

    print(f"saved and verified {DATASET_NAME} model + history to {checkpoint_path}")
    return checkpoint_path
