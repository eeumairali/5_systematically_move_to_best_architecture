"""Building the (image_path, boxes) frame index from the mmCows label files."""
from PIL import Image

from n10_funs.config import CAMERAS, DATASET_NAME, FRAME_STRIDE, IMAGE_ROOT, LABEL_ROOT, N_IDENTITIES


def parse_yolo_txt(path):
    """Read one 'id cx cy w h' label file."""
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts or len(parts) < 5:
            continue
        cid = int(parts[0])
        cx, cy, w, h = (float(value) for value in parts[1:5])
        if not 1 <= cid <= N_IDENTITIES:
            raise ValueError(f"Unexpected {DATASET_NAME} identity {cid} in {path}")
        boxes.append((cid, cx, cy, w, h))
    return boxes


def is_readable_image(path):
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        print(f"skipping unreadable image: {path}")
        return False


def load_frame_index():
    """Build usable (image_path, boxes) samples across all configured cameras.
    Subsampling restarts for each camera so every view contributes equally."""
    samples = []
    for camera in CAMERAS:
        image_dir = IMAGE_ROOT / camera
        label_dir = LABEL_ROOT / camera
        camera_frames = 0
        for i, txt_path in enumerate(sorted(label_dir.glob("*.txt"))):
            if i % FRAME_STRIDE != 0:
                continue
            boxes = parse_yolo_txt(txt_path)
            if not boxes:
                continue
            img_path = image_dir / (txt_path.stem + ".jpg")
            if img_path.exists() and is_readable_image(img_path):
                samples.append((img_path, boxes))
                camera_frames += 1
        print(f"{camera}: {camera_frames} usable frames")
    return samples


def split_samples(all_samples, max_samples, val_fraction=0.15):
    """Cap, then split into train/validation. Same logic the notebook used inline."""
    if max_samples is not None:
        all_samples = all_samples[:max_samples]

    n_val = max(1, int(val_fraction * len(all_samples)))
    if len(all_samples) < 2:
        raise RuntimeError("At least two readable labelled frames are required for train/validation split")
    val_samples = all_samples[:n_val]
    train_samples = all_samples[n_val:]
    if not train_samples or not val_samples:
        raise RuntimeError("Train and validation splits must both contain at least one frame")
    return train_samples, val_samples
