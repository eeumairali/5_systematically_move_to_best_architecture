"""CLI entry point for the n10 joint detection + re-ID pipeline.

Run from the repo root with:
    python -m n10_funs.main --epochs 25 --lr 1e-4
"""
import argparse
import random

from torch.utils.data import DataLoader

from n10_funs import config
from n10_funs.checkpoint import save_checkpoint
from n10_funs.dataset import MmCowsJointDataset, collate_joint
from n10_funs.loading_related_funs import load_frame_index, split_samples
from n10_funs.model import JointDetReIDModel, count_params, show_shapes
from n10_funs.train import evaluate_detection, evaluate_identity_head, fit


def parse_args():
    parser = argparse.ArgumentParser(description="Train the n10 joint detection + re-ID model on mmCows.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-drop-epoch", type=int, default=15)
    parser.add_argument("--batch", type=int, default=config.BATCH)
    parser.add_argument("--max-samples", type=int, default=config.MAX_SAMPLES)
    return parser.parse_args()


def main():
    args = parse_args()
    config.check_dataset()

    all_samples = load_frame_index()
    if not all_samples:
        raise RuntimeError(f"No readable labelled {config.DATASET_NAME} frames found under {config.DATA_ROOT}")
    random.shuffle(all_samples)
    print(f"total samples: {len(all_samples)}")

    train_samples, val_samples = split_samples(all_samples, args.max_samples)
    print(f"{len(train_samples)} train / {len(val_samples)} val")

    train_loader = DataLoader(MmCowsJointDataset(train_samples), batch_size=args.batch, shuffle=True,
                               collate_fn=collate_joint, num_workers=0)
    val_loader = DataLoader(MmCowsJointDataset(val_samples), batch_size=args.batch, shuffle=False,
                             collate_fn=collate_joint, num_workers=0)

    model = JointDetReIDModel().to(config.DEVICE)
    print("trainable parameters:", f"{count_params(model):,}")
    show_shapes(model)

    history = fit(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr,
                  lr_drop_epoch=args.lr_drop_epoch)

    precision, recall, tp, fp, fn = evaluate_detection(model, val_loader)
    id_acc = evaluate_identity_head(model, val_loader)
    print(f"Classification+Detection (IoU>=0.5): precision={precision:.2%}  recall={recall:.2%}  "
          f"(tp={tp}, fp={fp}, fn={fn})")
    print(f"Identity accuracy (closed-set, {config.N_IDENTITIES} known cows): {id_acc:.1f}%  "
          f"(random guess: {100 / config.N_IDENTITIES:.1f}%)")

    save_checkpoint(model, history)


if __name__ == "__main__":
    main()
