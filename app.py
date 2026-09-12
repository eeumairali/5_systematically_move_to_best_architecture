"""Command-line entry point for training the Webots JDE detector/ReID model."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_load import WebotsJDEDataset, jde_collate, read_classes
from jde import JDEModel, fit


def build_parser():
    parser = argparse.ArgumentParser(description="Train a CenterNet/FairMOT-inspired JDE model.")
    parser.add_argument("--data-root", type=Path, default=Path("benchmarks/webots_mcid-Small500"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--resume", nargs="?", const=True, default=None, help="Resume from output-dir/latest.pt, or provide a checkpoint path.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/jde_webots"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main(args=None):
    config = build_parser().parse_args(args)
    if not config.data_root.exists():
        raise FileNotFoundError(f"Dataset does not exist: {config.data_root}")
    device = torch.device(config.device)
    class_names = read_classes(config.data_root)
    n_identities = max(len(class_names), 1)
    image_size = (config.width, config.height)
    train_dataset = WebotsJDEDataset(config.data_root, "train", image_size, config.stride, n_identities)
    val_split = "val" if (config.data_root / "val.txt").exists() else "test"
    val_dataset = WebotsJDEDataset(config.data_root, val_split, image_size, config.stride, n_identities)
    loader_kwargs = {"batch_size": config.batch_size, "num_workers": config.workers, "collate_fn": jde_collate}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    model = JDEModel(n_identities=n_identities).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Dataset: {config.data_root} | train={len(train_dataset)} | val={len(val_dataset)}")
    print(f"Identities: {n_identities} | parameters: {parameters:,} | device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    fit(model, train_loader, val_loader, device, config.epochs, config.learning_rate, config.output_dir, patience=config.patience, resume=config.resume)


if __name__ == "__main__":
    main()
