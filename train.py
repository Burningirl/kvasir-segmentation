"""Script entry point for Kvasir-SEG. Implement UNet in src/segmentation.py first."""
from pathlib import Path
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
from src.segmentation import seed_everything, discover_pairs, PolypDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = discover_pairs(args.data_dir)
    idx = np.arange(len(pairs))
    train_idx, temp_idx = train_test_split(idx, test_size=.30, random_state=args.seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=.50, random_state=args.seed)
    print(f"device={device}; n={len(pairs)} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    print("TODO: instantiate UNet, build loaders, train on train_idx, select on val_idx, evaluate test_idx once.")
    print("Save metrics/config/plots under", args.output_dir)

if __name__ == "__main__":
    main()
