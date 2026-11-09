"""Minimal Kvasir-SEG scaffolding. Fill in the model TODO before training."""
from pathlib import Path
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def discover_pairs(data_dir):
    """Find matching image/mask files under data_dir/images and data_dir/masks."""
    data_dir = Path(data_dir)
    image_dir, mask_dir = data_dir / "images", data_dir / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            f"Expected {image_dir} and {mask_dir}. Update DATA_DIR or discover_pairs()."
        )
    masks = {p.stem: p for p in mask_dir.iterdir() if p.is_file()}
    pairs = [(p, masks[p.stem]) for p in image_dir.iterdir()
             if p.is_file() and p.stem in masks]
    if not pairs:
        raise RuntimeError("No matching image/mask filenames found.")
    return sorted(pairs)

class PolypDataset(Dataset):
    def __init__(self, pairs, image_size=256):
        self.pairs = list(pairs)
        self.image_size = image_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]
        image = Image.open(image_path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )
        mask = Image.open(mask_path).convert("L").resize(
            (self.image_size, self.image_size), Image.Resampling.NEAREST
        )
        x = np.asarray(image, dtype=np.float32) / 255.0
        y = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)
        x = torch.from_numpy(x).permute(2, 0, 1)
        y = torch.from_numpy(y).unsqueeze(0)
        return x, y

def dice_iou_from_logits(logits, targets, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    pred = probs >= threshold
    target = targets >= 0.5
    dims = (1, 2, 3)
    intersection = (pred & target).sum(dims).float()
    pred_sum = pred.sum(dims).float()
    target_sum = target.sum(dims).float()
    dice = (2 * intersection + eps) / (pred_sum + target_sum + eps)
    union = pred_sum + target_sum - intersection
    iou = (intersection + eps) / (union + eps)
    return dice.mean().item(), iou.mean().item()

# TODO: implement a compact U-Net here, or import your own model.
# Expected interface: model(images) -> logits with shape [B, 1, H, W].
