#!/usr/bin/env python3
"""
single_org_train.py — Training, validation, and testing pipeline for
single-organelle segmentation (one organelle per run).

A separate model is trained and saved for each organelle.

Usage examples:
  # Full run (train → per-epoch val → test):
  python single_org_train.py --domain drosophila-vnc --organelle mitochondria

  # Test only:
  python single_org_train.py --domain drosophila-vnc --organelle mitochondria test

  # Train only:
  python single_org_train.py --domain drosophila-vnc --organelle mitochondria train

  # Override output dir:
  python single_org_train.py --domain drosophila-vnc --organelle mitochondria --out_dir my_models/mito
"""

import os
import sys
import random
import argparse

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from tqdm import tqdm
from sklearn.metrics import jaccard_score, f1_score, recall_score, precision_score

from model import MultiHeadUNet


# ── ANSI colors ───────────────────────────────────────────────────────────────

class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"

def cprint(msg, color=C.WHITE, bold=False):
    prefix = (C.BOLD if bold else "") + color
    print(f"{prefix}{msg}{C.RESET}", flush=True)

def banner(msg):
    line = "─" * 64
    cprint(f"\n{line}", C.CYAN, bold=True)
    cprint(f"  {msg}", C.CYAN, bold=True)
    cprint(f"{line}", C.CYAN, bold=True)


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init(worker_id, seed):
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


# ── Dataset ───────────────────────────────────────────────────────────────────

class SingleOrganelleDataset(Dataset):
    """
    Patch-based dataset for single-organelle binary segmentation.

    Directory layout expected:
        <domain_path>/<split>/raw/         <- grayscale EM images
        <domain_path>/<split>/<organelle>/ <- binary masks
    """

    def __init__(self, domain_path: str, split: str, organelle: str,
                 patch_size: int = 256, stride: int = 128):
        self.raw_dir  = os.path.join(domain_path, split, "raw")
        self.mask_dir = os.path.join(domain_path, split, organelle)
        self.organelle  = organelle
        self.patch_size = patch_size
        self.stride     = stride
        self.to_tensor  = T.ToTensor()

        self.filenames = sorted(os.listdir(self.raw_dir))
        self.patches   = []
        for img_idx, fname in enumerate(self.filenames):
            img = Image.open(os.path.join(self.raw_dir, fname))
            w, h = img.size
            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    self.patches.append((img_idx, y, x))

        cprint(f"    [{split:>5}]  {len(self.filenames)} images → {len(self.patches)} patches", C.WHITE)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_idx, y, x = self.patches[idx]
        fname = self.filenames[img_idx]
        ps = self.patch_size

        img  = Image.open(os.path.join(self.raw_dir, fname)).convert("L")
        img_t = self.to_tensor(img.crop((x, y, x + ps, y + ps)))  # (1, H, W) in [0, 1]

        mask = Image.open(os.path.join(self.mask_dir, fname)).convert("L")
        mask_t = self.to_tensor(mask.crop((x, y, x + ps, y + ps)))
        mask_t = (mask_t > 0.5).float()  # (1, H, W)

        return img_t, mask_t


# ── Losses ────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        inter = (preds * targets).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        return 1 - ((2. * inter + self.smooth) / (union + self.smooth)).mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt  = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


# ── Per-patch metrics ─────────────────────────────────────────────────────────

def dice_coef_np(pred_bin: np.ndarray, target: np.ndarray, smooth=1.) -> float:
    inter = (pred_bin * target).sum()
    return float((2. * inter + smooth) / (pred_bin.sum() + target.sum() + smooth))


def variation_of_information(seg_a: np.ndarray, seg_b: np.ndarray, eps=1e-8) -> float:
    from sklearn.metrics import mutual_info_score
    a, b = seg_a.flatten(), seg_b.flatten()
    H_a = H_b = 0.
    for v in (0, 1):
        p = np.mean(a == v)
        if p > eps: H_a -= p * np.log2(p)
        p = np.mean(b == v)
        if p > eps: H_b -= p * np.log2(p)
    return H_a + H_b - 2 * mutual_info_score(a, b)


METRICS = ('dice', 'iou', 'precision', 'recall', 'f1', 'voi')


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, dice_fn, focal_fn, bce_fn, device):
    model.train()
    total_loss = 0.
    for imgs, masks in tqdm(loader, desc="    train", leave=False, ncols=80):
        imgs, masks = imgs.to(device), masks.to(device)
        pred = model(imgs)
        loss = dice_fn(pred, masks) + focal_fn(pred, masks) + bce_fn(pred, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


# ── Validation loop ───────────────────────────────────────────────────────────

@torch.no_grad()
def val_one_epoch(model, loader, dice_fn, focal_fn, bce_fn, device):
    """Returns (mean_loss, mean_dice)."""
    model.eval()
    total_loss = 0.
    dice_vals  = []

    for imgs, masks in tqdm(loader, desc="    val  ", leave=False, ncols=80):
        imgs, masks = imgs.to(device), masks.to(device)
        pred = model(imgs)
        loss = dice_fn(pred, masks) + focal_fn(pred, masks) + bce_fn(pred, masks)
        total_loss += loss.item() * imgs.size(0)

        pred_bin = (torch.sigmoid(pred) > 0.5).float()
        for i in range(imgs.size(0)):
            p = pred_bin[i, 0].cpu().numpy()
            t = masks[i, 0].cpu().numpy()
            dice_vals.append(dice_coef_np(p, t))

    return total_loss / len(loader.dataset), float(np.mean(dice_vals))


# ── Test loop ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def test_evaluate(model, loader, device):
    """Returns (mean_metrics, std_metrics) dicts keyed by metric name."""
    model.eval()
    accum = {m: [] for m in METRICS}

    for imgs, masks in tqdm(loader, desc="    test ", leave=False, ncols=80):
        imgs, masks = imgs.to(device), masks.to(device)
        pred_bin = (torch.sigmoid(model(imgs)) > 0.5).float()

        for i in range(imgs.size(0)):
            p  = pred_bin[i, 0].cpu().numpy()
            t  = masks[i, 0].cpu().numpy()
            pf, tf = p.flatten(), t.flatten()

            accum['dice'].append(dice_coef_np(p, t))
            accum['iou'].append(jaccard_score(tf, pf, zero_division=1.0))
            accum['precision'].append(precision_score(tf, pf, zero_division=1.0))
            accum['recall'].append(recall_score(tf, pf, zero_division=1.0))
            accum['f1'].append(f1_score(tf, pf, zero_division=1.0))
            accum['voi'].append(variation_of_information(t, p))

    mean_m = {k: float(np.mean(v)) for k, v in accum.items()}
    std_m  = {k: float(np.std(v))  for k, v in accum.items()}
    return mean_m, std_m


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Single-organelle segmentation: train → per-epoch val → test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Mode (positional, optional)
    p.add_argument('mode', nargs='?', default='all', choices=['all', 'train', 'test'],
                   help="'all' = train+test (default), 'train' = train only, 'test' = test only")
    # Data
    p.add_argument('--data_root',   default='data',  help='Root of the data/ directory')
    p.add_argument('--domain',      required=True,   help='Dataset sub-folder, e.g. drosophila-vnc')
    p.add_argument('--organelle',   required=True,   help='Organelle name, e.g. mitochondria')
    p.add_argument('--out_dir',     default=None,
                   help='Override models dir (default: models/single/<domain>/seed_<seed>/<organelle>)')
    # Patching
    p.add_argument('--patch_size',  type=int, default=256)
    p.add_argument('--stride',      type=int, default=128)
    # Training hyper-params
    p.add_argument('--batch_size',  type=int,   default=8)
    p.add_argument('--epochs',      type=int,   default=30)
    p.add_argument('--lr',          type=float, default=1e-5)
    p.add_argument('--seed',        type=int,   default=42,
                   help='Global random seed for full reproducibility')
    p.add_argument('--num_workers', type=int,   default=4)
    p.add_argument('--device',      default='cuda' if torch.cuda.is_available() else 'cpu')
    # Model architecture
    p.add_argument('--base_features', nargs='+', type=int, default=[32, 64, 128, 256],
                   help='Encoder feature map sizes at each level')
    # Optional explicit model path
    p.add_argument('--model_path',  default=None,
                   help='Explicit .pth to load/save (default: <out_dir>/model_best.pth)')
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    domain_path = os.path.join(args.data_root, args.domain)
    if not os.path.isdir(domain_path):
        cprint(f"[ERROR] Domain path not found: {domain_path}", C.RED, bold=True)
        sys.exit(1)

    # ── Auto-derive output directories ──
    seed_tag    = f"seed_{args.seed}"
    models_dir  = args.out_dir or os.path.join("models",  "single", args.domain, seed_tag, args.organelle)
    results_dir = os.path.join("results", "single", args.domain, seed_tag)
    os.makedirs(models_dir,  exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    do_train = args.mode in ('all', 'train')
    do_test  = args.mode in ('all', 'test')

    banner(f"Domain: {args.domain}  |  Mode: {args.mode}  |  Organelle: {args.organelle}")
    cprint(f"  Seed: {args.seed}  |  Device: {args.device}  |  Epochs: {args.epochs}", C.WHITE)
    cprint(f"  Features: {args.base_features}", C.WHITE)
    cprint(f"  Models dir:  {models_dir}", C.WHITE)
    cprint(f"  Results dir: {results_dir}", C.WHITE)

    # ── Resolve model paths ──
    best_model_path = args.model_path or os.path.join(models_dir, "model_best.pth")
    last_model_path = os.path.join(models_dir, "model_last.pth")

    # ── Build model (single head for one organelle) ──
    model = MultiHeadUNet(
        in_ch=1,
        base_features=args.base_features,
        out_ch_per_head=1,
        num_heads=1,
    ).to(args.device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cprint(f"  Trainable parameters: {total_params:,}", C.WHITE)

    # ── Loss functions ──
    dice_fn  = DiceLoss()
    focal_fn = FocalLoss(alpha=0.8, gamma=2)
    bce_fn   = nn.BCEWithLogitsLoss()

    # ─────────────────────────────── TRAINING ────────────────────────────────
    if do_train:
        banner(f"Training  [{args.organelle}]")

        train_ds = SingleOrganelleDataset(domain_path, "train", args.organelle, args.patch_size, args.stride)
        val_ds   = SingleOrganelleDataset(domain_path, "val",   args.organelle, args.patch_size, args.stride)

        g = torch.Generator()
        g.manual_seed(args.seed)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
            generator=g,
            worker_init_fn=lambda wid: _worker_init(wid, args.seed),
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        best_val_loss = float('inf')

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, dice_fn, focal_fn, bce_fn, args.device)
            val_loss, val_dice = val_one_epoch(model, val_loader, dice_fn, focal_fn, bce_fn, args.device)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                tag = f"  {C.GREEN}{C.BOLD}✓ best{C.RESET}"
            else:
                tag = ""

            print(
                f"{C.BOLD}{C.BLUE}[Epoch {epoch:>3}/{args.epochs}]{C.RESET}  "
                f"{C.YELLOW}train={train_loss:.4f}  val={val_loss:.4f}{C.RESET}  "
                f"dice={C.CYAN}{val_dice:.4f}{C.RESET}"
                f"{tag}",
                flush=True,
            )

        torch.save(model.state_dict(), last_model_path)
        cprint(f"\n  Best model  → {best_model_path}", C.GREEN, bold=True)
        cprint(f"  Last model  → {last_model_path}", C.WHITE)

    # ──────────────────────────────── TEST ───────────────────────────────────
    if do_test:
        banner(f"Test Evaluation  [{args.organelle}]")

        if not os.path.exists(best_model_path):
            cprint(f"  [ERROR] Model checkpoint not found: {best_model_path}", C.RED, bold=True)
            cprint("  Tip: run without 'test' mode first, or supply --model_path.", C.YELLOW)
            sys.exit(1)

        model.load_state_dict(torch.load(best_model_path, map_location=args.device))
        cprint(f"  Loaded checkpoint: {best_model_path}", C.WHITE)

        test_ds = SingleOrganelleDataset(domain_path, "test", args.organelle, args.patch_size, args.stride)
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

        mean_m, std_m = test_evaluate(model, test_loader, args.device)

        print()
        cprint(f"  {args.organelle}", C.YELLOW, bold=True)
        for metric in METRICS:
            mu, sd = mean_m[metric], std_m[metric]
            cprint(f"    {metric:>10}: {mu:.4f} ± {sd:.4f}", C.WHITE)

        csv_path = os.path.join(results_dir, "test_results.csv")
        row = {
            'domain':       args.domain,
            'organelle':    args.organelle,
            'model':        best_model_path,
            'seed':         args.seed,
            'test_patches': len(test_ds),
            **{f'{m}_mean': mean_m[m] for m in METRICS},
            **{f'{m}_std':  std_m[m]  for m in METRICS},
        }
        df = pd.DataFrame([row])
        if os.path.exists(csv_path):
            df = pd.concat([pd.read_csv(csv_path), df], ignore_index=True)
        df.to_csv(csv_path, index=False)
        cprint(f"\n  Results saved → {csv_path}", C.GREEN, bold=True)

    cprint("\nDone.\n", C.GREEN, bold=True)


if __name__ == '__main__':
    main()
