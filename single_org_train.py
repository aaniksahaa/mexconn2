import os
import argparse
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from model import MultiHeadUNet



# ---- Dataset ----

class SingleOrganelleSegDataset(Dataset):
    def __init__(self, root_dir, split, organelle="membranes", patch_size=256, stride=128):
        self.raw_dir = os.path.join(root_dir, split, "raw")
        self.mask_dir = os.path.join(root_dir, split, organelle)
        self.filenames = sorted(os.listdir(self.raw_dir))
        self.patch_size = patch_size
        self.stride = stride
        self.patches = []
        for img_idx, fname in enumerate(self.filenames):
            img = Image.open(os.path.join(self.raw_dir, fname))
            w, h = img.size
            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    self.patches.append((img_idx, y, x))
        print(f"[{split}] Total patches: {len(self.patches)}")
        self.img_transform = T.ToTensor()  # Divides by 255 and converts to (1,H,W) float

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_idx, y, x = self.patches[idx]
        fname = self.filenames[img_idx]
        img = Image.open(os.path.join(self.raw_dir, fname)).convert("L")
        img_patch = img.crop((x, y, x + self.patch_size, y + self.patch_size))
        img_patch = self.img_transform(img_patch)  # (1, H, W) in [0,1]

        mask = Image.open(os.path.join(self.mask_dir, fname)).convert("L")
        mask_patch = mask.crop((x, y, x + self.patch_size, y + self.patch_size))
        mask_patch = T.ToTensor()(mask_patch)
        mask_patch = (mask_patch > 0.5).float()  # (1, H, W)
        return img_patch, mask_patch

# ---- Losses ----

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super().__init__()
        self.smooth = smooth
    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        intersect = (preds * targets).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2. * intersect + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE
        return F_loss.mean()

def dice_coef(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersect = (pred * target).sum()
    return (2. * intersect + smooth) / (pred.sum() + target.sum() + smooth)

# ---- Train/Val loops ----

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(loader, desc="Train", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)  # preds: (B, 1, H, W)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Val", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            total_loss += loss.item() * imgs.size(0)
    n = len(loader.dataset)
    return total_loss / n

# ---- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data', help='Dataset root')
    parser.add_argument('--domain', type=str, required=True, help='Dataset folder (e.g. drosophila-vnc)')
    parser.add_argument('--organelle', type=str, required=True, help='Organelle name')
    parser.add_argument('--out_path', type=str, required=True, help='Path to save model')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    domain_path = os.path.join(args.data_root, args.domain)
    print(f"Training on: {domain_path} | Organelle: {args.organelle}")

    train_ds = SingleOrganelleSegDataset(domain_path, "train", organelle=args.organelle, patch_size=args.patch_size, stride=args.stride)
    val_ds = SingleOrganelleSegDataset(domain_path, "val", organelle=args.organelle, patch_size=args.patch_size, stride=args.stride)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = MultiHeadUNet(in_ch=1, base_features=[32, 64, 128, 256], out_ch_per_head=1, num_heads=1).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    dice_loss = DiceLoss()
    focal_loss = FocalLoss(alpha=0.8, gamma=2)
    bce_loss = nn.BCEWithLogitsLoss()

    def combined_loss(pred, target):
        return dice_loss(pred, target) + focal_loss(pred, target) + bce_loss(pred, target)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, combined_loss, optimizer, args.device)
        val_loss = val_epoch(model, val_loader, combined_loss, args.device)
        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        torch.save(model.state_dict(), args.out_path)
    print(f"Training complete.")

if __name__ == '__main__':
    main()





'''

python single_org_train.py --data_root data --domain drosophila-vnc --organelle mitochondria --out_path ./models/drosophila/model_mito.pth
python single_org_train.py --data_root data --domain drosophila-vnc --organelle synapses --out_path ./models/drosophila/model_syn.pth
python single_org_train.py --data_root data --domain drosophila-vnc --organelle membranes --out_path ./models/drosophila/model_mem.pth

python single_org_train.py --data_root data --domain multiclass --organelle mitochondria --out_path ./models/multiclass/model_mito.pth
python single_org_train.py --data_root data --domain multiclass --organelle vesicles --out_path ./models/multiclass/model_ves.pth
python single_org_train.py --data_root data --domain multiclass --organelle membranes --out_path ./models/multiclass/model_mem.pth

python single_org_train.py --data_root data --domain urocell_3 --organelle mitochondria --out_path ./models/urocell_3/model_mito.pth
python single_org_train.py --data_root data --domain urocell_3 --organelle lysosomes --out_path ./models/urocell_3/model_lyso.pth
python single_org_train.py --data_root data --domain urocell_3 --organelle fusiform-vesicles --out_path ./models/urocell_3/model_fusi.pth



'''

