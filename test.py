import os
import argparse
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, recall_score, jaccard_score
import numpy as np
from scipy.stats import entropy
from model import MultiHeadUNet
from tqdm import tqdm


# For organelles, keep the one to train and comment out the rest two

ORGANELLES = ("fusiform-vesicles", "mitochondria", "lysosomes")    # for urocell
ORGANELLES = ("membranes", "mitochondria", "synapses")             # for drosophila
ORGANELLES =  ("membranes", "mitochondria", "vesicles")            # for multiclass epfl

# ------------------ Dataset ------------------
class MultiOrganelleSegDataset(Dataset):
    def __init__(self, root_dir, split, organelles,
                 patch_size=256, stride=128):
        self.raw_dir = os.path.join(root_dir, split, "raw")
        self.mask_dirs = {org: os.path.join(root_dir, split, org) for org in organelles}
        self.organelles = organelles
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
        self.img_transform = T.ToTensor()

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_idx, y, x = self.patches[idx]
        fname = self.filenames[img_idx]
        img = Image.open(os.path.join(self.raw_dir, fname)).convert("L")
        img_patch = img.crop((x, y, x + self.patch_size, y + self.patch_size))
        img_patch = self.img_transform(img_patch)

        masks = []
        for org in self.organelles:
            mask = Image.open(os.path.join(self.mask_dirs[org], fname)).convert("L")
            mask_patch = mask.crop((x, y, x + self.patch_size, y + self.patch_size))
            mask_patch = T.ToTensor()(mask_patch)
            mask_patch = (mask_patch > 0.5).float()
            masks.append(mask_patch)
        masks = torch.cat(masks, dim=0)  # (3, H, W)
        return img_patch, masks



# ------------------ Metrics ------------------

def dice_coef(pred, target, smooth=1.):
    pred = (pred > 0.5).float()
    intersect = (pred * target).sum()
    return (2. * intersect + smooth) / (pred.sum() + target.sum() + smooth)

def variation_of_information(y_true, y_pred, eps=1e-10):
    y_true = y_true.flatten().astype(np.int32)
    y_pred = y_pred.flatten().astype(np.int32)
    contingency = np.histogram2d(y_true, y_pred, bins=(2, 2))[0]
    Pxy = contingency / contingency.sum()
    Px = Pxy.sum(axis=1)
    Py = Pxy.sum(axis=0)
    Hx = entropy(Px + eps)
    Hy = entropy(Py + eps)
    Ixy = np.nansum(Pxy * np.log((Pxy + eps) / ((Px[:, None] * Py[None, :]) + eps)))
    return Hx + Hy - 2 * Ixy



# ------------------ Evaluation ------------------

@torch.no_grad()
def evaluate(model, loader, device, num_classes=3):
    print("Device : " , device)
    model.eval()
    dices = [[] for _ in range(num_classes)]
    ious = [[] for _ in range(num_classes)]
    f1s = [[] for _ in range(num_classes)]
    recalls = [[] for _ in range(num_classes)]
    vis = [[] for _ in range(num_classes)]

    for imgs, masks in tqdm(loader, desc="Test", leave=False):
        imgs = imgs.to(device)
        masks = masks.to(device)
        preds = model(imgs)
        preds = torch.sigmoid(preds)
        preds_bin = (preds > 0.5).float()
        # For each class (organelle)
        for c in range(num_classes):
            for p, m in zip(preds_bin[:, c].cpu(), masks[:, c].cpu()):
                p_np = p.squeeze().numpy()
                m_np = m.squeeze().numpy()
                dices[c].append(dice_coef(torch.tensor(p_np), torch.tensor(m_np)).item())
                ious[c].append(jaccard_score(m_np.flatten(), p_np.flatten(), zero_division=1))
                f1s[c].append(f1_score(m_np.flatten(), p_np.flatten(), zero_division=1))
                recalls[c].append(recall_score(m_np.flatten(), p_np.flatten(), zero_division=1))
                vis[c].append(variation_of_information(m_np, p_np))

    results = {}


    for c, name in enumerate(ORGANELLES):

        results[f"{name}_dice"] = np.mean(dices[c])
        results[f"{name}_iou"] = np.mean(ious[c])
        results[f"{name}_f1"] = np.mean(f1s[c])
        results[f"{name}_recall"] = np.mean(recalls[c])
        results[f"{name}_vi"] = np.mean(vis[c])


    results["avg_dice"] = np.mean([np.mean(d) for d in dices])
    results["avg_iou"] = np.mean([np.mean(i) for i in ious])
    results["avg_f1"] = np.mean([np.mean(f) for f in f1s])
    results["avg_recall"] = np.mean([np.mean(r) for r in recalls])
    results["avg_vi"] = np.mean([np.mean(v) for v in vis])
    return results



# ------------------ Main ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data', help='Dataset root')
    parser.add_argument('--domain', required=True, help='Domain name (e.g. drosophila-vnc)')
    parser.add_argument('--model_path', required=True, help='Path to trained .pth model')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    domain_path = os.path.join(args.data_root, args.domain)
    organelles = ORGANELLES
    test_ds = MultiOrganelleSegDataset(domain_path, "test", organelles, patch_size=256, stride=128)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    model = MultiHeadUNet(in_ch=1, base_features=[32, 64, 128, 256], out_ch_per_head=1, num_heads=3).to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    metrics = evaluate(model, test_loader, args.device, num_classes=3)

    print(f"\nDomain: {args.domain}")
    print(f"Avg Dice:   {metrics['avg_dice']:.4f}")
    print(f"Avg IoU:    {metrics['avg_iou']:.4f}")
    print(f"Avg F1:     {metrics['avg_f1']:.4f}")
    print(f"Avg Recall:    {metrics['avg_recall']:.4f}")
    print(f"Avg VI:     {metrics['avg_vi']:.4f}\n")

    for n in organelles:
        print(f"{n.capitalize()} Results:")
        print(f"  Dice:   {metrics[f'{n}_dice']:.4f}")
        print(f"  IoU:    {metrics[f'{n}_iou']:.4f}")
        print(f"  F1:     {metrics[f'{n}_f1']:.4f}")
        print(f"  Recall:    {metrics[f'{n}_recall']:.4f}")
        print(f"  VI:     {metrics[f'{n}_vi']:.4f}\n")



if __name__ == '__main__':
    main()



'''

python test.py --data_root data --domain drosophila-vnc --model_path ./models/drosophila/model.pth --batch_size 8 --device cuda

python test.py --data_root data --domain urocell_3  --model_path ./models/urocell_3/model.pth --batch_size 8 --device cuda

python test.py --data_root data --domain multiclass  --model_path ./models/multiclass/model.pth --batch_size 8 --device cuda




'''
