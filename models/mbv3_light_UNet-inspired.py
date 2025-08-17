# =========================
# Setup
# =========================
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_large
from ptflops import get_model_complexity_info
from PIL import Image

# =========================
# Config (tunable parameters)
# =========================
BATCH_SIZE = 192               # mini-batch size
INIT_LR = 1e-3                 # initial learning rate for Adam
MAX_EPOCHS = 50                # max epochs
EARLY_STOP_PATIENCE = 5        # early-stop patience on val Dice
THRESH = 0.5                   # binarization threshold for evaluation
STEP_SIZE = 10                 # StepLR: decay every 'step_size' epochs
GAMMA = 0.1                    # StepLR: new_lr = lr * gamma
ALPHA_DICE = 0.5               # Hybrid loss weight: total = α*Dice + (1-α)*BCE
NUM_WORKERS = 4                # dataloader workers
PIN_MEMORY = True              # pin memory for faster host->GPU transfer

# =========================
# Custom Dataset for Image-Mask Pairs
# =========================
class SegmentationDataset(Dataset):
    """
    Dataset that loads image and mask pairs from a text file.
    Each line in the .txt file should contain:
        path/to/image path/to/mask
    Images are read as RGB, masks as single-channel (L) and binarized (>0.5 -> 1).
    """
    def __init__(self, txt_file):
        with open(txt_file, 'r') as f:
            self.pairs = [line.strip().split() for line in f.readlines()]
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image = self.transform(image)
        mask = (self.transform(mask) > 0.5).float()  # binarize
        return image, mask

# =========================
# Lightweight Encoder-Decoder (U-Net-inspired)
# =========================
class UNetMobileNetV3(nn.Module):
    """
    As the focus of this study was not to develop a novel segmentation architecture
    but rather to evaluate the impact of compression on prediction performance
    and efficiency, we employed a lightweight encoder–decoder model with a
    MobileNetV3-Large backbone. The design was inspired by U-Net, but long skip
    connections were omitted to simplify the architecture and reduce model complexity.

    NOTE: Structure is intentionally simple and unchanged (Sigmoid kept),
    since we only adjust training strategy/parameters to match the paper.
    """
    def __init__(self):
        super().__init__()
        base = mobilenet_v3_large(weights=None).features
        self.encoder = base
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(960, 256, 2, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 2, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64,  2, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,  32,  2, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32,  16,  2, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1), nn.Sigmoid()  # keep Sigmoid (structure unchanged)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)   # probabilities in [0,1]
        return x

# =========================
# Losses (Hybrid: Dice + BCE) — works with Sigmoid outputs
# =========================
class DiceLossFromProbs(nn.Module):
    """
    Dice loss computed from probabilities (expects preds in [0,1]).
    This matches our current model head that already outputs Sigmoid probabilities.
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, probs, targets):
        targets = targets.float()
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()

class HybridLoss(nn.Module):
    """
    total = alpha * DiceLoss + (1 - alpha) * BCELoss
    Both terms consume probabilities in [0,1] (consistent with model's Sigmoid output).
    """
    def __init__(self, alpha=0.5, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.dice = DiceLossFromProbs(smooth)
        self.bce = nn.BCELoss()

    def forward(self, probs, targets):
        return self.alpha * self.dice(probs, targets) + (1 - self.alpha) * self.bce(probs, targets)

# =========================
# Training Function
# =========================
def train(loader, model, criterion, optimizer, device):
    """
    Train the model for one epoch using the provided criterion (Hybrid Dice+BCE).
    """
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(loader, desc="Training"):
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        preds = model(images)                 # preds are probabilities (Sigmoid)
        loss = criterion(preds, masks)        # Hybrid(Dice+BCE) on probabilities
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)

# =========================
# Evaluation Function
# =========================
@torch.no_grad()
def evaluate(loader, model, device, thresh=THRESH):
    """
    Evaluate with Dice, IoU, Precision, Recall.
    Thresholding is applied on probabilities: (preds > thresh).
    """
    model.eval()
    dice_scores, iou_scores, precision_scores, recall_scores = [], [], [], []
    for images, masks in loader:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        probs = model(images)                       # probabilities
        preds = (probs > thresh).float()            # binarize

        intersection = (preds * masks).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))

        dice = (2 * intersection + 1e-8) / (union + 1e-8)
        iou = intersection / (union - intersection + 1e-8)
        precision = intersection / (preds.sum(dim=(1, 2, 3)) + 1e-8)
        recall = intersection / (masks.sum(dim=(1, 2, 3)) + 1e-8)

        dice_scores.extend(dice.detach().cpu().numpy())
        iou_scores.extend(iou.detach().cpu().numpy())
        precision_scores.extend(precision.detach().cpu().numpy())
        recall_scores.extend(recall.detach().cpu().numpy())

    return np.mean(dice_scores), np.mean(iou_scores), np.mean(precision_scores), np.mean(recall_scores)

# =========================
# Training Wrapper for Each Format/Fold
# =========================
def run_for_format(fmt, fold, device, flops):
    """
    Run training and evaluation for a given format and fold.
    Saves best model, metrics, and efficiency stats.
    """
    print(f"\n====== [Start] {fmt} | Fold {fold} ======")

    # Define input/output directories
    base_dir = Path("./data")
    splits_dir = base_dir / "splits" / "3fold"
    train_txt = splits_dir / f"fold{fold}_train.txt"
    val_txt = splits_dir / f"fold{fold}_val.txt"

    results_dir = Path("./results") / fmt / f"fold{fold}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_loader = DataLoader(
        SegmentationDataset(train_txt),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        SegmentationDataset(val_txt),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    # Initialize model, loss, optimizer, scheduler
    model = UNetMobileNetV3().to(device)
    criterion = HybridLoss(alpha=ALPHA_DICE)                # << Dice+BCE
    optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    # Training loop with early stopping
    best_dice = 0.0
    epochs_no_improve = 0
    metrics_log = []
    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"\nEpoch {epoch} | LR={optimizer.param_groups[0]['lr']:.2e}")
        loss = train(train_loader, model, criterion, optimizer, device)
        dice, iou, precision, recall = evaluate(val_loader, model, device, thresh=THRESH)

        metrics_log.append({
            "Epoch": epoch, "Loss": loss,
            "Dice": dice, "mIoU": iou,
            "Precision": precision, "Recall": recall,
            "LR": optimizer.param_groups[0]['lr']
        })

        print(f"[Val] Dice: {dice:.4f}, mIoU: {iou:.4f} | P {precision:.4f} | R {recall:.4f}")

        if dice > best_dice:
            best_dice = dice
            epochs_no_improve = 0
            torch.save(model.state_dict(), results_dir / "best_model.pth")
            print(f"✅ [Best] Model saved at Epoch {epoch} (Dice: {best_dice:.4f})")
        else:
            epochs_no_improve += 1
            print(f"⏳ EarlyStop: {epochs_no_improve}/{EARLY_STOP_PATIENCE}")

        scheduler.step()

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print("🛑 Early stopping triggered.")
            break

    # Save logs
    pd.DataFrame(metrics_log).to_csv(results_dir / "metrics.csv", index=False)

    # Save efficiency info
    hours = (time.time() - start_time) / 3600
    efficiency = best_dice / (flops * hours + 1e-8)
    with open(results_dir / "efficiency.txt", "w") as f:
        f.write(f"Best Dice: {best_dice:.4f}\n")
        f.write(f"FLOPs: {flops:.2f}G\n")
        f.write(f"Training Time: {hours:.2f}h\n")
        f.write(f"Efficiency Index: {efficiency:.6e}\n")

    return best_dice, hours, efficiency

# =========================
# FLOPs helper
# =========================
def get_flops_safe(model, device, input_size=(3, 512, 512)):
    """
    Get FLOPs (GFLOPs) using ptflops; handles CPU-only environment gracefully.
    """
    try:
        if device.type == "cuda":
            with torch.cuda.device(device.index or 0):
                macs, _ = get_model_complexity_info(model, input_size, as_strings=False, print_per_layer_stat=False)
        else:
            macs, _ = get_model_complexity_info(model, input_size, as_strings=False, print_per_layer_stat=False, backend="pytorch")
        return macs / 1e9
    except Exception as e:
        print(f"⚠️ FLOPs estimation failed: {e}")
        return float("nan")

# =========================
# Main Entry Point
# =========================
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Estimate FLOPs for model
    model_tmp = UNetMobileNetV3().to(device)
    flops = get_flops_safe(model_tmp, device, input_size=(3, 512, 512))
    if not np.isnan(flops):
        print(f"📊 FLOPs: {flops:.2f}G")
    else:
        print("📊 FLOPs: N/A")

    formats = ["png"]  # Extend to ["tiff", "tiff_lzw"] as needed
    summary = []

    for fmt in formats:
        dice_list, time_list, eff_list = [], [], []
        for fold in range(3):  # 3-fold cross-validation
            dice, hours, eff = run_for_format(fmt, fold, device, flops if not np.isnan(flops) else 0.0)
            dice_list.append(dice)
            time_list.append(hours)
            eff_list.append(eff)

        avg_dice = np.mean(dice_list)
        avg_time = np.mean(time_list)
        avg_eff = np.mean(eff_list)

        summary.append({
            "Format": fmt,
            "Avg Dice": float(avg_dice),
            "Avg Time (h)": float(avg_time),
            "Avg Efficiency": float(avg_eff)
        })

        print(f"✅ [Summary] {fmt} - Dice: {avg_dice:.4f}, Time: {avg_time:.2f}h, Efficiency: {avg_eff:.6e}")

    Path("./results").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(Path("./results") / "results_summary.csv", index=False)
