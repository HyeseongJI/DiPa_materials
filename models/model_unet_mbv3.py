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
# Custom Dataset for Image-Mask Pairs
# =========================
class SegmentationDataset(Dataset):
    """
    Dataset that loads image and mask pairs from a text file.
    Each line in the .txt file should contain:
        path/to/image path/to/mask
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
        mask = (self.transform(mask) > 0.5).float()  # Binarize mask
        return image, mask

# =========================
# Lightweight U-Net with MobileNetV3 Encoder
# =========================
class UNetMobileNetV3(nn.Module):
    """
    U-Net with MobileNetV3-Large as encoder.
    A lightweight architecture for image segmentation.
    """
    def __init__(self):
        super().__init__()
        base = mobilenet_v3_large(weights=None).features
        self.encoder = base
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(960, 256, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2), nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# =========================
# Training Function
# =========================
def train(loader, model, criterion, optimizer, device):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0
    for images, masks in tqdm(loader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# =========================
# Evaluation Function
# =========================
def evaluate(loader, model, device):
    """
    Evaluate the model using Dice, IoU, Precision, and Recall.
    """
    model.eval()
    dice_scores, iou_scores, precision_scores, recall_scores = [], [], [], []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            preds = (model(images) > 0.5).float()

            intersection = (preds * masks).sum(dim=(1, 2, 3))
            union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))

            dice = (2 * intersection + 1e-8) / (union + 1e-8)
            iou = intersection / (union - intersection + 1e-8)
            precision = intersection / (preds.sum(dim=(1, 2, 3)) + 1e-8)
            recall = intersection / (masks.sum(dim=(1, 2, 3)) + 1e-8)

            dice_scores.extend(dice.cpu().numpy())
            iou_scores.extend(iou.cpu().numpy())
            precision_scores.extend(precision.cpu().numpy())
            recall_scores.extend(recall.cpu().numpy())

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
    BATCH_SIZE = 192
    train_loader = DataLoader(SegmentationDataset(train_txt), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SegmentationDataset(val_txt), batch_size=BATCH_SIZE)

    # Initialize model, loss, optimizer
    model = UNetMobileNetV3().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop with early stopping
    best_dice = 0
    epochs_no_improve = 0
    early_stop_patience = 5
    metrics_log = []
    start_time = time.time()

    for epoch in range(1, 50):
        print(f"\nEpoch {epoch}")
        loss = train(train_loader, model, criterion, optimizer, device)
        dice, iou, precision, recall = evaluate(val_loader, model, device)

        metrics_log.append({
            "Epoch": epoch, "Loss": loss,
            "Dice": dice, "mIoU": iou,
            "Precision": precision, "Recall": recall
        })

        print(f"[Val] Dice: {dice:.4f}, mIoU: {iou:.4f}")

        if dice > best_dice:
            best_dice = dice
            epochs_no_improve = 0
            torch.save(model.state_dict(), results_dir / "best_model.pth")
            print(f"✅ [Best] Model saved at Epoch {epoch} (Dice: {best_dice:.4f})")
        else:
            epochs_no_improve += 1
            print(f"⏳ EarlyStop: {epochs_no_improve}/{early_stop_patience}")

        if epochs_no_improve >= early_stop_patience:
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
# Main Entry Point
# =========================
if __name__ == "__main__":
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Estimate FLOPs for model
    model_tmp = UNetMobileNetV3().to(device)
    with torch.cuda.device(0):
        macs, _ = get_model_complexity_info(model_tmp, (3, 512, 512), as_strings=False, print_per_layer_stat=False)
    flops = macs / 1e9  # Convert to GFLOPs
    print(f"📊 FLOPs: {flops:.2f}G")

    formats = ["png"]  # Extend to ["tiff", "tiff_lzw"] as needed
    summary = []

    for fmt in formats:
        dice_list, time_list, eff_list = [], [], []
        for fold in range(3):  # 3-fold cross-validation
            dice, hours, eff = run_for_format(fmt, fold, device, flops)
            dice_list.append(dice)
            time_list.append(hours)
            eff_list.append(eff)

        avg_dice = np.mean(dice_list)
        avg_time = np.mean(time_list)
        avg_eff = np.mean(eff_list)

        summary.append({
            "Format": fmt,
            "Avg Dice": avg_dice,
            "Avg Time (h)": avg_time,
            "Avg Efficiency": avg_eff
        })

        print(f"✅ [Summary] {fmt} - Dice: {avg_dice:.4f}, Time: {avg_time:.2f}h, Efficiency: {avg_eff:.6e}")

    # Save summary across folds
    pd.DataFrame(summary).to_csv(Path("./results") / "results_summary.csv", index=False)
