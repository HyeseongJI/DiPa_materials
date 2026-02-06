import argparse
import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp
from ptflops import get_model_complexity_info
from PIL import Image

# Import the model selector
from models import get_model 

# =========================
# Argument Parser
# =========================
def get_args():
    parser = argparse.ArgumentParser(description="Train WSI Segmentation Model (JPEG2000 Study)")
    
    # 1. Paths (User Required)
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory containing images')
    parser.add_argument('--split_dir', type=str, required=True, help='Directory containing K-Fold split .txt files')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save checkpoints and logs')
    parser.add_argument('--weight_path', type=str, default=None, help='Path to local pre-trained weights .pth (optional)')

    # 2. Model Selection
    parser.add_argument('--model', type=str, default='mobilenet', 
                        choices=['mobilenet', 'resnet18', 'resnet50', 'segformer'],
                        help='Model architecture to train')
    
    # 3. Hyperparameters (Default: Matches Manuscript)
    parser.add_argument('--batch_size', type=int, default=192, help='Batch size (Default: 192 as per paper)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    
    return parser.parse_args()

# =========================
# Hybrid Loss (Dice + BCE)
# =========================
class HybridLoss(nn.Module):
    def __init__(self):
        super(HybridLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.dice = smp.losses.DiceLoss(mode="binary")

    def forward(self, preds, targets):
        return self.bce(preds, targets) + self.dice(preds, targets)

# =========================
# Dataset Class
# =========================
class SegmentationDataset(Dataset):
    def __init__(self, txt_file, data_dir):
        self.data_dir = Path(data_dir)
        with open(txt_file, 'r') as f:
            # Lines format: "image_path mask_path"
            self.pairs = [line.strip().split() for line in f.readlines()]
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_rel_path, mask_rel_path = self.pairs[idx]
        
        # Handle paths (Join with data_dir if relative)
        if not os.path.isabs(img_rel_path):
            img_path = self.data_dir / img_rel_path
            mask_path = self.data_dir / mask_rel_path
        else:
            img_path = Path(img_rel_path)
            mask_path = Path(mask_rel_path)

        # Load Image (RGB) and Mask (Grayscale)
        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image or mask not found: {img_path}")
        
        image = self.transform(image)
        mask = (self.transform(mask) > 0.5).float()
        return image, mask

# =========================
# Training & Evaluation Loop
# =========================
def train_epoch(loader, model, criterion, optimizer, device):
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

def evaluate(loader, model, device):
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            preds = (model(images) > 0.5).float()
            
            intersection = (preds * masks).sum(dim=(1, 2, 3))
            union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
            
            dice = (2 * intersection + 1e-8) / (union + 1e-8)
            dice_scores.extend(dice.cpu().numpy())
    return np.mean(dice_scores)

# =========================
# Main Execution
# =========================
def run_fold(args, fold, device):
    print(f"\n====== [Start] Fold {fold} | Model: {args.model} ======")
    
    # 1. Setup Directories
    splits_dir = Path(args.split_dir)
    train_txt = splits_dir / f"fold{fold}_train.txt"
    val_txt = splits_dir / f"fold{fold}_val.txt"
    
    save_dir = Path(args.save_dir) / args.model / f"fold{fold}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 2. Dataloaders
    train_dataset = SegmentationDataset(train_txt, args.data_dir)
    val_dataset = SegmentationDataset(val_txt, args.data_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # 3. Model Initialization
    model = get_model(args.model, classes=1)
    
    # Load custom local weights if provided
    if args.weight_path:
        print(f"📂 Loading weights from: {args.weight_path}")
        try:
            state_dict = torch.load(args.weight_path, map_location=device)
            model.encoder.load_state_dict(state_dict, strict=False)
            print("✅ Custom encoder weights loaded.")
        except Exception as e:
            print(f"⚠️ Failed to load encoder weights: {e}. Trying full model load...")
            model.load_state_dict(state_dict, strict=False)

    model = model.to(device)

    # 4. Optimizer, Loss, Scheduler
    criterion = HybridLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Scheduler: StepLR as per manuscript
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # 5. Training Loop
    best_dice = 0
    patience_counter = 0
    metrics_log = []
    
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(train_loader, model, criterion, optimizer, device)
        val_dice = evaluate(val_loader, model, device)
        
        # Step Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch} | LR: {current_lr:.1e} | Loss: {loss:.4f} | Val Dice: {val_dice:.4f}")
        
        metrics_log.append({"Epoch": epoch, "Loss": loss, "Dice": val_dice, "LR": current_lr})

        # Early Stopping & Saving
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print(f"✅ Saved Best Model (Dice: {best_dice:.4f})")
        else:
            patience_counter += 1
            print(f"⏳ Early Stop Counter: {patience_counter}/{args.patience}")
        
        if patience_counter >= args.patience:
            print("🛑 Early stopping triggered.")
            break
            
    # Save logs
    pd.DataFrame(metrics_log).to_csv(save_dir / "training_log.csv", index=False)

if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🚀 Device: {device} | Model: {args.model} | Batch Size: {args.batch_size}")

    # Optional: FLOPs Calculation
    try:
        model_tmp = get_model(args.model).to(device)
        macs, _ = get_model_complexity_info(model_tmp, (3, 512, 512), as_strings=False, print_per_layer_stat=False)
        print(f"📊 {args.model} FLOPs: {macs}")
    except Exception:
        pass

    # Run 3-Fold Cross Validation
    for fold in range(3):
        run_fold(args, fold, device)