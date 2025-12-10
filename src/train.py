# model/train.py
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from dataset import SteelDefectDataset
from utils import create_model, save_model

def main():
    # base project directory (one level up from this file)
    base_dir = Path(__file__).resolve().parents[1]

    # ---------------------------
    # Define these directories
    # ---------------------------
    # Primary locations we expect (change if your layout is different)
    img_dir_candidate = base_dir / "data" / "train" / "img"        # earlier you had data/train/img
    mask_dir_candidate = base_dir / "data" / "train_masks"        # earlier generate_masks saved here

    # Fallbacks (common alternatives)
    if not img_dir_candidate.exists():
        img_dir_candidate = base_dir / "data" / "images"
    if not mask_dir_candidate.exists():
        mask_dir_candidate = base_dir / "data" / "masks"

    img_dir = str(img_dir_candidate)
    mask_dir = str(mask_dir_candidate)

    print("Using img_dir:", img_dir)
    print("Using mask_dir:", mask_dir)

    # ---------------------------
    # Hyperparams
    # ---------------------------
    num_classes = 5  # background + 4 defect types
    batch_size = 4
    img_size = (256, 512)  # (H, W)
    epochs = 50  # smoke-test; increase for real training

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # ---------------------------
    # Dataset and loader
    # ---------------------------
    dataset = SteelDefectDataset(img_dir=img_dir, mask_dir=mask_dir, img_size=img_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # ---------------------------
    # Model, loss, optimizer
    # ---------------------------
    model = create_model(num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ---------------------------
    # Training loop (smoke test)
    # ---------------------------
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (imgs, masks) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)             # (B,3,H,W)
            masks = masks.to(device, non_blocking=True)           # (B,H,W) int64

            optimizer.zero_grad()
            outputs = model(imgs)['out']                          # (B,C,H,W)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} Batch {batch_idx} Loss {loss.item():.4f}")

            # quick smoke test: break after few batches
            if batch_idx == 5:
                break

        print(f"Epoch {epoch+1} Summary Loss: {total_loss:.4f}")

    # ---------------------------
    # Save model
    # ---------------------------
    out_dir = base_dir / "outputs" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "defect_fcn_resnet50.pth"
    save_model(model, str(save_path))
    print("Saved model to:", save_path)

if __name__ == "__main__":
    main()
