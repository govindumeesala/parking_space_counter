#!/usr/bin/env python3
import os
import time
import torch
import random
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ─── CONFIG ──────────────────────────────────────────────────────
DATA_DIR       = "slots"           # slots/train & slots/val
BATCH_SIZE     = 32
IMG_SIZE       = 224
EPOCHS         = 30
LR             = 1e-4
PATIENCE       = 3                # early stopping patience
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_WEIGHTS = "slot_classifier_resnet18_es.pt"
SEED           = 42

# ─── REPRODUCIBILITY ─────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if DEVICE.startswith("cuda"):
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ─── TRANSFORMS ─────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])

def main():
    # ─── DATASETS & DATALOADERS ─────────────────────────────────
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # ─── MODEL SETUP ────────────────────────────────────────────
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=3
    )

    best_val_acc = 0.0
    wait = 0

    for epoch in range(1, EPOCHS+1):
        start = time.time()

        # — TRAINING —
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_ds)

        # — VALIDATION —
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss    = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().tolist())
        val_loss = val_loss / len(val_ds)
        val_acc  = accuracy_score(all_labels, all_preds)

        # — SCHEDULER & EARLY STOPPING —
        scheduler.step(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
            torch.save(model.state_dict(), OUTPUT_WEIGHTS)
            improved = "*"
        else:
            wait += 1
            improved = ""
        elapsed = time.time() - start

        # — LOGGING —
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:02d}/{EPOCHS:02d} | "
              f"Time: {elapsed:.1f}s | "
              f"LR: {current_lr:.1e} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} {improved}")

        if wait >= PATIENCE:
            print(f"No improvement in {PATIENCE} epochs, stopping early.")
            break

    # ─── FINAL EVALUATION ────────────────────────────────────────
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_ds.classes))

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    from torchvision import transforms  # ensure transforms is imported under __main__
    main()
