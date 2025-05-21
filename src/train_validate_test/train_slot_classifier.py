#!/usr/bin/env python3
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def main():
    # ─── CONFIG ──────────────────────────────────────────────────────
    DATA_DIR       = "slots"            # slots/train & slots/val
    BATCH_SIZE     = 32
    IMG_SIZE       = 224
    EPOCHS         = 15
    LR             = 1e-4
    DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_WEIGHTS = "slot_classifier_resnet18.pt"

    # ─── TRANSFORMS ─────────────────────────────────────────────────
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])

    # ─── DATASETS & DATALOADERS ─────────────────────────────────────
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # ─── MODEL SETUP ─────────────────────────────────────────────────
    # use weights parameter instead of deprecated pretrained
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    all_labels, all_preds = [], []

    # ─── TRAIN & VALIDATE ───────────────────────────────────────────
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        avg_loss = total_loss / len(train_ds)

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                logits = model(imgs)
                preds = logits.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())

        val_acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_WEIGHTS)
            print(f"  → New best accuracy, saved to {OUTPUT_WEIGHTS}")

    print("\nBest Validation Accuracy:", best_val_acc)
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_ds.classes))

if __name__ == "__main__":
    # for Windows multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()
    from torchvision import transforms  # delayed import
    main()
