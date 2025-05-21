#!/usr/bin/env python3
import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ─── CONFIG ──────────────────────────────────────────────────────
DATA_DIR       = "slots"           # contains train/ val/ test/ subfolders
TEST_SUBFOLDER = "test"
BATCH_SIZE     = 32
IMG_SIZE       = 224
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_WEIGHTS  = "slot_classifier_resnet18.pt"  # path to your saved weights

# ─── TRANSFORMS ─────────────────────────────────────────────────
test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])

def main():
    # 1) Prepare test dataset & loader
    test_path = os.path.join(DATA_DIR, TEST_SUBFOLDER)
    test_ds   = datasets.ImageFolder(test_path, transform=test_tf)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2) Load model architecture and weights
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, len(test_ds.classes))
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model.to(DEVICE).eval()

    # 3) Run inference & gather predictions
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    # 4) Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    cm  = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=test_ds.classes)

    # 5) Print results
    print(f"\nTest Accuracy : {acc:.4f}")
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print(cm)
    print("\nClassification Report:")
    print(report)

if __name__ == "__main__":
    main()
