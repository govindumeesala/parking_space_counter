import torch
from ultralytics import YOLO

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

if __name__ == "__main__":
    # 1. Load a pretrained YOLOv8 small model
    model = YOLO('yolov8s.pt')

    # 2. Training configuration
    results = model.train(
        data         = 'data.yaml',   # path to your data.yaml
        epochs       = 100,           # total epochs
        imgsz        = 640,           # input image size
        batch        = 16,            # batch size
        device       = 'cpu',             # GPU index (or 'cpu')
        optimizer    = 'AdamW',       # optimizer
        lr0          = 0.001,         # initial learning rate
        lrf          = 0.1,           # final LR = lr0 * lrf (one‑cycle decay)
        weight_decay = 1e-4,          # L2 regularization
        warmup_epochs= 5,             # warmup period
        warmup_bias_lr=0.1,           # bias warmup LR
        box          = 7.5,           # box loss gain
        cls          = 0.5,           # classification loss gain
        dfl          = 1.5,           # distribution focal loss gain
        patience     = 7,             # early stopping patience
        save_period  = 5,             # checkpoint every N epochs
        augment      = True,          # standard augmentations
        mosaic       = True,          # mosaic augmentation
        mixup        = True,          # mixup augmentation
        cache        = 'disk',          # cache images
        project      = 'runs/train',  # output folder
        name         = 'parking',     # subfolder name
        exist_ok     = True,          # overwrite existing
        verbose      = True           # detailed logging
    )

    # 3. Validate & compute F1
    metrics = model.val(data='data.yaml')
    mp      = metrics.box.mp()       # mean precision
    mr      = metrics.box.mr()       # mean recall
    f1      = (2 * mp * mr / (mp + mr)) if (mp + mr) > 0 else 0

    print(f"\nValidation results:\n"
          f"  mAP50    : {metrics.box.map50():.3f}\n"
          f"  mAP50-95 : {metrics.box.map():.3f}\n"
          f"  Precision: {mp:.3f}\n"
          f"  Recall   : {mr:.3f}\n"
          f"  F1 Score : {f1:.3f}")
