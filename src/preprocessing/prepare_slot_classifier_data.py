#!/usr/bin/env python3
import os
import glob
import shutil
import random

# ─── CONFIG ──────────────────────────────────────────────────────
SRC_ROOT    = "dataset"      # contains 'empty/' and 'occupied/'
DST_ROOT    = "slots"        # will contain train/val/test splits
CLASSES     = ["empty", "occupied"]
SPLITS      = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED        = 42

# ─── MAKE DEST FOLDERS ───────────────────────────────────────────
for split in SPLITS:
    for cls in CLASSES:
        os.makedirs(os.path.join(DST_ROOT, split, cls), exist_ok=True)

random.seed(SEED)

# ─── PROCESS EACH CLASS ─────────────────────────────────────────
for cls in CLASSES:
    src_dir = os.path.join(SRC_ROOT, cls)
    imgs    = glob.glob(os.path.join(src_dir, "*.jpg")) + glob.glob(os.path.join(src_dir, "*.png"))
    random.shuffle(imgs)

    n = len(imgs)
    n_train = int(SPLITS["train"] * n)
    n_val   = int(SPLITS["val"]   * n)
    # remainder goes to test
    n_test  = n - n_train - n_val

    splits = {
        "train": imgs[:n_train],
        "val":   imgs[n_train:n_train+n_val],
        "test":  imgs[n_train+n_val:]
    }

    print(f"{cls}: {n_train} train, {n_val} val, {n_test} test  (total {n})")

    for split, files in splits.items():
        dst_dir = os.path.join(DST_ROOT, split, cls)
        for src_path in files:
            fname = os.path.basename(src_path)
            dst_path = os.path.join(dst_dir, fname)
            shutil.copy(src_path, dst_path)

print("Done! Your data is organized under:", DST_ROOT)
