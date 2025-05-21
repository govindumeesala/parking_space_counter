import os
import glob
import shutil
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# ─── CONFIG ────────────────────────────────────────────────────
# Path to the top-level PKLot folder (contains PUCPR, UFPR, etc.)
DATA_ROOT    = "PKLot\PKLot\PKLot"  
# Where to create datasets/{train,val,test}/{images,labels}
OUTPUT_ROOT  = "./datasets"            
# Train/val/test split ratios
RATIOS       = (0.70, 0.15, 0.15)      
SEED         = 42                      

# ─── GATHER SAMPLES ────────────────────────────────────────────
samples = []
for lot in os.listdir(DATA_ROOT):
    lot_path = os.path.join(DATA_ROOT, lot)
    if not os.path.isdir(lot_path): 
        continue
    for weather in os.listdir(lot_path):
        weather_path = os.path.join(lot_path, weather)
        if not os.path.isdir(weather_path):
            continue
        # dive one more level (date folders)
        for date_folder in os.listdir(weather_path):
            folder = os.path.join(weather_path, date_folder)
            if not os.path.isdir(folder):
                continue
            for img in glob.glob(os.path.join(folder, "*.jpg")):
                xml = img[:-4] + ".xml"
                if os.path.exists(xml):
                    samples.append({
                        "img":     img,
                        "xml":     xml,
                        "lot":     lot,
                        "weather": weather
                    })

if not samples:
    raise RuntimeError(f"No image/XML pairs found under {DATA_ROOT}")

# ─── SPLIT ──────────────────────────────────────────────────────
df = samples
train_and_temp, test = train_test_split(df,
                                        test_size=RATIOS[2],
                                        stratify=[ (s["lot"], s["weather"]) for s in df ],
                                        random_state=SEED)

train, val = train_test_split(train_and_temp,
                              test_size=RATIOS[1] / (RATIOS[0] + RATIOS[1]),
                              stratify=[ (s["lot"], s["weather"]) for s in train_and_temp ],
                              random_state=SEED)

splits = {
    "train": train,
    "val":   val,
    "test":  test
}

# ─── PROCESS & WRITE ────────────────────────────────────────────
for split_name, split_samples in splits.items():
    img_out_dir   = os.path.join(OUTPUT_ROOT, split_name, "images")
    label_out_dir = os.path.join(OUTPUT_ROOT, split_name, "labels")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)

    print(f"[{split_name}] Processing {len(split_samples)} samples...")
    for rec in split_samples:
        src_img = rec["img"]
        src_xml = rec["xml"]

        # Copy image
        dst_img = os.path.join(img_out_dir, os.path.basename(src_img))
        shutil.copy(src_img, dst_img)

        # Parse XML → YOLO label
        try:
            tree = ET.parse(src_xml)
            root = tree.getroot()

            # Image dimensions
            img_cv = cv2.imread(src_img)
            h_img, w_img = img_cv.shape[:2]

            lines = []
            for space in root.findall("space"):
                occ    = int(space.get("occupied", "0"))
                cls_id = 1 if occ == 1 else 0

                rr     = space.find("rotatedRect")
                cx     = float(rr.find("center").get("x"))
                cy     = float(rr.find("center").get("y"))
                w_rect = float(rr.find("size").get("w"))
                h_rect = float(rr.find("size").get("h"))
                angle  = float(rr.find("angle").get("d"))

                # build rotated box → 4 points
                rect = ((cx, cy), (w_rect, h_rect), angle)
                pts  = cv2.boxPoints(rect)  # (4×2) float
                xs, ys = pts[:,0], pts[:,1]

                xmin, xmax = float(xs.min()), float(xs.max())
                ymin, ymax = float(ys.min()), float(ys.max())

                # normalize to YOLO format
                x_center = ((xmin + xmax)/2) / w_img
                y_center = ((ymin + ymax)/2) / h_img
                width    = (xmax - xmin)      / w_img
                height   = (ymax - ymin)      / h_img

                lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            # Write label file
            label_path = os.path.join(label_out_dir, os.path.basename(src_img)[:-4] + ".txt")
            with open(label_path, "w") as f:
                f.writelines(lines)

        except Exception as e:
            print(f" [ERROR] {src_xml}: {e}")

print("Done. Your YOLO dataset is in:", OUTPUT_ROOT)
