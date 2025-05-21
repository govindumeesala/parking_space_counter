#!/usr/bin/env python3
import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO

# ─── CONFIG ──────────────────────────────────────────────────────
MODEL_PATH = r"runs\train\parking\weights\best.pt"  # path to your best checkpoint
IMG_PATH = r"test\test_img.png"
CLASS_NAMES = {0: "free", 1: "occupied"}

# ─── FUNCTIONS ───────────────────────────────────────────────────
def draw_boxes(img, boxes, classes, confs):
    """Draws colored boxes with labels & confidence on img."""
    for (x1, y1, x2, y2), cls, conf in zip(boxes, classes, confs):
        color = (0,255,0) if cls == 0 else (0,0,255)
        label = f"{CLASS_NAMES[cls]} {conf:.2f}"
        cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
        # put label above box
        ((w, h), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (int(x1), int(y1)-h-4), (int(x1)+w, int(y1)), color, -1)
        cv2.putText(img, label, (int(x1), int(y1)-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return img

def main():
    if not os.path.isfile(IMG_PATH):
        print(f"File not found: {IMG_PATH}")
        sys.exit(1)

    # 1) Load model
    model = YOLO(MODEL_PATH)

    # 2) Run inference
    results = model.predict(source=IMG_PATH, conf=0.3, save=False)[0]

    # 3) Parse results
    boxes = results.boxes.xyxy.cpu().numpy()       # [N,4]
    classes = results.boxes.cls.cpu().numpy().astype(int)  # [N]
    confs = results.boxes.conf.cpu().numpy()       # [N]

    # 4) Count slots
    total_slots    = len(classes)
    occupied_slots = int((classes == 1).sum())
    free_slots     = total_slots - occupied_slots

    print(f"Results for {os.path.basename(IMG_PATH)}:")
    print(f"  Total slots    : {total_slots}")
    print(f"  Occupied slots : {occupied_slots}")
    print(f"  Free slots     : {free_slots}")

    # 5) Visualize
    img = cv2.imread(IMG_PATH)
    vis = draw_boxes(img.copy(), boxes, classes, confs)

    # 6) Save output visualization
    out_path = os.path.splitext(IMG_PATH)[0] + "_pred.jpg"
    cv2.imwrite(out_path, vis)
    print(f"Visualization saved to {out_path}")

if __name__ == "__main__":
    main()
