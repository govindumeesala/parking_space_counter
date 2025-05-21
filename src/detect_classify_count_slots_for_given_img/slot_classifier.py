#!/usr/bin/env python3
import cv2
import json
import torch
from torchvision import transforms as T
from PIL import Image
import argparse
import os
import sys
import csv

def load_model(weights_path, device):
    model = torch.hub.load('pytorch/vision:v0.14.0', 'resnet18', weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model.to(device).eval()

def get_args():
    p = argparse.ArgumentParser(description="Classify parking slots from JSON bboxes")
    p.add_argument("--image",   required=True, help="Path to input image (e.g. pk.jpg)")
    p.add_argument("--json",    required=True, help="Path to slots.json")
    p.add_argument("--weights", default="slot_classifier_resnet18.pt",
                   help="Path to trained classifier weights")
    p.add_argument("--out",     default=None, help="Output path for annotated image")
    p.add_argument("--csv",     default="results.csv", help="CSV output path")
    return p.parse_args()

def main():
    args   = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_model(args.weights, device)

    # Preprocessing pipeline (must match training)
    IMG_SIZE = 224
    preprocess = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    # Load image
    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Error: cannot read image {args.image}")

    h_img, w_img = img.shape[:2]

    # Load JSON boxes
    with open(args.json, "r") as f:
        slots = json.load(f)

    counts = {"empty":0, "occupied":0}

    # Classify each slot
    for idx, slot in enumerate(slots):
        x, y, w, h = slot["x"], slot["y"], slot["w"], slot["h"]

        # Ensure box is within image bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2]
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        inp = preprocess(pil).unsqueeze(0).to(device)

        with torch.no_grad():
            cls = model(inp).argmax(1).item()  # 0=empty, 1=occupied

        label = "empty" if cls == 0 else "occupied"
        counts[label] += 1

        # Draw box
        color = (0,255,0) if cls == 0 else (0,0,255)
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        text = f"{idx+1}:{label}"
        cv2.putText(img, text, (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Summary
    total    = counts["empty"] + counts["occupied"]
    occupied = counts["occupied"]
    available= counts["empty"]

    print(f"Total slots    : {total}")
    print(f"Occupied slots : {occupied}")
    print(f"Available slots: {available}")

    # Save annotated image
    out_img = args.out or os.path.splitext(args.image)[0] + "_classified.jpg"
    cv2.imwrite(out_img, img)
    print(f"Annotated image saved to {out_img}")

    # Write CSV
    with open(args.csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["TotalNumberOfSlots","OccupiedSlots","AvailableSlots"])
        writer.writerow([total, occupied, available])
    print(f"Results CSV saved to {args.csv}")

if __name__ == "__main__":
    main()
