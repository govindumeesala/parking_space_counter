#!/usr/bin/env python3
import cv2
import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Draw slot boxes from JSON onto an image")
    parser.add_argument("--image", required=True, help="Path to the parking image (e.g. pk.jpg)")
    parser.add_argument("--json", required=True, help="Path to slots JSON file")
    parser.add_argument("--output", default=None, help="Output path (optional)")
    args = parser.parse_args()

    # Load image
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")

    # Load JSON boxes
    with open(args.json, "r") as f:
        boxes = json.load(f)

    # Draw each box
    for idx, box in enumerate(boxes):
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(img, f"{idx+1}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Output path
    out_path = args.output or os.path.splitext(args.image)[0] + "_with_slots.jpg"
    cv2.imwrite(out_path, img)
    print(f"Saved annotated image to {out_path}")

if __name__ == "__main__":
    main()
