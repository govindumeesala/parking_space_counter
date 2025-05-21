#!/usr/bin/env python3
import cv2
import json
import argparse
import os

def process_mask(mask_path, use_connected_components=False, min_area=500):
    # Load grayscale mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")
    
    # Ensure binary mask
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Morphological erosion to separate close slots
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.erode(binary, kernel, iterations=1)

    boxes = []

    if use_connected_components:
        # Better for touching blobs
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, num_labels):  # skip background
            x, y, w, h, area = stats[i]
            if area >= min_area:
                boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
    else:
        # Classic contours
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w * h >= min_area:
                boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})

    return boxes

def main():
    parser = argparse.ArgumentParser(description="Generate slot bounding boxes from mask image")
    parser.add_argument("--mask", required=True, help="Path to binary parking mask (white=slot)")
    parser.add_argument("--output", default="slots.json", help="Path to output JSON")
    parser.add_argument("--connected", action="store_true", help="Use connected components instead of contours")
    parser.add_argument("--min_area", type=int, default=500, help="Minimum area to filter out noise")
    args = parser.parse_args()

    boxes = process_mask(args.mask, args.connected, args.min_area)

    with open(args.output, "w") as f:
        json.dump(boxes, f, indent=2)

    print(f"Saved {len(boxes)} slots to {args.output}")

if __name__ == "__main__":
    main()
