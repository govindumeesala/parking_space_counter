from ultralytics import YOLO
import glob, os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import cv2

# Point to the best checkpoint
model = YOLO('runs/train/parking/weights/best.pt')

# 1. Validate and save TXT predictions for the “test” split
metrics = model.val(
    data='data.yaml',
    split='test',
    conf=0.3,
    iou=0.65,
    batch=16,
    save_txt=True,   # write YOLO-format .txt preds into runs/val
    project='runs/val', 
    name='test_preds',
    exist_ok=True
)

# 2. Print the usual detection metrics
print(f"mAP50:    {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")
print(f"Precision: {metrics.box.mp:.3f}")
print(f"Recall:    {metrics.box.mr:.3f}")

# You can also access confusion matrix and per-class metrics
if hasattr(metrics, "confusion_matrix"):
    print("\n🧩 Confusion Matrix:")
    print(metrics.confusion_matrix.matrix)


# 3. Load ground‑truth and predicted labels and compute confusion matrix
def yolo_to_xyxy(box, img_w, img_h):
    xc, yc, w, h = box
    x1 = (xc - w/2) * img_w
    y1 = (yc - h/2) * img_h
    x2 = (xc + w/2) * img_w
    y2 = (yc + h/2) * img_h
    return [x1, y1, x2, y2]

def iou(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = (a[2]-a[0])*(a[3]-a[1])
    areaB = (b[2]-b[0])*(b[3]-b[1])
    return inter / (areaA + areaB - inter + 1e-6)

y_true, y_pred = [], []

# Paths to your test split labels + the val‑predictions
gt_dir  = "datasets/test/labels"
pd_dir  = "runs/val/test_preds/labels"
img_dir = "datasets/test/images"

for gt_path in glob.glob(os.path.join(gt_dir, "*.txt")):
    fname   = os.path.basename(gt_path)
    pd_path = os.path.join(pd_dir, fname)
    img_path= os.path.join(img_dir, fname.replace(".txt", ".jpg"))
    img     = cv2.imread(img_path)
    h, w    = img.shape[:2]

    # load GT
    gt_boxes = []
    for l in open(gt_path).read().splitlines():
        cls, xc,yc,bbw,bbh = l.split()
        gt_boxes.append((int(cls), [float(xc),float(yc),float(bbw),float(bbh)]))

    # load preds
    pd_boxes = []
    if os.path.exists(pd_path):
        for l in open(pd_path).read().splitlines():
            cls, xc,yc,bbw,bbh = l.split()
            pd_boxes.append((int(cls), [float(xc),float(yc),float(bbw),float(bbh)]))

    used = set()
    for true_cls, box in gt_boxes:
        gt_xy = yolo_to_xyxy(box, w, h)
        best_iou, best_j = 0, None
        for j,(pd_cls,pd_box) in enumerate(pd_boxes):
            if j in used: continue
            pd_xy = yolo_to_xyxy(pd_box, w, h)
            val   = iou(gt_xy, pd_xy)
            if val>best_iou:
                best_iou, best_j = val, j

        if best_iou>=0.5:
            pred_cls = pd_boxes[best_j][0]
            used.add(best_j)
        else:
            pred_cls = 0

        y_true.append(true_cls)
        y_pred.append(pred_cls)

# 4. Compute and print accuracy + confusion matrix
acc = accuracy_score(y_true, y_pred)
cm  = confusion_matrix(y_true, y_pred)
print(f"\nSlot‑level Accuracy: {acc:.4f}\n")
print("Confusion Matrix (rows=actual, cols=pred):")
print(cm)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['empty','occupied']))
