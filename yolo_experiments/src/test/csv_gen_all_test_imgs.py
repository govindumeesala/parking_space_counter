import glob
import os
import pandas as pd

rows = []
for img_path in glob.glob('datasets/test/images/*.jpg'):
    name      = os.path.basename(img_path)
    # prediction path
    pred_txt  = f"runs/detect/test_pred/labels/{name.replace('.jpg','.txt')}"
    # ground‑truth path
    gt_txt    = f"datasets/test/labels/{name.replace('.jpg','.txt')}"

    # --- read predictions ---
    pred_total, pred_occ = 0, 0
    if os.path.exists(pred_txt):
        for line in open(pred_txt):
            cls = int(line.split()[0])
            pred_total += 1
            pred_occ   += (cls == 1)

    # --- read ground truth ---
    gt_total, gt_occ = 0, 0
    if os.path.exists(gt_txt):
        for line in open(gt_txt):
            cls = int(line.split()[0])
            gt_total += 1
            gt_occ   += (cls == 1)

    rows.append({
        'Image':            name,
        # predictions
        'Pred_Total':       pred_total,
        'Pred_Occupied':    pred_occ,
        'Pred_Available':   pred_total - pred_occ,
        # ground truth
        'GT_Total':         gt_total,
        'GT_Occupied':      gt_occ,
        'GT_Available':     gt_total - gt_occ
    })

df = pd.DataFrame(rows)
df.to_csv('parking_test_report_with_gt.csv', index=False)
print(df.head())
