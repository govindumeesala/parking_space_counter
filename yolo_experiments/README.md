# YOLOv8 Parking Slot Detection Experiments

This folder contains all scripts, configs, and results for training, validating, and testing a YOLOv8 model for parking slot detection on the PKLot dataset.

---

## Dataset

- **PKLot Dataset:**  
  Download from [Kaggle - Parking Lot Dataset](https://www.kaggle.com/datasets/blanderbuss/parking-lot-dataset)

---

## Installation & Setup

1. **Create a new virtual environment (recommended):**
   ```sh
   python -m venv venv_yolo
   ```
   Activate it:
   - **Windows:**  
     `venv_yolo\Scripts\activate`
   - **Linux/macOS:**  
     `source venv_yolo/bin/activate`

2. **Install requirements:**
   ```sh
   pip install -r requirements.txt
   ```

---

## Data Preparation

- The dataset is organized and converted to YOLO format using  
  [`src/preprocessing/preparing_dataset_for_yolo.py`](src/preprocessing/preparing_dataset_for_yolo.py).
- This script:
  - Recursively scans the PKLot dataset folders for image/XML pairs.
  - Splits data into train/val/test (default: 70/15/15) with stratification.
  - Converts rotated-rect XML annotations to YOLO `[class x_center y_center width height]` format.
  - Writes images and labels to `datasets/{train,val,test}/{images,labels}`.

---

## Training & Validation

- Training is performed with [`src/train_validate/train_validate.py`](src/train_validate/train_validate.py).
- Key settings (see [`runs/train/args.yaml`](runs/train/args.yaml)):
  - Model: YOLOv8s, optimizer: AdamW, batch size: 16, epochs: 100, image size: 640.
  - Augmentations: mosaic, mixup, color jitter, etc.
  - Output: checkpoints and logs in `runs/train/parking/`.

- Validation:
  - After training, validation metrics are computed on the test split using  
    [`src/train_validate/validate.py`](src/train_validate/validate.py).
  - This script:
    - Runs inference on test images.
    - Saves YOLO-format predictions to `runs/val/test_preds/labels`.
    - Computes mAP, precision, recall, confusion matrix, and slot-level accuracy.

---

## Metrics & Results

- **Best Training Metrics (from `runs/train/results.csv`):**
  - Example (best epoch):
    - `mAP50`: **0.995**
    - `mAP50-95`: **0.994**
    - `Precision`: **0.994**
    - `Recall`: **0.994**
    - (See [`runs/train/results.csv`](runs/train/results.csv) for full details.)

- **Validation/Test predictions:**  
  - Annotated images: `runs/detect/test_pred/images/`
  - YOLO-format prediction labels: `runs/detect/test_pred/labels/` or `runs/val/test_preds/labels/`
- **Slot-level evaluation:**  
  - Confusion matrix and classification report printed by `validate.py`.

---

## Inference

- Run detection on a single image:  
  [`src/predict/predict_one_img.py`](src/predict/predict_one_img.py)
- Run detection on all test images:  
  [`src/test/test.py`](src/test/test.py)

---

## Visualization

- Visualize random ground-truth labels:  
  [`src/preprocessing/visvualize_random_img_from_labels.py`](src/preprocessing/visvualize_random_img_from_labels.py)
- Visualize sample predictions:  
  [`src/test/visualize_sample_pred.py`](src/test/visualize_sample_pred.py)

---

## Notes

- All output folders (`runs/train`, `runs/detect`, `runs/val`) are auto-created.
- For reproducibility, seeds and deterministic settings are used.
- For custom data, adjust paths in `data.yaml` and rerun preprocessing.

---