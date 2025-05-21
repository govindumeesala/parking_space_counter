# Automated Parking Slot Identification & Tracking  
**IIT Tirupati Navavishkar I-Hub Foundation**  
**Indian Institute of Technology Tirupati**  
**Yerpedu – Venkatagiri Road, Tirupati District, Andhra Pradesh – 517619, India**  

---

## 1. Introduction

Urbanization has led to a surge in vehicle ownership, making efficient parking management a critical challenge in commercial zones such as malls, hospitals, and business complexes. Manual supervision of parking lots often results in underutilized spaces, long queues, and poor user experience. This project addresses these challenges by developing an automated computer vision system for parking slot identification and occupancy tracking using overhead camera images. The system aims to provide real-time insights into slot availability, streamline parking operations, and enable integration with digital displays and mobile applications for enhanced user experience and data-driven infrastructure planning.

Our approach is two-stage:  
1. **Parking Slot Detection** using YOLOv8 object detection on the PKLot dataset.  
2. **Occupancy Classification** using ResNet-based classifiers on cropped slot images.

---

## 2. Data Overview

### 2.1 PKLot Dataset

- **Source:** [Kaggle PKLot dataset](https://www.kaggle.com/datasets/blanderbuss/parking-lot-dataset)
- **Description:** 12,417 images from three parking lots:
  - **UFPR04 & UFPR05:** Different views (4th & 5th floors) of the same lot at Federal University of Parana (UFPR)
  - **PUCPR:** 10th floor view at Pontifical Catholic University of Parana (PUCPR)
- **Limitation:** Only three camera angles, which impacts model generalization.

### 2.2 Custom Mask-Based Annotation

- For robust slot classification, we generated `slots.json` by annotating parking spaces using binary masks on input images.
- Each slot is defined by its coordinates and status (occupied/empty), enabling precise cropping for classification.

**Figure 1:** Sample parking-lot image  
![Parking Lot](./figures/parking_zone.png)

---

### Flowchart 1: Data Pipeline

```flowchart
st=>start: Load raw images
a=>operation: Preprocess (CLAHE, denoise)
b=>operation: YOLO label generation (XML→YOLO)
c=>operation: Mask→JSON slot bboxes
d=>operation: Crop slots & label CSV
e=>operation: Train ResNet classifiers
f=>end: Output CSV + annotated image
st->a->b->c->d->e->f
```

---

## 3. Methodology

### 3.1 YOLOv8 Slot Detection

- **Preprocessing:** Contrast enhancement (CLAHE), denoising.
- **Label Conversion:** PKLot XML annotations converted to YOLO format.
- **Training Configuration:**  
  - Model: YOLOv8s  
  - Epochs: 100  
  - Optimizer: AdamW  
  - Augmentations: Mosaic, mixup, color jitter  
  - Batch size: 16  
- **Limitation:**  
  - The model struggled to generalize to unseen camera angles due to limited diversity in the PKLot dataset.
  - Good at classifying slot occupancy when the slot is correctly detected, but missed many slots in new views.

**Table 1: YOLOv8 Validation Metrics**

| Metric      | UFPR Val | PUCPR Val |
|-------------|----------|-----------|
| mAP50       | 85.2%    | 78.4%     |
| mAP50–95    | 65.1%    | 58.3%     |

---

### 3.2 ResNet Slot Classification

- **Slot Cropping & Labeling:**  
  - Used custom masks to generate `slots.json` and crop individual slot images.
  - Each slot labeled as occupied or empty.
- **Dataset:**  
  - 3,000 empty + 3,000 occupied slot images.
- **Models:**  
  - **Model A:** Standard ResNet-18, no augmentation.
  - **Model B:** ResNet-18 with strong augmentations and early stopping.

**Training & Validation Comparison**

| Epoch | Model A val_acc | Model B val_acc |
|-------|-----------------|-----------------|
| 1     | 94.30%          | ~95.5%          |
| 2     | 98.57%          | ~98.6%          |
| 3     | 93.75%          | early stopped   |

**Test Performance**

| Model   | Test Acc | False Positives | False Negatives |
|---------|----------|-----------------|-----------------|
| A       | 97.82%   | 20              | 0               |
| B       | 99.02%   | 9               | 0               |

---

## 4. Results & Visualization

- **YOLOv8:**  
  - High precision on known angles, but fails to detect slots in unseen perspectives.
  - Example failure cases visualized in `visualization/yolo_failures/`.
- **ResNet Classifier:**  
  - Robust classification on cropped slots, even with challenging lighting or occlusions.
  - Visualizations of predictions and confusion matrices in `visualization/resnet_results/`.

**Figure 2:** Example output visualization  
![Annotated Output](./visualization/sample_annotated_output.png)

---

## 5. Discussion

- **Why YOLO Struggled:**  
  - Limited camera angles in PKLot dataset led to poor generalization for slot detection.
- **ResNet Success:**  
  - Mask-based slot annotation and cropping enabled high-accuracy classification.
  - Strong augmentations and early stopping prevented overfitting.
- **Trade-offs:**  
  - Two-stage approach (detection + classification) is more robust for this dataset than end-to-end detection.

---

## 6. Conclusion & Future Work

- **Achievements:**  
  - Developed a robust pipeline for parking slot detection and occupancy classification.
  - Achieved up to 99% test accuracy on slot classification.
  - Automated output generation: annotated images and CSV with slot counts.
- **Next Steps:**  
  - Explore homography/perspective correction for better slot localization.
  - Investigate Mask-RCNN for joint detection and classification.
  - Integrate with live video feeds and mobile applications.

---

## 7. References

1. Almeida, P. R., Oliveira, L. S., Silva, E. J., Britto, A. S., & Koerich, A. L. (2015). PKLot–A robust dataset for parking lot classification. Expert Systems with Applications, 42(11), 4937-4949.
2. YOLOv8 Documentation: https://docs.ultralytics.com/
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
4. Short citations for augmentation & early stopping best practices.

---

