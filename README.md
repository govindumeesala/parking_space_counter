# Parking Slot Detection and Occupancy Classification System

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Workflow](#workflow)
- [Model Architectures and Performance](#model-architectures-and-performance)
- [Usage](#usage)
- [Model Training Details](#model-training-details)
- [YOLO Experiments](#yolo-experiments)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview
This project implements an automated parking slots identification and tracking model that detects and monitors the occupancy status of individual parking spots using overhead camera images. The system uses deep learning approaches to solve the following tasks:

1. Preprocessing of parking area images
2. Parking Slot Detection using mask images
3. Occupancy Status Classification using ResNet
4. Output Visualization and Statistics Generation

## Dataset
The dataset can be accessed from [this Google Drive link](https://drive.google.com/drive/folders/1CjEFWihRqTLNUnYRwHXxGAVwSXF2k8QC). It contains parking lot images with annotated parking slots and their occupancy status.
**For YOLO-based experiments, see the [Kaggle PKLot dataset](https://www.kaggle.com/datasets/blanderbuss/parking-lot-dataset).**


## Project Structure
```
pklot/
├── README.md
├── requirements.txt
├── src/
│   ├── models/                          # Model architectures and weights
│   ├── preprocessing/                   # Image preprocessing utilities
│   ├── train_validate_test/             # Training and validation scripts
│   └── detect_classify_count_slots_for_given_img/  # Main inference pipeline
│       ├── input/                       # Input images and masks
│       ├── output/                      # Generated visualizations and results
│       ├── utils/                       # Utility functions
│       └── slot_classifier.py           # Main classification script
├── dataset/                             # Training and testing datasets
├── slots/                               # Generated slot information
├── yolo_experiments/                    # YOLO model experiments (see yolo_experiments/README.md)
└── venv/                                # Python virtual environment
```

## YOLO Experiments

YOLOv8-based detection experiments are included in the `yolo_experiments/` folder.  
See [`yolo_experiments/README.md`](yolo_experiments/README.md) for setup, training, and best metrics (mAP50: 0.995, Precision: 0.994, etc.).

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd pklot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Workflow

### 1. Slot Detection Process
1. Input Requirements:
   - Original parking lot image
   - Corresponding mask image for slot detection
2. Process:
   - System processes the mask image to identify individual parking slots
   - Generates `slots.json` containing slot coordinates and information

### 2. Classification Process
1. Using the generated `slots.json` and original input image:
   - Each slot is extracted and classified using the trained model
   - Results are compiled into a comprehensive analysis

### 3. Output Generation
The system produces:
   - Annotated image showing slot status (occupied/empty)
   - CSV file with statistics:
     - Total number of slots
     - Number of occupied slots
     - Number of available slots

## Model Architectures and Performance

We implemented and compared two ResNet-based approaches for parking slot classification:

### Model A: Standard ResNet Training
- Basic preprocessing with standard image resizing and normalization
- Fixed 5-epoch training schedule
- No data augmentation
- Performance metrics:
  - Test Accuracy: 97.82%
  - F1-Score: 0.98
  - Confusion Matrix: [[438, 20], [0, 458]]

### Model B: Enhanced ResNet with Augmentations
- Advanced preprocessing with extensive data augmentations:
  - Random horizontal flips
  - Color jitter (brightness/contrast)
  - Random erasing
- Early stopping implementation
- Learning rate scheduling
- Performance metrics:
  - Test Accuracy: 99.02%
  - F1-Score: 0.99
  - Confusion Matrix: [[449, 9], [0, 458]]

### Key Improvements in Model B
1. Data Augmentation: Enhanced model robustness through various image transformations
2. Early Stopping: Prevented overfitting by monitoring validation performance
3. Learning Rate Scheduling: Optimized training convergence

## Usage

1. Training the Model:
```bash
python src/train_validate_test/train.py --model [model_a/model_b] --data_dir dataset/processed
```

2. Running Inference on a Single Image:
```bash
python src/detect_classify_count_slots_for_given_img/slot_classifier.py '
    --input_image path/to/image.jpg '
    --mask_image path/to/mask.jpg '
    --model_path path/to/saved_model.pth
```

## Model Training Details

### Preprocessing Pipeline
- Image resizing to 224×224 pixels
- Normalization using ImageNet parameters:
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]

### Training Configuration
- Optimizer: Adam with initial learning rate 0.001
- Loss Function: Cross-entropy loss
- Batch Size: 32
- Early Stopping Patience: 2 epochs (Model B only)

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments
- Dataset providers
- PyTorch team for the deep learning framework
