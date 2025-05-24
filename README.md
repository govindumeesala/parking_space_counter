# Parking Slot Detection and Occupancy Classification System

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Pre-trained Models](#pre-trained-models)
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


## Pre-trained Models
All three models (YOLOv8, ResNet Model A, and ResNet Model B) can be downloaded from [this Google Drive link](https://drive.google.com/drive/folders/1NiUcyLTjmBCH5NyZxVDz-BMt2oKCJrlX?usp=drive_link).

- **YOLOv8 Model**: Trained on PKLot dataset (99.42% mAP50)
- **ResNet Model A**: Standard ResNet-18 (97.82% accuracy)
- **ResNet Model B**: Enhanced ResNet-18 with augmentations (99.02% accuracy)


## Project Structure
```
pklot/
├── README.md
├── requirements.txt
├── src/
│   ├── models/                          # Model architectures and weights
│   ├── preprocessing/                   # Image preprocessing utilities
│   ├── train_validate_test/             # Training and validation scripts
│   ├── visualization/                   # Training metrics and model comparison plots
│   │   ├── plot_training_metrics.py     # Script to generate performance plots
│   │   ├── model_comparison.png         # Comparison plots of Model A and B
│   │   ├── model_b_metrics.png          # Detailed metrics for Model B
│   │   └── final_comparison.png         # Final performance comparison visualization
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
- Architecture: ResNet-18 pretrained on ImageNet
- Training Configuration:
  ```python
  # Basic preprocessing with standard transforms
  transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
  ])
  ```
- Training Parameters:
  - Planned for 15 epochs (stopped at epoch 5 due to overfitting)
  - Batch size: 32
  - Learning rate: 1e-4
  - Optimizer: Adam
  - Loss Function: CrossEntropyLoss
- No data augmentation

#### Training History
| Epoch | Train Loss | Val Acc |    Note    |
| :---: | :--------: | :-----: | :--------: |
|   1   |   0.0321   | 94.30%  | Best saved |
|   2   |   0.0012   | 98.57%  | Best saved |
|   3   |   0.0007   | 93.75%  | Overfitting starts |
|   4   |   0.0008   | 91.56%  | Performance drops |
|   5   |   0.0016   | 95.18%  | Training stopped |

Training Observations:
- Best validation accuracy achieved at epoch 2 (98.57%)
- Clear signs of overfitting after epoch 2:
  - Training loss continues to decrease (0.0012 → 0.0007)
  - Validation accuracy drops significantly (98.57% → 93.75%)
- Training terminated early at epoch 5 to prevent further overfitting
- Final model uses weights from epoch 2 checkpoint

Final Performance:
- Test Accuracy: 97.82%
- F1-Score: 0.98
- Confusion Matrix: [[438, 20], [0, 458]]

### Model B: Enhanced ResNet with Augmentations
- Architecture: ResNet-18 pretrained on ImageNet
- Advanced preprocessing with extensive data augmentations:
  ```python
  transforms.Compose([
      transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
      transforms.RandomHorizontalFlip(),
      transforms.ColorJitter(brightness=0.2, contrast=0.2),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
  ])
  ```
- Training Parameters:
  - Early stopping (patience=3)
  - Batch size: 32
  - Initial Learning rate: 1e-4
  - Optimizer: Adam
  - Loss Function: CrossEntropyLoss

#### Training History
| Epoch | Train Loss | Val Loss |   Val Acc  |
| :---: | :--------: | :------: | :--------: |
|   1   |   0.0800   |  0.0350  |   93.00%   |
|   2   |   0.0405   |  0.0208  |   97.10%   |
|   3   |   0.0252   |  0.0153  |   98.25%   |
|   4   |   0.0198   |  0.0175  |   98.02%   |
|   5   |   0.0173   |  0.0168  |   98.18%   |
|   6   |   0.0149   |  0.0159  |   98.50%   |
|   7   |   0.0125   |  0.0172  |   98.42%   |
|   8   |   0.0111   |  0.0180  |   98.30%   |
|   9   |   0.0098   |  0.0195  |   98.20%   |
|   10  |   0.0087   |  0.0207  |   98.10%   |
|   11  |   0.0079   |  0.0165  | **99.02%** |

Training Observations:
- Best checkpoint achieved at epoch 11 (val_acc = 99.02%)
- Early stopping triggered after 3 epochs without improvement
- Training loss showed consistent decrease throughout training
- Validation loss stabilized around epochs 2-3, with final improvement at epoch 11
- Data augmentation and learning rate scheduling helped recover from minor overfitting

Final Performance:
- Test Accuracy: 99.02%
- F1-Score: 0.99
- Confusion Matrix: [[449, 9], [0, 458]]

### Key Improvements in Model B
1. Data Augmentation:
   - RandomResizedCrop with scale variation (0.8-1.0)
   - Random horizontal flips
   - Color jitter for brightness and contrast
2. Early Stopping: Prevented overfitting by monitoring validation performance
3. Training Dynamics:
   - Initial rapid improvement (93% → 98.25% in first 3 epochs)
   - Stable performance with slight fluctuations
   - Final convergence to best performance at epoch 11

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
