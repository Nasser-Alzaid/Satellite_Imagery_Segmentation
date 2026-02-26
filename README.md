# Semantic Segmentation of Satellite Imagery

## Overview

This project performs semantic segmentation on satellite imagery to classify land cover types:

- Water
- Land
- Road
- Building
- Vegetation
- Unlabeled

The dataset is processed through patch extraction, preprocessing, label encoding, and train-test splitting to prepare it for deep learning model training.

---

## Dataset

Dataset Name:
Semantic segmentation dataset

Dataset Structure:
- Satellite RGB images
- Corresponding RGB mask images
- Organized into Tiles containing multiple image parts

Expected Dataset Path:

/content/drive/MyDrive/dataset/Semantic segmentation dataset

---

## Installation

Install required dependency:

pip install patchify

Required Python libraries:

- os
- cv2
- PIL
- numpy
- patchify
- sklearn
- matplotlib
- random

---

## Pipeline

### 1. Load Dependencies

Import required libraries for image processing and data handling.

### 2. Define Dataset Path

Set:
- dataset_root_folder
- dataset_name

### 3. Image Patching

Large images are divided into smaller patches:

Patch Size:
256 x 256 pixels

### 4. Data Preprocessing

- Images scaled using MinMaxScaler
- Mask RGB values converted to integer class labels

---

## Label Mapping

RGB to Class Mapping:

Water       [226, 169, 41]  -> 0
Land        [132, 41, 246]  -> 1
Road        [110, 193, 228] -> 2
Building    [60, 16, 152]   -> 3
Vegetation  [254, 221, 58]  -> 4
Unlabeled   [155, 155, 155] -> 5

Number of Classes: 6

---

## One-Hot Encoding

Integer mask labels are converted into one-hot encoded format.

Output shape per mask:
(256, 256, 6)

---

## Train-Test Split

Training: 85%
Testing: 15%

---

## Dataset Summary

Patch Size: 256 x 256
Total Patches: 945 images and 945 masks

Training Data:
X_train shape: (803, 256, 256, 3)
y_train shape: (803, 256, 256, 6)

Testing Data:
X_test shape: (142, 256, 256, 3)
y_test shape: (142, 256, 256, 6)

---

## Status

✔ Data preprocessing complete  
✔ Dataset ready for training  
⬜ Model training  
⬜ Evaluation  

---

## Next Steps

- Train segmentation model (U-Net, DeepLabV3+, FCN, etc.)
- Evaluate using IoU / Dice score
- Visualize predictions
