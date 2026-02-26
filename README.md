# 🌍 Satellite Imagery Semantic Segmentation (Dubai Dataset)

## 📌 Project Overview

This project implements **multi-class semantic segmentation** on high-resolution satellite imagery from the **Dubai Dataset** using a custom **U-Net deep learning architecture**.

The objective is to classify every pixel in a satellite image into one of six land-cover categories:

- 🏢 Building  
- 🟣 Land  
- 🛣 Road  
- 🌿 Vegetation  
- 🌊 Water  
- ⬜ Unlabeled  

The full pipeline includes:

- Data exploration  
- Image preprocessing  
- Patch extraction  
- RGB mask → class label conversion  
- One-hot encoding  
- U-Net model implementation  
- Dice + Focal loss optimization  
- Model training and evaluation  
- Prediction visualization  
- Model saving  

---

# 📂 Dataset

**Dataset Name:** DubaiDataset  

### Directory Structure

```
DubaiDataset/
 ├── Tile 1/
 │   ├── images/
 │   ├── masks/
 ├── Tile 2/
 ├── ...
 ├── Tile 8/
 └── classes.json
```

Each tile contains:

- RGB satellite images (`.jpg`)
- Corresponding segmentation masks (`.png`)

---

# 🖼 Data Preprocessing

## 1️⃣ Image Cropping

Images are cropped so dimensions are divisible by the patch size:

```
Patch Size = 256 × 256
```

This ensures clean patch extraction without padding.

---

## 2️⃣ Patch Extraction

Large satellite images are divided into smaller patches:

- Patch size: **256 × 256**
- Step size: **256**

After patching:

- **945 image patches**
- **945 mask patches**

---

## 3️⃣ Image Normalization

Images are scaled to `[0, 1]` range:

```python
image = image.astype(np.float32) / 255.0
```

---

# 🎨 Label Encoding

Masks are RGB images where each color corresponds to a class.

### RGB → Class Mapping

| Class        | RGB Value         | Label |
|-------------|------------------|--------|
| Building    | [60, 16, 152]    | 0 |
| Land        | [132, 41, 246]   | 1 |
| Road        | [110, 193, 228]  | 2 |
| Vegetation  | [254, 221, 58]   | 3 |
| Water       | [226, 169, 41]   | 4 |
| Unlabeled   | [155, 155, 155]  | 5 |

Conversion Pipeline:

```
RGB Mask → Integer Label → One-Hot Encoding
```

Final label shape:

```
(945, 256, 256, 6)
```

---

# 📊 Dataset Split

Train-test split:

- **85% Training**
- **15% Testing**

### Final Shapes

```
X_train: (803, 256, 256, 3)
X_test:  (142, 256, 256, 3)

y_train: (803, 256, 256, 6)
y_test:  (142, 256, 256, 6)
```

---

# 🧠 Model Architecture

## Multi-Class U-Net

A custom U-Net architecture was implemented with:

- Encoder depth: 4 levels  
- Dropout rate: 0.2  
- Activation: ReLU  
- Output activation: Softmax  
- Total parameters: ~1.94 Million  

### Architecture Flow

```
Input (256x256x3)
   ↓
Conv → Conv → MaxPool
   ↓
Conv → Conv → MaxPool
   ↓
Conv → Conv → MaxPool
   ↓
Conv → Conv → MaxPool
   ↓
Bottleneck (256 filters)
   ↓
Upsampling + Skip Connections
   ↓
Final Conv (1x1) → Softmax (6 classes)
```

---

# 📉 Loss Function

To improve segmentation accuracy and handle class imbalance:

### 🔹 Dice Loss  
Improves overlap between predicted and true masks.

### 🔹 Categorical Focal Loss  
Focuses on difficult-to-classify pixels.

### 🔹 Final Loss Function

```
Total Loss = Dice Loss + Focal Loss
```

---

# 📈 Metrics

Model performance is evaluated using:

- Accuracy  
- Jaccard Coefficient (IoU)  

---

# 🚀 Training Configuration

```
Batch size: 16
Epochs: 10+
Optimizer: Adam
Learning rate: 1e-4 (recommended)
Shuffle: True
```

Early stopping is recommended to prevent overfitting.

---

# 📊 Training Visualization

Custom callback plots:

- Training vs Validation Loss
- Training vs Validation IoU

This enables real-time performance monitoring.

---

# 🔍 Prediction & Evaluation

After training:

```python
y_pred = model.predict(X_test)
y_pred_argmax = np.argmax(y_pred, axis=3)
```

Visualization includes:

- Original Image
- Ground Truth Mask
- Predicted Mask

---

# 💾 Model Saving

The trained model is saved as:

```
satellite_segmentation_full.h5
```

This allows:

- Future inference
- Deployment
- Fine-tuning

---

# 🏁 Complete Pipeline Summary

✔ Dataset exploration  
✔ Patch extraction  
✔ RGB → Label conversion  
✔ One-hot encoding  
✔ Train-test split  
✔ U-Net implementation  
✔ Dice + Focal loss  
✔ IoU metric  
✔ Training visualization  
✔ Prediction comparison  
✔ Model saving  

---

# 🔮 Future Improvements

- Data augmentation  
- Pretrained backbone (ResNet34 / EfficientNet)  
- Mean IoU per class  
- Class imbalance optimization  
- tf.data pipeline optimization  
- Model checkpointing  
- Deployment as API  

---

# 🛠 Requirements

```
patchify
numpy
opencv-python
Pillow
matplotlib
scikit-learn
tensorflow
segmentation-models
```

Install dependencies:

```
pip install patchify segmentation-models tensorflow opencv-python
```

---

# 📌 Project Type

Deep Learning | Computer Vision | Semantic Segmentation | Remote Sensing

---

# 👤 Author

Satellite Imagery Segmentation using Deep Learning  
U-Net Architecture | Multi-Class Segmentation | Dubai Dataset

---
