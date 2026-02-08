# Evaluating JPEG2000 Compression Effects on Lung Cancer WSI Segmentation

This repository contains the **model architecture definitions** described in the manuscript: **"Evaluating JPEG2000 compression effects on prediction accuracy and storage efficiency in lung cancer whole-slide images: A deep learning–based analysis"**.

## 📌 Project Overview
We investigate the trade-off between storage efficiency and AI diagnostic accuracy by applying JPEG2000 compression to Whole Slide Images (WSIs). We compare the robustness of CNN-based models and Transformer-based models against compression artifacts.

### Supported Architectures
The `models.py` file includes the definitions for the following models:
- **CNN-based:**
  - `MobileNetV3-Large + U-Net` (Proposed, efficient)
  - `ResNet-18 + U-Net`
  - `ResNet-50 + U-Net`
- **Transformer-based:**
  - `SegFormer (MiT-B1) + U-Net`

## 📨 Request for Training Code
For academic integrity and reproducibility, the **full training scripts and detailed experimental setups** are available upon reasonable request.

If you are interested in reproducing the results or using the training pipeline, please contact the corresponding author:

📧 **Contact:** hyeseongji0827@gmail.com

Please include your **name and affiliation** in the email.

## 🚀 Usage (Model Inference)
You can initialize the models using the provided `get_model` function:

```python
import torch
from models import get_model

# Example: Initialize MobileNetV3 U-Net
model = get_model("mobilenet", classes=1)
print(model)