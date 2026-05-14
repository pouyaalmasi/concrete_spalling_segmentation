# Concrete Spalling Segmentation using U-Net

A deep learning framework for pixel-level concrete spalling segmentation using a U-Net model trained on image-mask pairs.

This repository contains:
- U-Net training code
- Model evaluation pipeline
- Validation/test overlay generation
- Training logs and performance summaries
- Best trained model output

---

# Overview

This repository presents a U-Net-based semantic segmentation framework for detecting and segmenting concrete spalling regions from images.

The model performs binary segmentation:

- Background
- Spalling

The framework supports:
- Training from image-mask datasets
- Validation threshold tuning
- Patch-based test evaluation
- Visual overlay generation
- Export of model checkpoints, logs, and plots

---

# Key Features

## Spalling Segmentation
- Pixel-level concrete spalling detection
- U-Net architecture
- Binary semantic segmentation
- Threshold optimization using validation data

## Model Training
- GPU-supported PyTorch training
- Training and validation logging
- Best model checkpoint saving
- Performance plots

## Evaluation
- Validation metrics
- Patch-based test metrics
- IoU, Dice, Precision, Recall, F1, and Pixel Accuracy
- Overlay visualization for validation and test images

---

# Model Performance

The model was trained and evaluated using CUDA-enabled GPU acceleration. The best validation threshold selected from the validation sweep was:

```text
0.22
```

## Validation Performance at Best Threshold

| Metric | Value |
|---|---:|
| IoU | 0.6867 |
| Dice | 0.8143 |
| Pixel Accuracy | 0.9371 |
| Precision | 0.7393 |
| Recall | 0.9062 |
| F1 Score | 0.8143 |

## Test Performance at Best Threshold

| Metric | Value |
|---|---:|
| IoU | 0.6431 |
| Dice | 0.7828 |
| Pixel Accuracy | 0.9087 |
| Precision | 0.7024 |
| Recall | 0.8840 |
| F1 Score | 0.7828 |

These results are based on the saved training summary.  [oai_citation:0‡run_summary.json](sediment://file_000000005df071fdbb85d3973198af87)

---

# Repository Structure

```text
├── code/
│   ├── train_unet_spalling.py
│
├── dataset_samples/
│   ├── images/
│   ├── masks/
│
├── sample_results/
├── LICENSE
└── README.md
```

---

# Download Trained Model

Download the trained model through GitHub Releases.

Download:

```text
unet_spalling_meta_best.pth
```

from the latest release and place it in:

```text
results_meta_unet/models/unet_spalling_meta_best.pth
```

---

# Requirements

## Software

- Python 3.10 or later
- PyTorch
- NumPy
- OpenCV
- Matplotlib
- scikit-learn
- tqdm
- Albumentations

## Hardware

A CUDA-capable NVIDIA GPU is recommended for training.

The reported experiment was run using:

```text
device: cuda
```

---

# Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Dataset Format

Prepare your dataset using the following structure:

```text
data/
├── train/
│   ├── images/
│   └── masks/
│
├── val/
│   ├── images/
│   └── masks/
│
└── test/
    ├── images/
    └── masks/
```

Mask requirements:
- Binary masks
- Background = 0
- Spalling = 1 or 255
- Image and mask filenames should match

Example:

```text
data/train/images/image_001.jpg
data/train/masks/image_001.png
```

---

# Training

Run:

```bash
python code/train_unet_spalling.py
```

The training script saves outputs to:

```text
results_meta_unet/
```

Generated outputs include:
- Best model checkpoint
- Training logs
- PSO/metaheuristic logs if enabled
- Configuration file
- Training plots
- Validation overlays
- Test overlays

---

# Output Files

The main output paths are:

```text
results_meta_unet/models/unet_spalling_meta_best.pth
results_meta_unet/logs/training_log.csv
results_meta_unet/logs/pso_log.csv
results_meta_unet/configs/meta_config.json
results_meta_unet/plots/
results_meta_unet/overlays_val/
results_meta_unet/overlays_test/
```

---

# Inference

Place test images in:

```text
sample_images/
```

Run:

```bash
python code/inference_spalling.py
```

Outputs will be saved in:

```text
sample_results/
```

Generated outputs:
- Binary spalling mask
- Spalling overlay
- Optional probability map

---

# Example Inference Script

```python
import torch
import cv2
import numpy as np
from pathlib import Path

# User settings
model_path = Path("results_meta_unet/models/unet_spalling_meta_best.pth")
image_path = Path("sample_images/sample.jpg")
output_dir = Path("sample_results")
threshold = 0.22

output_dir.mkdir(parents=True, exist_ok=True)

# Load image
image = cv2.imread(str(image_path))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image_rgb, (256, 256))

# Normalize
x = image_resized.astype(np.float32) / 255.0
x = np.transpose(x, (2, 0, 1))
x = torch.tensor(x).unsqueeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)

# Load model
# Note: replace UNet() with the same model definition used in training.
model = UNet()
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Predict
with torch.no_grad():
    logits = model(x)
    probs = torch.sigmoid(logits)
    mask = (probs > threshold).float()

mask_np = mask.squeeze().cpu().numpy().astype(np.uint8) * 255
mask_np = cv2.resize(mask_np, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

# Overlay
overlay = image.copy()
overlay[mask_np > 0] = [0, 0, 255]

blended = cv2.addWeighted(image, 0.65, overlay, 0.35, 0)

cv2.imwrite(str(output_dir / "predicted_mask.png"), mask_np)
cv2.imwrite(str(output_dir / "overlay.png"), blended)
```

---

# Important Notes

- The inference script must use the same U-Net architecture definition used during training.
- The recommended threshold based on validation sweep is:

```text
0.22
```

- Thresholds may need recalibration for new datasets.
- Results depend on image quality, lighting, surface texture, and annotation consistency.

---

# Limitations

- Generalization to new concrete surfaces should be validated before deployment.
- Spalling boundaries may be sensitive to shadows, stains, surface roughness, and patch repairs.
- The model performance depends strongly on the quality and consistency of ground-truth masks.
- This repository is intended for research and educational use.

---

# Citation

If you use this repository, please cite the related study or repository.

```bibtex
@misc{almasi2026spalling,
  title={Concrete Spalling Segmentation using U-Net},
  author={Almasi, Pouya},
  year={2026},
  note={GitHub repository}
}
```

---

# Author

Pouya Almasi  
Ph.D. Candidate in Civil Engineering  
New Mexico State University

Research Areas:
- Structural Health Monitoring
- UAV-based Infrastructure Inspection
- Deep Learning
- Computer Vision
- Concrete Damage Detection
- Semantic Segmentation

---

# License

This project is licensed under the MIT License.

---

# Acknowledgment

This repository was developed as part of research on AI-enabled infrastructure inspection and automated concrete damage assessment.
