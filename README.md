# Baseline CNN for Mars Rover Image Classification

**Collaborator:** Donia Mabrouk

---

## Project Overview
This project uses **Convolutional Neural Networks (CNNs)** to classify images captured by **NASA Mars rovers**. The model learns to recognize **terrain, rocks, and other Martian features** from **64×64 grayscale images**. It provides a **lightweight baseline CNN** with a full **training and evaluation workflow**, designed for **reproducible experimentation, future optimization, and embedded AI deployment** on **GPU and FPGA/SoC hardware**.

---

## Baseline Results (Minimal CNN)
- **Test Accuracy:** 90.47%  
- **Model Size:** 1.56 MB  
- **Training Time:** ~15 seconds per epoch  
- **Best Class:** apxs cal target (100%)  
- **Worst Class:** chemcam cal target (33%)  

---

## Key Contributions
- **Data Analysis:** Explored 24 classes with 64×64 grayscale images  
- **Model Architecture:** 2 Conv layers + 2 Dense layers (407,832 params)  
- **Training Pipeline:** Scripted training with validation (91% accuracy)  
- **Evaluation:** Metrics, confusion matrix, per-class analysis  
- **Documentation:** Clear results, reproducibility, and future optimization potential  

---

## Project Structure
accelerated_network_SoC/
├── data/
│ ├── train/ # Training images (24 classes)
│ └── test/ # Test images
├── models/
│ ├── baseline/ # minimal_cnn.h5 (90.47% accuracy)
│ ├── optimized/ # Future optimized models
│ └── transfer_learning/
├── results/
│ ├── baseline/ # Baseline evaluation
│ ├── optimized/
│ └── comparisons/
├── scripts/
│ ├── train_baseline.py
│ └── evaluate_baseline.py
├── utils/
├── notebooks/
├── training/
├── evaluation/
├── config.py
└── requirements.txt

text

## Quick Start

1. **Activate your virtual environment**
   ```bash
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux / Mac
Install dependencies

bash
pip install -r requirements.txt
Train the baseline model

bash
python scripts/train_baseline.py
Evaluate the baseline model

bash
python scripts/evaluate_baseline.py
Dataset Statistics
Total Classes: 24

Image Size: 64×64 grayscale

Training Images: ~4700

Test Images: 1028

Class Distribution: Highly imbalanced (15 to 1878 samples)

Dependencies
See requirements.txt for the full list

Python 3.10+ recommended
