## 🎯 Baseline CNN Implementation (Collaborator: Donia)

### Overview
Implemented a minimal CNN achieving **90.47% test accuracy** on 24-class Mars rover instrument classification.

### Key Contributions:
1. **Data Analysis**: Explored 24 classes with 64x64 grayscale images
2. **Model Architecture**: 2 Conv layers + 2 Dense layers (407,832 params)
3. **Training Pipeline**: Scripted training with validation (91% accuracy)
4. **Evaluation**: Comprehensive metrics, confusion matrix, per-class analysis
5. **Documentation**: Clear results and reproducibility

### Files Added:
- `scripts/train_baseline.py` - Training script
- `scripts/evaluate_baseline.py` - Evaluation script  
- `models/baseline/minimal_cnn.h5` - Trained model
- `results/baseline/` - All evaluation results
- Updated documentation and configuration
# Martian Rover Image Classification

## Project Overview
Classification of 24 types of Martian rover images using deep learning.

## 📊 Baseline Results (Minimal CNN)
- **Test Accuracy**: 90.47%
- **Model Size**: 1.56 MB
- **Training Time**: ~15 seconds per epoch
- **Best Class**: apxs cal target (100%)
- **Worst Class**: chemcam cal target (33%)

## 🏗️ Project Structure
accelerated_network_SoC/
├── data/ # Dataset
│ ├── train/ # Training images (24 classes)
│ └── test/ # Test images
├── models/ # Saved models
│ ├── baseline/ # minimal_cnn.h5 (90.47% accuracy)
│ ├── optimized/ # Future optimized models
│ └── transfer_learning/ # Transfer learning models
├── results/ # Evaluation results
│ ├── baseline/ # Baseline evaluation
│ ├── optimized/ # Future optimized results
│ └── comparisons/ # Model comparisons
├── scripts/ # Python scripts
│ ├── train_baseline.py # Train minimal CNN
│ └── evaluate_baseline.py # Evaluate model
├── utils/ # Utility functions
├── notebooks/ # Jupyter notebooks
├── training/ # Training logs/history
├── evaluation/ # Evaluation metrics
├── config.py # Project configuration
└── requirements.txt # Dependencies


## 🚀 Quick Start
```bash
# 1. Activate virtual environment
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train baseline model
python scripts/train_baseline.py

# 4. Evaluate baseline model
python scripts/evaluate_baseline.py

📈 Dataset Statistics
Total Classes: 24

Image Size: 64×64 grayscale

Training Images: ~4700

Test Images: 1028

Class Distribution: Highly imbalanced (15 to 1878 samples)

🔧 Dependencies
See requirements.txt for complete list.