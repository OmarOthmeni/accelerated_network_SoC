"""
Project Configuration
Central settings for Rover Image Classification
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Data configuration
IMAGE_SIZE = (64, 64)
NUM_CLASSES = 24
CLASS_NAMES = [
    'apxs', 'apxs cal target', 'chemcam cal target', 'chemin inlet open',
    'drill', 'drill holes', 'drt front', 'drt side', 'ground', 'horizon',
    'inlet', 'mahli', 'mahli cal target', 'mastcam', 'mastcam cal target',
    'observation tray', 'portion box', 'portion tube', 'portion tube opening',
    'rems uv sensor', 'rover rear deck', 'scoop', 'turret', 'wheel'
]

# Training configuration
BATCH_SIZE = 32
EPOCHS_BASELINE = 5
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Model paths
BASELINE_MODEL_PATH = MODELS_DIR / "baseline" / "minimal_cnn.h5"

# Create directories if they don't exist
for dir_path in [DATA_DIR / "processed", DATA_DIR / "augmented",
                 MODELS_DIR / "optimized", MODELS_DIR / "transfer_learning",
                 RESULTS_DIR / "baseline", RESULTS_DIR / "optimized", 
                 RESULTS_DIR / "comparisons"]:
    dir_path.mkdir(parents=True, exist_ok=True)

print("✅ Project configuration loaded")