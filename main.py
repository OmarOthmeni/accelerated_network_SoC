#!/usr/bin/env python3
"""
Main entry point for Rover Image Classification Project
Student: Donia
Project: Martian Rover Image Classification
Baseline Accuracy: 90.47%
"""

import os
import sys
from datetime import datetime

def main():
    print("=" * 60)
    print("MARTIAN ROVER IMAGE CLASSIFICATION PROJECT")
    print("=" * 60)
    print(f"Student: Donia")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Project overview
    print("📊 PROJECT OVERVIEW:")
    print("  - Classes: 24 rover components/targets")
    print("  - Image size: 64x64 grayscale")
    print("  - Training images: ~4700")
    print("  - Test images: 1028")
    print("  - Baseline accuracy: 90.47%")
    print()
    
    # Project structure
    print("📁 PROJECT STRUCTURE:")
    print("  data/           - Training and test images")
    print("  models/         - Saved models (baseline: 90.47%)")
    print("  scripts/        - Python scripts")
    print("  results/        - Evaluation results")
    print("  utils/          - Utility functions")
    print()
    
    # Available scripts
    print("🚀 AVAILABLE SCRIPTS:")
    scripts_dir = "scripts"
    if os.path.exists(scripts_dir):
        for script in sorted(os.listdir(scripts_dir)):
            if script.endswith('.py'):
                print(f"  - python scripts/{script}")
    else:
        print("  (scripts folder not found)")
    print()
    
    # Model status
    print("🤖 MODEL STATUS:")
    model_path = "models/baseline/minimal_cnn.h5"
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        print(f"  ✓ Baseline model: minimal_cnn.h5 ({size:,} bytes)")
        print(f"    Accuracy: 90.47% on test set")
    else:
        print("  ✗ No baseline model found")
    print()
    
    # Quick commands
    print("⚡ QUICK COMMANDS:")
    print("  1. Test project: python test_project.py")
    print("  2. Evaluate model: python scripts/evaluate_baseline.py")
    print("  3. View results: dir results\\baseline\\")
    print()
    
    print("=" * 60)
    print("✅ Project organized and ready for optimization!")
    print("=" * 60)

if __name__ == "__main__":
    main()