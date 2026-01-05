"""
Baseline Model Evaluation
Your exact evaluation code from earlier
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from PIL import Image
import json
from datetime import datetime
import pandas as pd

# Load config
import sys
sys.path.append('..')
try:
    from config import CLASS_NAMES, IMAGE_SIZE
except ImportError:
    # Fallback if config not available
    CLASS_NAMES = [
        'apxs', 'apxs cal target', 'chemcam cal target', 'chemin inlet open',
        'drill', 'drill holes', 'drt front', 'drt side', 'ground', 'horizon',
        'inlet', 'mahli', 'mahli cal target', 'mastcam', 'mastcam cal target',
        'observation tray', 'portion box', 'portion tube', 'portion tube opening',
        'rems uv sensor', 'rover rear deck', 'scoop', 'turret', 'wheel'
    ]
    IMAGE_SIZE = (64, 64)

def load_test_data(test_path="data/test"):
    """Load test data exactly as you did"""
    X_test = []
    y_test = []
    
    for idx, cls in enumerate(CLASS_NAMES):
        cls_path = os.path.join(test_path, cls)
        if not os.path.exists(cls_path):
            continue
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            img = Image.open(img_path).convert("L")
            img_array = np.array(img) / 255.0
            X_test.append(img_array)
            y_test.append(idx)
    
    X_test = np.array(X_test).reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
    y_test = np.array(y_test)
    
    print(f"✅ Loaded {len(X_test)} test images")
    return X_test, y_test

def evaluate_model(model_path, X_test, y_test):
    """Evaluate model and save results"""
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/baseline/evaluation_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"📦 Loading model from {model_path}")
    model = load_model(model_path)
    
    print("🎯 Making predictions...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
    
    # Save metrics
    metrics = {
        'model_name': os.path.basename(model_path),
        'evaluation_date': timestamp,
        'overall_accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'test_samples': len(X_test),
        'per_class_samples': {CLASS_NAMES[i]: int(np.sum(y_test == i)) for i in range(len(CLASS_NAMES))}
    }
    
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save confusion matrix plot
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                annot_kws={"size": 8})
    plt.title(f'Confusion Matrix - Accuracy: {accuracy:.2%}', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save per-class accuracy plot
    per_class_acc = []
    per_class_counts = []
    
    for i in range(len(CLASS_NAMES)):
        mask = y_test == i
        count = np.sum(mask)
        per_class_counts.append(count)
        if count > 0:
            class_acc = accuracy_score(y_test[mask], y_pred[mask])
        else:
            class_acc = 0
        per_class_acc.append(class_acc)
    
    # Create DataFrame for analysis
    analysis_df = pd.DataFrame({
        'Class': CLASS_NAMES,
        'Accuracy': per_class_acc,
        'Samples': per_class_counts,
        'Status': ['Good' if acc >= 0.8 else 'Needs Improvement' if acc >= 0.6 else 'Problematic' for acc in per_class_acc]
    })
    
    analysis_df.to_csv(os.path.join(results_dir, 'class_analysis.csv'), index=False)
    
    # Plot
    plt.figure(figsize=(18, 6))
    bars = plt.bar(range(len(CLASS_NAMES)), per_class_acc)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Per-Class Accuracy (Overall: {accuracy:.2%})', fontsize=16, pad=20)
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha='right', fontsize=9)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    
    # Color coding
    for bar, acc, count in zip(bars, per_class_acc, per_class_counts):
        if acc >= 0.8:
            bar.set_color('green')
        elif acc >= 0.6:
            bar.set_color('orange')
        else:
            bar.set_color('red')
        # Add count label
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{count}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'per_class_accuracy.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\n" + "="*70)
    print("BASELINE MODEL EVALUATION RESULTS")
    print("="*70)
    print(f"📊 Overall Accuracy: {accuracy:.2%}")
    print(f"📁 Test Samples: {len(X_test)}")
    print(f"💾 Results saved to: {results_dir}")
    print("="*70)
    
    return metrics, results_dir, y_pred

def main():
    print("🚀 Starting Baseline Model Evaluation")
    print("-" * 50)
    
    # Load test data
    print("📂 Loading test data...")
    X_test, y_test = load_test_data()
    
    # Evaluate baseline model
    model_path = "models/baseline/minimal_cnn.h5"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please run: python scripts/train_baseline.py")
        return
    
    metrics, results_dir, y_pred = evaluate_model(model_path, X_test, y_test)
    
    # Print detailed analysis
    print("\n📈 PERFORMANCE ANALYSIS")
    print("-" * 50)
    
    # Calculate per-class accuracy
    per_class_acc = []
    for i, cls in enumerate(CLASS_NAMES):
        mask = y_test == i
        if np.sum(mask) > 0:
            class_acc = accuracy_score(y_test[mask], y_pred[mask])
            per_class_acc.append((cls, class_acc, np.sum(mask)))
    
    # Sort by accuracy
    per_class_acc.sort(key=lambda x: x[1], reverse=True)
    
    print("\n🏆 TOP 5 BEST PERFORMING CLASSES:")
    for cls, acc, count in per_class_acc[:5]:
        print(f"  {cls:25s}: {acc:.2%} ({count:3d} samples)")
    
    print("\n⚠️  TOP 5 WORST PERFORMING CLASSES:")
    for cls, acc, count in per_class_acc[-5:]:
        print(f"  {cls:25s}: {acc:.2%} ({count:3d} samples)")
    
    print("\n📋 RECOMMENDATIONS FOR OPTIMIZATION:")
    print("  1. Address class imbalance (augment small classes)")
    print("  2. Add dropout layers to prevent overfitting")
    print("  3. Use data augmentation")
    print("  4. Try deeper CNN architecture")
    print("  5. Implement class weighting")
    
    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    main()