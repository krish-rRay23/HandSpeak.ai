"""
Evaluate trained model without fine-tuning

Usage:
    python evaluate_model.py --model ./models_trained/run_20260127_201733/best_model.h5
"""

import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model_path, dataset_path, output_dir=None):
    """Evaluate model and generate metrics"""
    
    model_path = Path(model_path)
    if output_dir is None:
        output_dir = model_path.parent
    else:
        output_dir = Path(output_dir)
    
    print("=" * 70)
    print("  Model Evaluation")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_path}")
    print("=" * 70)
    
    # Load model
    print("\n[1/4] Loading model...")
    model = load_model(str(model_path))
    print(f"  Parameters: {model.count_params():,}")
    
    # Load class mapping
    class_mapping_file = model_path.parent / "class_mapping.json"
    if class_mapping_file.exists():
        class_indices = json.loads(class_mapping_file.read_text())
        print(f"  Classes: {len(class_indices)}")
    
    # Create validation generator
    print("\n[2/4] Loading validation data...")
    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    val_generator = val_datagen.flow_from_directory(
        dataset_path,
        target_size=(400, 400),
        batch_size=16,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    print(f"  Validation samples: {val_generator.samples}")
    
    # Evaluate
    print("\n[3/4] Evaluating model...")
    val_loss, val_acc, val_top3 = model.evaluate(val_generator, verbose=1)
    
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  Validation Loss:     {val_loss:.4f}")
    print(f"  Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"  Top-3 Accuracy:      {val_top3:.4f} ({val_top3*100:.2f}%)")
    print("=" * 70)
    
    # Confusion matrix
    print("\n[4/4] Generating confusion matrix...")
    val_generator.reset()
    predictions = model.predict(val_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_generator.classes
    
    class_names = list(val_generator.class_indices.keys())
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f'Confusion Matrix\nAccuracy: {val_acc*100:.2f}%')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    print(f"  Saved: {cm_path}")
    plt.close()
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_file = output_dir / "classification_report.json"
    report_file.write_text(json.dumps(report, indent=2))
    print(f"  Saved: {report_file}")
    
    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(12, 6))
    colors = ['red' if acc < 0.7 else 'orange' if acc < 0.85 else 'green' for acc in per_class_acc]
    plt.bar(class_names, per_class_acc, color=colors)
    plt.title('Per-Class Accuracy')
    plt.xlabel('Letter')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=0)
    plt.ylim([0, 1.05])
    plt.axhline(y=0.7, color='red', linestyle='--', alpha=0.3, label='Poor (<70%)')
    plt.axhline(y=0.85, color='orange', linestyle='--', alpha=0.3, label='Good (70-85%)')
    plt.axhline(y=0.85, color='green', linestyle='--', alpha=0.3, label='Excellent (>85%)')
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    acc_path = output_dir / "per_class_accuracy.png"
    plt.savefig(acc_path, dpi=150)
    print(f"  Saved: {acc_path}")
    plt.close()
    
    # Identify weak letters
    weak_letters = [(class_names[i], per_class_acc[i]) for i in range(len(class_names)) if per_class_acc[i] < 0.7]
    if weak_letters:
        print("\n⚠️  Weak Letters (< 70% accuracy):")
        for letter, acc in weak_letters:
            print(f"    {letter}: {acc*100:.1f}%")
    
    # Save summary
    summary = {
        "model_path": str(model_path),
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss),
        "top3_accuracy": float(val_top3),
        "validation_samples": val_generator.samples,
        "parameters": int(model.count_params()),
        "per_class_accuracy": {class_names[i]: float(per_class_acc[i]) for i in range(len(class_names))},
        "weak_letters": [(letter, float(acc)) for letter, acc in weak_letters]
    }
    
    summary_file = output_dir / "evaluation_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"\n  Summary saved: {summary_file}")
    
    print("\n" + "=" * 70)
    print("  Evaluation Complete!")
    print("=" * 70)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained 26-class model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.h5 file)')
    parser.add_argument('--dataset', type=str, default='./dataset_v2_mobile/skeletons',
                       help='Path to dataset directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for results (default: same as model)')
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        return
    
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Error: Dataset not found at {dataset_path}")
        return
    
    evaluate_model(model_path, dataset_path, args.output)


if __name__ == "__main__":
    main()
