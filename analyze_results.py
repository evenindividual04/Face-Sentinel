import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    roc_curve, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)
import os
import pandas as pd

# =============================================
# CONFIGURATION
# =============================================
#lastval
RESULTS_PATH = "/Users/anmolsen/Documents/icpr2020/icpr2020dfdc/results/net-EfficientNetB4_traindb-celebdf_face-scale_size-224_seed-0_lastval_augment/celebdf_test.pkl"
#bestval
RESULTS_PATH = "/Users/anmolsen/Documents/icpr2020/icpr2020dfdc/results/net-EfficientNetB4_traindb-celebdf_face-scale_size-224_seed-0_bestval_augment/celebdf_test.pkl"
OUTPUT_DIR = "/Users/anmolsen/Documents/icpr2020/icpr2020dfdc/results_analysis/results_analysis_efficientnetb4_aug"
# =============================================

def analyze_and_visualize(results_path, output_dir):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        print(f"Successfully loaded results from {results_path}")
    except FileNotFoundError:
        print(f"Error: File not found at {results_path}")
        return
    
    # Print structure for debugging
    print("\nResults structure:")
    print(f"Type: {type(results)}")
    if isinstance(results, pd.DataFrame):
        print("DataFrame columns:", results.columns.tolist())
    else:
        print("ERROR: Expected DataFrame, got", type(results))
        return
    
    # Extract predictions and labels
    if 'score' in results.columns and 'label' in results.columns:
        preds = results['score'].values
        labels = results['label'].values
    else:
        print("\nERROR: Required columns 'score' or 'label' not found")
        print("Available columns:", results.columns.tolist())
        return
    
    # Compute metrics
    auc = roc_auc_score(labels, preds)
    acc = accuracy_score(labels, (preds > 0.5).astype(int))
    
    # Print metrics
    print("\n" + "="*50)
    print(f"Model Performance Metrics:")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print("="*50 + "\n")
    
    # Visualization
    # 1. ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(labels, preds)
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}', color='blue')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    roc_path = f"{output_dir}/roc_curve.png"
    plt.savefig(roc_path, dpi=300)
    print(f"Saved ROC curve to: {roc_path}")
    
    # 2. Prediction Distribution Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(preds[labels == 0], bins=50, alpha=0.5, label='Real', color='blue')
    plt.hist(preds[labels == 1], bins=50, alpha=0.5, label='Fake', color='red')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Prediction Distribution')
    plt.legend()
    hist_path = f"{output_dir}/prediction_distribution.png"
    plt.savefig(hist_path, dpi=300)
    print(f"Saved prediction histogram to: {hist_path}")
    
    # 3. Confusion Matrix
    plt.figure(figsize=(8, 6))
    y_pred = (preds > 0.5).astype(int)
    cm = confusion_matrix(labels, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Real', 'Fake'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    cm_path = f"{output_dir}/confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    print(f"Saved confusion matrix to: {cm_path}")
    
    plt.show()

if __name__ == "__main__":
    analyze_and_visualize(RESULTS_PATH, OUTPUT_DIR)
    print("\nAnalysis complete! Check the output directory for results.")
