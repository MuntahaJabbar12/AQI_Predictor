"""
Model Evaluation and Analysis
==============================
This script provides detailed evaluation and analysis of trained models.

Features:
- Cross-validation scores
- Residual analysis
- Feature importance
- SHAP values for interpretability
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not available. Install with: pip install shap")


def load_model_and_data(model_name='random_forest'):
    """
    Load trained model, scaler, and feature names.
    
    Args:
        model_name: Name of the model to load
    
    Returns:
        model, scaler, feature_names
    """
    try:
        model_path = f'models/{model_name}.pkl'
        scaler_path = 'models/scaler.pkl'
        features_path = 'models/feature_names.pkl'
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_names = joblib.load(features_path)
        
        print(f"✓ Loaded {model_name}")
        return model, scaler, feature_names
        
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        return None, None, None


def plot_residuals(y_true, y_pred, model_name='Model'):
    """
    Plot residual analysis.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Predicted vs Actual
    axes[0].scatter(y_true, y_pred, alpha=0.5)
    axes[0].plot([y_true.min(), y_true.max()], 
                 [y_true.min(), y_true.max()], 
                 'r--', lw=2)
    axes[0].set_xlabel('Actual AQI')
    axes[0].set_ylabel('Predicted AQI')
    axes[0].set_title(f'{model_name}: Predicted vs Actual')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Residuals vs Predicted
    axes[1].scatter(y_pred, residuals, alpha=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted AQI')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title(f'{model_name}: Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Residual Distribution
    axes[2].hist(residuals, bins=30, edgecolor='black')
    axes[2].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[2].set_xlabel('Residuals')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title(f'{model_name}: Residual Distribution')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{model_name.replace(" ", "_").lower()}_residuals.png', 
                dpi=300, bbox_inches='tight')
    print(f"✓ Residual plots saved to plots/{model_name.replace(' ', '_').lower()}_residuals.png")
    
    plt.close()


def plot_feature_importance(model, feature_names, model_name='Model', top_n=20):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        model_name: Name of the model
        top_n: Number of top features to show
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"⚠️  {model_name} doesn't have feature_importances_")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.title(f'{model_name}: Top {top_n} Feature Importances')
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{model_name.replace(" ", "_").lower()}_feature_importance.png',
                dpi=300, bbox_inches='tight')
    print(f"✓ Feature importance plot saved to plots/{model_name.replace(' ', '_').lower()}_feature_importance.png")
    
    plt.close()
    
    # Print top features
    print(f"\n🔝 Top 10 Most Important Features for {model_name}:")
    for i in range(min(10, top_n)):
        idx = indices[i]
        print(f"   {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")


def shap_analysis(model, X_sample, feature_names, model_name='Model'):
    """
    Perform SHAP analysis for model interpretability.
    
    Args:
        model: Trained model
        X_sample: Sample of features (not too large, e.g., 100 samples)
        feature_names: List of feature names
        model_name: Name of the model
    """
    if not SHAP_AVAILABLE:
        print("⚠️  SHAP not available. Install with: pip install shap")
        return
    
    try:
        print(f"\n🔍 Running SHAP analysis for {model_name}...")
        print(f"   Using {len(X_sample)} samples")
        
        # Create explainer based on model type
        if hasattr(model, 'feature_importances_'):
            # Tree-based model
            explainer = shap.TreeExplainer(model)
        else:
            # Linear model or other
            explainer = shap.LinearExplainer(model, X_sample)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        
        # Save plot
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/{model_name.replace(" ", "_").lower()}_shap_summary.png',
                    dpi=300, bbox_inches='tight')
        print(f"✓ SHAP summary plot saved to plots/{model_name.replace(' ', '_').lower()}_shap_summary.png")
        
        plt.close()
        
        # Feature importance from SHAP
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                         plot_type='bar', show=False)
        plt.tight_layout()
        plt.savefig(f'plots/{model_name.replace(" ", "_").lower()}_shap_importance.png',
                    dpi=300, bbox_inches='tight')
        print(f"✓ SHAP importance plot saved to plots/{model_name.replace(' ', '_').lower()}_shap_importance.png")
        
        plt.close()
        
        print("✓ SHAP analysis complete!")
        
    except Exception as e:
        print(f"✗ Error in SHAP analysis: {str(e)}")


def evaluate_model_comprehensive(model_name='random_forest'):
    """
    Comprehensive evaluation of a trained model.
    
    Args:
        model_name: Name of the model to evaluate
    """
    print("\n" + "="*60)
    print(f"📊 COMPREHENSIVE MODEL EVALUATION: {model_name.upper()}")
    print("="*60 + "\n")
    
    # Load model
    model, scaler, feature_names = load_model_and_data(model_name)
    if model is None:
        return
    
    # Load test data (you'll need to save this during training)
    # For now, we'll note that this requires the training script to save test data
    print("⚠️  Note: Full evaluation requires test data from training pipeline")
    print("   Run train_model.py first to generate test data")
    
    print("\n✅ Model loaded successfully!")
    print(f"   Number of features: {len(feature_names)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, default='random_forest',
                       help='Model name to evaluate (e.g., random_forest, xgboost)')
    
    args = parser.parse_args()
    
    evaluate_model_comprehensive(args.model)
