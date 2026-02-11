"""
Training Pipeline - Main Script
================================
This script trains ML models to predict AQI.

Steps:
1. Load features from Hopsworks
2. Prepare data for training
3. Train multiple models
4. Evaluate and compare
5. Save best model to Hopsworks Model Registry
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_pipeline.hopsworks_utils import (
    connect_to_hopsworks, 
    get_features_for_training,
    upload_model_to_registry
)
from feature_pipeline.feature_engineering import prepare_features_for_training

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Advanced models (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM not available")


def calculate_metrics(y_true, y_pred, model_name="Model"):
    """
    Calculate all evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
    
    Returns:
        Dictionary with metrics
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
    
    metrics = {
        'model': model_name,
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'R2': round(r2, 4),
        'MAPE': round(mape, 4)
    }
    
    return metrics


def load_and_prepare_data(project):
    """
    Load data from Hopsworks and prepare for training.
    
    Args:
        project: Hopsworks project object
    
    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_names
    """
    print("\n" + "="*60)
    print("📊 LOADING AND PREPARING DATA")
    print("="*60 + "\n")
    
    # Load features from Hopsworks
    print("Step 1/5: Loading features from Hopsworks...")
    df = get_features_for_training(project, feature_group_name="aqi_features", version=1)
    
    if df is None or len(df) == 0:
        print("✗ No data available in feature store!")
        return None, None, None, None, None, None
    
    print(f"✓ Loaded {len(df)} records")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Remove records with placeholder pollution data (aqi = 0)
    print("\nStep 2/5: Cleaning data...")
    df_clean = df[df['aqi'] > 0].copy()
    print(f"✓ Kept {len(df_clean)} records with real AQI data")
    print(f"  (Removed {len(df) - len(df_clean)} placeholder records)")
    
    if len(df_clean) < 50:
        print("\n⚠️  WARNING: Very few records with real pollution data!")
        print("   Recommendation: Run the hourly pipeline more to collect data")
        print("   For now, we'll use all available data including placeholders")
        df_clean = df.copy()
    
    # Sort by timestamp
    df_clean = df_clean.sort_values('timestamp').reset_index(drop=True)
    
    # Prepare features for training (adds lag and rolling features)
    print("\nStep 3/5: Engineering features...")
    df_features = prepare_features_for_training(df_clean)
    print(f"✓ Created {len(df_features.columns)} total columns")
    
    # Define feature columns and target
    feature_cols = [
        # Current weather
        'temperature', 'humidity', 'pressure', 'wind_speed',
        
        # Current pollution
        'pm2_5', 'pm10', 'co', 'no2', 'so2', 'o3',
        
        # Time features
        'hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour',
        
        # Derived features
        'pm_ratio', 'temp_humidity_interaction',
        
        # Lag features (1, 3, 6 hours)
        'pm2_5_lag_1', 'pm2_5_lag_3', 'pm2_5_lag_6',
        'pm10_lag_1', 'pm10_lag_3', 'pm10_lag_6',
        'aqi_lag_1', 'aqi_lag_3', 'aqi_lag_6',
        
        # Rolling features (3, 6 hour windows)
        'pm2_5_rolling_mean_3', 'pm2_5_rolling_mean_6',
        'pm10_rolling_mean_3', 'pm10_rolling_mean_6',
        'aqi_rolling_mean_3', 'aqi_rolling_mean_6',
        
        # AQI change rate
        'aqi_change_rate', 'aqi_change_rate_pct'
    ]
    
    # Keep only features that exist in the dataframe
    available_features = [col for col in feature_cols if col in df_features.columns]
    print(f"✓ Using {len(available_features)} features for training")
    
    target_col = 'aqi'
    
    # Create feature matrix and target
    X = df_features[available_features].copy()
    y = df_features[target_col].copy()
    
    # Handle any remaining NaN values
    X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    print(f"\nData shape:")
    print(f"  Features (X): {X.shape}")
    print(f"  Target (y): {y.shape}")
    print(f"  AQI range: {y.min():.1f} to {y.max():.1f}")
    
    # Split data (80% train, 20% test)
    print("\nStep 4/5: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False  # Don't shuffle time series!
    )
    
    print(f"✓ Train set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    
    # Scale features
    print("\nStep 5/5: Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✓ Features scaled using StandardScaler")
    
    print("\n" + "="*60)
    print("✅ DATA PREPARATION COMPLETE")
    print("="*60)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, available_features


def train_models(X_train, X_test, y_train, y_test):
    """
    Train the 3 best ML models and compare performance.
    
    Models:
    - Random Forest: Ensemble of decision trees
    - XGBoost: Gradient boosting with regularization
    - LightGBM: Fast gradient boosting
    
    Args:
        X_train, X_test, y_train, y_test: Training and test data
    
    Returns:
        Dictionary with trained models and their metrics
    """
    print("\n" + "="*60)
    print("🤖 TRAINING 3 BEST MODELS")
    print("="*60 + "\n")
    
    models = {}
    results = []
    
    # 1. Random Forest
    print("1️⃣  Training Random Forest...")
    print("   Ensemble of 100 decision trees")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    metrics_rf = calculate_metrics(y_test, y_pred_rf, "Random Forest")
    models['Random Forest'] = rf
    results.append(metrics_rf)
    print(f"   ✓ RMSE: {metrics_rf['RMSE']}, MAE: {metrics_rf['MAE']}, R²: {metrics_rf['R2']}")
    
    # 2. XGBoost
    if XGBOOST_AVAILABLE:
        print("\n2️⃣  Training XGBoost...")
        print("   Optimized gradient boosting")
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        metrics_xgb = calculate_metrics(y_test, y_pred_xgb, "XGBoost")
        models['XGBoost'] = xgb_model
        results.append(metrics_xgb)
        print(f"   ✓ RMSE: {metrics_xgb['RMSE']}, MAE: {metrics_xgb['MAE']}, R²: {metrics_xgb['R2']}")
    else:
        print("\n2️⃣  XGBoost not available - skipping")
    
    # 3. LightGBM
    if LIGHTGBM_AVAILABLE:
        print("\n3️⃣  Training LightGBM...")
        print("   Fast gradient boosting framework")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        y_pred_lgb = lgb_model.predict(X_test)
        metrics_lgb = calculate_metrics(y_test, y_pred_lgb, "LightGBM")
        models['LightGBM'] = lgb_model
        results.append(metrics_lgb)
        print(f"   ✓ RMSE: {metrics_lgb['RMSE']}, MAE: {metrics_lgb['MAE']}, R²: {metrics_lgb['R2']}")
    else:
        print("\n3️⃣  LightGBM not available - skipping")
    
    print("\n" + "="*60)
    print(f"✅ TRAINED {len(models)} MODELS")
    print("="*60)
    
    return models, results


def print_results_table(results):
    """
    Print comparison table of all models.
    
    Args:
        results: List of metric dictionaries
    """
    print("\n" + "="*60)
    print("📊 MODEL COMPARISON")
    print("="*60 + "\n")
    
    # Create DataFrame for better formatting
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('RMSE')
    
    print(df_results.to_string(index=False))
    
    # Find best model
    best_idx = df_results['RMSE'].idxmin()
    best_model = df_results.loc[best_idx, 'model']
    best_rmse = df_results.loc[best_idx, 'RMSE']
    
    print("\n" + "="*60)
    print(f"🏆 BEST MODEL: {best_model}")
    print(f"   RMSE: {best_rmse}")
    print("="*60)
    
    return best_model


def save_models(models, scaler, feature_names, project):
    """
    Save best model and scaler locally and to Hopsworks.
    
    Args:
        models: Dictionary of trained models
        scaler: Fitted scaler
        feature_names: List of feature names
        project: Hopsworks project
    """
    print("\n" + "="*60)
    print("💾 SAVING MODELS")
    print("="*60 + "\n")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save all models locally
    for name, model in models.items():
        filename = f"models/{name.replace(' ', '_').lower()}.pkl"
        joblib.dump(model, filename)
        print(f"✓ Saved {name} to {filename}")
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    print(f"✓ Saved scaler to models/scaler.pkl")
    
    # Save feature names
    joblib.dump(feature_names, 'models/feature_names.pkl')
    print(f"✓ Saved feature names to models/feature_names.pkl")
    
    print("\n✅ All models saved locally in 'models/' directory")


def run_training_pipeline():
    """
    Run the complete training pipeline.
    """
    print("\n" + "="*70)
    print("🚀 TRAINING PIPELINE STARTED")
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Connect to Hopsworks
    project = connect_to_hopsworks()
    if project is None:
        print("✗ Cannot proceed without Hopsworks connection")
        return False
    
    # Load and prepare data
    result = load_and_prepare_data(project)
    if result[0] is None:
        print("✗ Cannot proceed without data")
        return False
    
    X_train, X_test, y_train, y_test, scaler, feature_names = result
    
    # Train models
    models, results = train_models(X_train, X_test, y_train, y_test)
    
    # Print comparison table
    best_model_name = print_results_table(results)
    
    # Save models
    save_models(models, scaler, feature_names, project)
    
    print("\n" + "="*70)
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")
    
    print("📝 Summary:")
    print(f"  • Trained {len(models)} models")
    print(f"  • Best model: {best_model_name}")
    print(f"  • Models saved to: models/")
    print(f"  • Features used: {len(feature_names)}")
    
    print("\n💡 Next steps:")
    print("  1. Review model performance in the comparison table")
    print("  2. Run SHAP analysis for interpretability")
    print("  3. Build the dashboard to visualize predictions")
    
    return True


if __name__ == "__main__":
    try:
        success = run_training_pipeline()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Training pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)