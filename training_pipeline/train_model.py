"""
Training Pipeline - WITH HOPSWORKS MODEL REGISTRY
==================================================
Trains models AND uploads best model to Hopsworks Registry!

Developed by Muntaha Jabbar • 2026
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_pipeline.hopsworks_utils import (
    connect_to_hopsworks, 
    get_features_for_training,
    upload_model_to_registry
)
from feature_pipeline.feature_engineering import prepare_features_for_training

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

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
    """Calculate all evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
    
    return {
        'model': model_name,
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'R2': round(r2, 4),
        'MAPE': round(mape, 4)
    }


def load_and_prepare_data(project):
    """Load data from combined CSV or Hopsworks."""
    print("\n" + "="*60)
    print("📊 LOADING AND PREPARING DATA")
    print("="*60 + "\n")
    
    combined_files = [
        '../combined_aqi_data.csv',
        'combined_aqi_data.csv',
        os.path.join(os.path.dirname(__file__), '..', 'combined_aqi_data.csv')
    ]
    
    df = None
    data_source = None
    
    for filepath in combined_files:
        if os.path.exists(filepath):
            print(f"Step 1/5: Loading combined data from {os.path.basename(filepath)}...")
            try:
                df = pd.read_csv(filepath)
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
                data_source = "Combined (Historical + Real-time)"
                print(f"✓ Found and loaded combined dataset!")
                break
            except Exception as e:
                print(f"⚠️  Error loading {filepath}: {str(e)}")
                continue
    
    if df is None:
        print("Step 1/5: Loading features from Hopsworks...")
        print("   (No combined_aqi_data.csv found - using real-time data only)")
        df = get_features_for_training(project, feature_group_name="aqi_features", version=1)
        data_source = "Real-time only (Hopsworks)"
    
    if df is None or len(df) == 0:
        print("✗ No data available!")
        return None, None, None, None, None, None, None, None
    
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    
    duration_days = (df['timestamp'].max() - df['timestamp'].min()).days
    
    print(f"✓ Loaded {len(df)} records")
    print(f"  Data source: {data_source}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Duration: {duration_days} days")
    
    if duration_days >= 90:
        print(f"  ✅ ✅ ✅ MEETS 3-MONTH REQUIREMENT! ({duration_days} days) ✅ ✅ ✅")
    elif duration_days >= 30:
        print(f"  ⚠️  {duration_days} days (target: 90+ days for best results)")
    else:
        print(f"  ⚠️  Only {duration_days} days - more data recommended")
    
    print("\nStep 2/5: Cleaning data...")
    df_clean = df[df['aqi'] > 0].copy()
    print(f"✓ Kept {len(df_clean)} records with real AQI data")
    print(f"  (Removed {len(df) - len(df_clean)} placeholder records)")
    
    if len(df_clean) < 50:
        print("\n⚠️  WARNING: Very few records with real pollution data!")
        df_clean = df.copy()
    
    df_clean = df_clean.sort_values('timestamp').reset_index(drop=True)
    
    print("\nStep 3/5: Engineering features...")
    df_features = prepare_features_for_training(df_clean)
    print(f"✓ Created {len(df_features.columns)} total columns")
    
    feature_cols = [
        'temperature', 'humidity', 'pressure', 'wind_speed',
        'pm2_5', 'pm10', 'co', 'no2', 'so2', 'o3',
        'hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour',
        'pm_ratio', 'temp_humidity_interaction',
        'pm2_5_lag_1', 'pm2_5_lag_3', 'pm2_5_lag_6',
        'pm10_lag_1', 'pm10_lag_3', 'pm10_lag_6',
        'aqi_lag_1', 'aqi_lag_3', 'aqi_lag_6',
        'pm2_5_rolling_mean_3', 'pm2_5_rolling_mean_6',
        'pm10_rolling_mean_3', 'pm10_rolling_mean_6',
        'aqi_rolling_mean_3', 'aqi_rolling_mean_6',
        'aqi_change_rate', 'aqi_change_rate_pct'
    ]
    
    available_features = [col for col in feature_cols if col in df_features.columns]
    print(f"✓ Using {len(available_features)} features for training")
    
    X = df_features[available_features].copy()
    y = df_features['aqi'].copy()
    X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    print(f"\nData shape:")
    print(f"  Features (X): {X.shape}")
    print(f"  Target (y): {y.shape}")
    print(f"  AQI range: {y.min():.1f} to {y.max():.1f}")
    
    print("\nStep 4/5: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    print(f"✓ Train set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    
    print("\nStep 5/5: Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✓ Features scaled using StandardScaler")
    
    print("\n" + "="*60)
    print("✅ DATA PREPARATION COMPLETE")
    print("="*60)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, available_features, data_source, duration_days


def train_models(X_train, X_test, y_train, y_test):
    """Train 3 best models."""
    print("\n" + "="*60)
    print("🤖 TRAINING 3 BEST MODELS")
    print("="*60 + "\n")
    
    models = {}
    results = []
    
    print("1️⃣  Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    metrics_rf = calculate_metrics(y_test, y_pred_rf, "RandomForest")
    models['RandomForest'] = rf
    results.append(metrics_rf)
    print(f"   ✓ RMSE: {metrics_rf['RMSE']}, MAE: {metrics_rf['MAE']}, R²: {metrics_rf['R2']}")
    
    if XGBOOST_AVAILABLE:
        print("\n2️⃣  Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        metrics_xgb = calculate_metrics(y_test, y_pred_xgb, "XGBoost")
        models['XGBoost'] = xgb_model
        results.append(metrics_xgb)
        print(f"   ✓ RMSE: {metrics_xgb['RMSE']}, MAE: {metrics_xgb['MAE']}, R²: {metrics_xgb['R2']}")
    
    if LIGHTGBM_AVAILABLE:
        print("\n3️⃣  Training LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        y_pred_lgb = lgb_model.predict(X_test)
        metrics_lgb = calculate_metrics(y_test, y_pred_lgb, "LightGBM")
        models['LightGBM'] = lgb_model
        results.append(metrics_lgb)
        print(f"   ✓ RMSE: {metrics_lgb['RMSE']}, MAE: {metrics_lgb['MAE']}, R²: {metrics_lgb['R2']}")
    
    print("\n" + "="*60)
    print(f"✅ TRAINED {len(models)} MODELS")
    print("="*60)
    
    return models, results


def print_results_table(results):
    """Print model comparison."""
    print("\n" + "="*60)
    print("📊 MODEL COMPARISON")
    print("="*60 + "\n")
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('RMSE')
    print(df_results.to_string(index=False))
    
    best_idx = df_results['RMSE'].idxmin()
    best_model = df_results.loc[best_idx, 'model']
    best_rmse = df_results.loc[best_idx, 'RMSE']
    
    print("\n" + "="*60)
    print(f"🏆 BEST MODEL: {best_model}")
    print(f"   RMSE: {best_rmse}")
    print("="*60)
    
    return best_model, df_results.loc[best_idx].to_dict()


def save_models(models, scaler, feature_names, project, data_source, duration_days, best_model_name, best_metrics):
    """Save models locally AND upload best to Hopsworks Registry."""
    print("\n" + "="*60)
    print("💾 SAVING MODELS")
    print("="*60 + "\n")
    
    os.makedirs('models', exist_ok=True)
    
    # Save all models locally
    for name, model in models.items():
        filename = f"models/{name.lower()}.pkl"
        joblib.dump(model, filename)
        print(f"✓ Saved {name} to {filename}")
    
    joblib.dump(scaler, 'models/scaler.pkl')
    print(f"✓ Saved scaler to models/scaler.pkl")
    
    joblib.dump(feature_names, 'models/feature_names.pkl')
    print(f"✓ Saved feature names to models/feature_names.pkl")
    
    # Save metadata
    metadata = {
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_source': data_source,
        'duration_days': duration_days,
        'meets_3_month_requirement': duration_days >= 90,
        'num_features': len(feature_names),
        'feature_names': feature_names,
        'best_model': best_model_name,
        'best_rmse': float(best_metrics['RMSE']),
        'best_mae': float(best_metrics['MAE']),
        'best_r2': float(best_metrics['R2'])
    }
    
    import json
    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to models/metadata.json")
    
    print("\n✅ All models saved locally in 'models/' directory")
    
    # ========== UPLOAD BEST MODEL TO HOPSWORKS REGISTRY ==========
    print("\n" + "="*60)
    print("☁️  UPLOADING BEST MODEL TO HOPSWORKS MODEL REGISTRY")
    print("="*60 + "\n")
    
    try:
        best_model = models.get(best_model_name)
        
        if best_model:
            # Prepare metrics for registry (ONLY NUMERIC VALUES!)
            registry_metrics = {
                'rmse': float(best_metrics['RMSE']),
                'mae': float(best_metrics['MAE']),
                'r2_score': float(best_metrics['R2']),
                'mape': float(best_metrics['MAPE']),
                'training_days': int(duration_days)
            }
            
            # Description (can include strings)
            description = f"AQI Predictor - {best_model_name} model. Trained on {duration_days} days of {data_source}. Accuracy: R²={best_metrics['R2']:.4f}, RMSE={best_metrics['RMSE']:.4f}"
            
            print(f"📤 Uploading {best_model_name} to Hopsworks...")
            print(f"   Metrics: RMSE={best_metrics['RMSE']}, R²={best_metrics['R2']}")
            
            # Upload to registry
            model_version = upload_model_to_registry(
                project=project,
                model=best_model,
                model_name="aqi_predictor_best",
                metrics=registry_metrics,
                description=description
            )
            
            if model_version:
                print(f"\n✅ MODEL UPLOADED TO HOPSWORKS REGISTRY!")
                print(f"   Name: aqi_predictor_best")
                print(f"   Algorithm: {best_model_name}")
                print(f"   Version: {model_version}")
                print(f"   RMSE: {best_metrics['RMSE']}")
                print(f"   R²: {best_metrics['R2']}")
                print(f"   Access: Hopsworks → Model Registry → aqi_predictor_best")
            else:
                print(f"\n⚠️  Model upload failed, but local models are saved")
        else:
            print(f"⚠️  Best model '{best_model_name}' not found in trained models")
    
    except Exception as e:
        print(f"\n⚠️  Error uploading to registry: {str(e)}")
        print(f"   Local models are safe in 'models/' directory")
        import traceback
        traceback.print_exc()


def run_training_pipeline():
    """Run complete training pipeline WITH registry upload."""
    print("\n" + "="*70)
    print("🚀 TRAINING PIPELINE STARTED (WITH MODEL REGISTRY)")
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    project = connect_to_hopsworks()
    if project is None:
        print("✗ Cannot proceed without Hopsworks connection")
        return False
    
    result = load_and_prepare_data(project)
    if result[0] is None:
        print("✗ Cannot proceed without data")
        return False
    
    X_train, X_test, y_train, y_test, scaler, feature_names, data_source, duration_days = result
    
    models, results = train_models(X_train, X_test, y_train, y_test)
    
    best_model_name, best_metrics = print_results_table(results)
    
    save_models(models, scaler, feature_names, project, data_source, duration_days, best_model_name, best_metrics)
    
    print("\n" + "="*70)
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")
    
    print("📝 Summary:")
    print(f"  • Data source: {data_source}")
    print(f"  • Data duration: {duration_days} days")
    if duration_days >= 90:
        print(f"  • ✅ Meets 3-month requirement!")
    print(f"  • Trained {len(models)} models")
    print(f"  • Best model: {best_model_name}")
    print(f"  • Best RMSE: {best_metrics['RMSE']}")
    print(f"  • Local: models/ directory")
    print(f"  • Cloud: Hopsworks Model Registry")
    
    print("\n💡 Next steps:")
    print("  1. Check Hopsworks UI → Model Registry → aqi_predictor_best")
    print("  2. Deploy dashboard (loads from local or registry)")
    print("  3. Dashboard shows best model automatically")
    
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