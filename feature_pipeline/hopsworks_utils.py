"""
Hopsworks Integration Module - FIXED VERSION
=============================================
FIXED: Now handles UNLIMITED records (not stuck at 100!)

Key changes:
- overwrite=False (append mode)
- wait_for_job=False (async processing)
- start_offline_materialization=False (no blocking)
"""

import os
import pandas as pd
from typing import Optional, Dict, List
from datetime import datetime
import hopsworks

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

HOPSWORKS_API_KEY = os.getenv('HOPSWORKS_API_KEY')
HOPSWORKS_PROJECT_NAME = os.getenv('HOPSWORKS_PROJECT_NAME', 'aqi_predictor')


def connect_to_hopsworks():
    """Connect to Hopsworks."""
    try:
        print("\n🔗 Connecting to Hopsworks...")
        project = hopsworks.login(
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT_NAME
        )
        print(f"✓ Connected to project: {project.name}")
        return project
    except Exception as e:
        print(f"✗ Error connecting: {str(e)}")
        return None


def insert_features(project, df: pd.DataFrame, feature_group_name: str = "aqi_features",
                    version: int = 1, description: str = "AQI prediction features"):
    """
    Insert features - FIXED FOR UNLIMITED RECORDS!
    
    Changes:
    - overwrite=False → Append mode
    - wait_for_job=False → Async (no blocking)
    - start_offline_materialization=False → No waiting
    """
    try:
        fs = project.get_feature_store()
        
        print(f"⚙ Working with feature group: {feature_group_name}")
        print(f"  Records to insert: {len(df)}")
        
        # Convert timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Convert integers
        int_columns = ['aqi', 'hour', 'day_of_week', 'day', 'month', 'is_weekend', 'is_rush_hour']
        for col in int_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype('int64')
        
        # Get or create feature group
        fg = None
        try:
            fg = fs.get_feature_group(name=feature_group_name, version=version)
            print(f"✓ Found existing feature group")
        except:
            print(f"  Creating new feature group...")
            fg = fs.get_or_create_feature_group(
                name=feature_group_name,
                version=version,
                description=description,
                primary_key=['city', 'timestamp'],
                event_time='timestamp',
                online_enabled=False,
                statistics_config=False
            )
            print(f"✓ Feature group created")
        
        # Insert with FIXED settings
        print(f"📤 Inserting {len(df)} records...")
        
        fg.insert(
            df, 
            overwrite=False,  # ← KEY FIX: Append, don't replace!
            write_options={
                "start_offline_materialization": False,  # ← No blocking!
                "wait_for_job": False  # ← Async processing!
            }
        )
        
        print(f"✓ Successfully queued {len(df)} records")
        print(f"  ℹ️ Processing asynchronously (takes 30-60 seconds)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def get_features_for_training(project, feature_group_name: str = "aqi_features",
                               version: int = 1, start_date: str = None,
                               end_date: str = None) -> Optional[pd.DataFrame]:
    """Retrieve features for training."""
    try:
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name=feature_group_name, version=version)
        
        query = fg.select_all()
        
        if start_date:
            query = query.filter(fg.timestamp >= start_date)
        if end_date:
            query = query.filter(fg.timestamp <= end_date)
        
        print(f"📥 Retrieving features from {feature_group_name}...")
        df = query.read()
        print(f"✓ Retrieved {len(df)} records")
        return df
        
    except Exception as e:
        print(f"✗ Error retrieving features: {str(e)}")
        return None


def upload_model_to_registry(project, model, model_name: str = "aqi_predictor",
                              metrics: Dict = None, description: str = ""):
    """Upload model to registry."""
    try:
        import joblib
        from hsml.schema import Schema
        from hsml.model_schema import ModelSchema
        
        mr = project.get_model_registry()
        
        model_dir = "model_files"
        os.makedirs(model_dir, exist_ok=True)
        model_path = f"{model_dir}/{model_name}.pkl"
        
        joblib.dump(model, model_path)
        
        input_schema = Schema([])
        output_schema = Schema([])
        model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)
        
        aqi_model = mr.python.create_model(
            name=model_name,
            metrics=metrics or {},
            description=description,
            model_schema=model_schema
        )
        
        aqi_model.save(model_dir)
        print(f"✓ Model uploaded: {model_name}")
        return aqi_model
        
    except Exception as e:
        print(f"✗ Error uploading model: {str(e)}")
        return None


def get_model_from_registry(project, model_name: str = "aqi_predictor", version: int = None):
    """Download model from registry."""
    try:
        import joblib
        mr = project.get_model_registry()
        
        if version:
            model = mr.get_model(model_name, version=version)
        else:
            model = mr.get_model(model_name)
        
        model_dir = model.download()
        model_path = f"{model_dir}/{model_name}.pkl"
        loaded_model = joblib.load(model_path)
        
        print(f"✓ Model loaded: {model_name}")
        return loaded_model
        
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        return None


if __name__ == "__main__":
    print("Testing Hopsworks (FIXED VERSION)\n")
    
    project = connect_to_hopsworks()
    
    if project:
        print("\n✓ Connection successful!")
        try:
            fs = project.get_feature_store()
            fg = fs.get_feature_group("aqi_features", version=1)
            count = fg.read().count()
            print(f"\n📊 Current records: {count}")
            
            if count > 100:
                print("  ✅ Feature group accepting unlimited records!")
            elif count == 100:
                print("  ⚠️ Stuck at 100 - replace with this fixed version!")
        except:
            print("\n  ℹ️ Feature group not created yet")