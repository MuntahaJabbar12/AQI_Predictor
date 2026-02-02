"""
Hopsworks Integration Module
=============================
This module handles all interactions with Hopsworks Feature Store.

Functions:
- Connect to Hopsworks
- Create/Get feature groups
- Insert features
- Retrieve features for training
"""

import os
import pandas as pd
from typing import Optional, Dict, List
from datetime import datetime
import hopsworks
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

HOPSWORKS_API_KEY = os.getenv('HOPSWORKS_API_KEY')
HOPSWORKS_PROJECT_NAME = os.getenv('HOPSWORKS_PROJECT_NAME', 'aqi_predictor')


def connect_to_hopsworks():
    """
    Connect to Hopsworks and return project object.
    
    Returns:
        Hopsworks project object
    """
    try:
        print("\n🔗 Connecting to Hopsworks...")
        
        project = hopsworks.login(
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT_NAME
        )
        
        print(f"✓ Connected to project: {project.name}")
        return project
        
    except Exception as e:
        print(f"✗ Error connecting to Hopsworks: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check your HOPSWORKS_API_KEY in .env file")
        print("2. Verify your project name is correct")
        print("3. Make sure you have internet connection")
        return None


def get_or_create_feature_group(project, name: str, version: int = 1, 
                                 description: str = "", primary_keys: List[str] = None,
                                 event_time: str = "timestamp"):
    """
    Get existing feature group or create new one.
    
    Args:
        project: Hopsworks project object
        name: Name of feature group
        version: Version number
        description: Description of feature group
        primary_keys: List of primary key column names
        event_time: Name of timestamp column
    
    Returns:
        Feature group object
    """
    try:
        fs = project.get_feature_store()
        
        # Try to get existing feature group
        try:
            fg = fs.get_feature_group(name=name, version=version)
            print(f"✓ Retrieved existing feature group: {name} (v{version})")
            return fg
        except:
            # Create new feature group
            print(f"⚙ Creating new feature group: {name} (v{version})")
            
            # We'll create it when we first insert data
            return None
            
    except Exception as e:
        print(f"✗ Error with feature group: {str(e)}")
        return None


def insert_features(project, df: pd.DataFrame, feature_group_name: str = "aqi_features",
                    version: int = 1, description: str = "AQI prediction features"):
    """
    Insert features into Hopsworks feature group.
    
    Args:
        project: Hopsworks project object
        df: DataFrame with features
        feature_group_name: Name of feature group
        version: Version number
        description: Description
    
    Returns:
        True if successful, False otherwise
    """
    try:
        fs = project.get_feature_store()
        
        print(f"⚙ Working with feature group: {feature_group_name}")
        print(f"  Data shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print(f"  Converted timestamp column to datetime")
        
        # Convert integer columns to proper int type (not float)
        int_columns = ['aqi', 'hour', 'day_of_week', 'day', 'month', 'is_weekend', 'is_rush_hour']
        for col in int_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype('int64')
        print(f"  Converted integer columns to int64")
        
        # Try to get existing feature group
        fg = None
        try:
            fg = fs.get_feature_group(name=feature_group_name, version=version)
            print(f"✓ Found existing feature group: {feature_group_name}")
        except Exception as e:
            print(f"  Feature group doesn't exist yet, will create it")
        
        if fg is not None:
            # Insert into existing feature group
            print(f"📤 Inserting {len(df)} records to existing feature group...")
            fg.insert(df, write_options={"start_offline_materialization": False})
            print(f"✓ Successfully inserted {len(df)} records")
        else:
            # Create new feature group with data
            print(f"📝 Creating new feature group: {feature_group_name}")
            fg = fs.get_or_create_feature_group(
                name=feature_group_name,
                version=version,
                description=description,
                primary_key=['city', 'timestamp'],
                event_time='timestamp',
                online_enabled=False
            )
            print(f"✓ Feature group created")
            
            # Insert data (offline mode to avoid Kafka requirement)
            print(f"📤 Inserting {len(df)} records...")
            fg.insert(df, write_options={"start_offline_materialization": False})
            print(f"✓ Successfully inserted {len(df)} records")
        
        return True
        
    except Exception as e:
        print(f"✗ Error inserting features: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def get_features_for_training(project, feature_group_name: str = "aqi_features",
                               version: int = 1, start_date: str = None,
                               end_date: str = None) -> Optional[pd.DataFrame]:
    """
    Retrieve features from Hopsworks for model training.
    
    Args:
        project: Hopsworks project object
        feature_group_name: Name of feature group
        version: Version number
        start_date: Start date filter (YYYY-MM-DD)
        end_date: End date filter (YYYY-MM-DD)
    
    Returns:
        DataFrame with features or None if error
    """
    try:
        fs = project.get_feature_store()
        
        # Get feature group
        fg = fs.get_feature_group(name=feature_group_name, version=version)
        
        # Build query
        query = fg.select_all()
        
        # Add date filters if provided
        if start_date:
            query = query.filter(fg.timestamp >= start_date)
        if end_date:
            query = query.filter(fg.timestamp <= end_date)
        
        # Read data
        print(f"📥 Retrieving features from {feature_group_name}...")
        df = query.read()
        
        print(f"✓ Retrieved {len(df)} records")
        return df
        
    except Exception as e:
        print(f"✗ Error retrieving features: {str(e)}")
        return None


def upload_model_to_registry(project, model, model_name: str = "aqi_predictor",
                              metrics: Dict = None, description: str = ""):
    """
    Upload trained model to Hopsworks Model Registry.
    
    Args:
        project: Hopsworks project object
        model: Trained model object
        model_name: Name for the model
        metrics: Dictionary of evaluation metrics
        description: Model description
    
    Returns:
        Model version object or None if error
    """
    try:
        import joblib
        from hsml.schema import Schema
        from hsml.model_schema import ModelSchema
        
        mr = project.get_model_registry()
        
        # Save model locally first
        model_dir = "model_files"
        os.makedirs(model_dir, exist_ok=True)
        model_path = f"{model_dir}/{model_name}.pkl"
        
        joblib.dump(model, model_path)
        
        # Create model schema (optional but recommended)
        input_schema = Schema([])  # Define based on your features
        output_schema = Schema([])  # Define based on your output
        model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)
        
        # Create model in registry
        aqi_model = mr.python.create_model(
            name=model_name,
            metrics=metrics or {},
            description=description,
            model_schema=model_schema
        )
        
        # Upload model
        aqi_model.save(model_dir)
        
        print(f"✓ Model uploaded to registry: {model_name}")
        return aqi_model
        
    except Exception as e:
        print(f"✗ Error uploading model: {str(e)}")
        return None


def get_model_from_registry(project, model_name: str = "aqi_predictor", version: int = None):
    """
    Download model from Hopsworks Model Registry.
    
    Args:
        project: Hopsworks project object
        model_name: Name of the model
        version: Model version (None for latest)
    
    Returns:
        Trained model object or None if error
    """
    try:
        import joblib
        
        mr = project.get_model_registry()
        
        # Get model
        if version:
            model = mr.get_model(model_name, version=version)
        else:
            model = mr.get_model(model_name)
        
        # Download model
        model_dir = model.download()
        model_path = f"{model_dir}/{model_name}.pkl"
        
        # Load model
        loaded_model = joblib.load(model_path)
        
        print(f"✓ Model loaded from registry: {model_name}")
        return loaded_model
        
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        return None


# Test function
if __name__ == "__main__":
    print("Testing Hopsworks Integration\n")
    
    # Test connection
    project = connect_to_hopsworks()
    
    if project:
        print("\n✓ Hopsworks integration working!")
        print(f"Project: {project.name}")
    else:
        print("\n✗ Please configure your Hopsworks API key in .env file")
        print("\nSteps:")
        print("1. Go to https://app.hopsworks.ai/")
        print("2. Create a free account")
        print("3. Create a project named 'aqi_predictor'")
        print("4. Go to Settings → API Keys")
        print("5. Generate a new API key")
        print("6. Add it to your .env file as HOPSWORKS_API_KEY")