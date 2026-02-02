"""
Feature Pipeline - Main Runner
===============================
This script runs the complete feature pipeline:
1. Fetch current data from APIs
2. Engineer features
3. Store in Hopsworks

This script will be run hourly by GitHub Actions.
"""

import sys
import pandas as pd
from datetime import datetime
from fetch_data import fetch_all_current_data
from feature_engineering import create_features_from_raw_data
from hopsworks_utils import connect_to_hopsworks, insert_features


def run_feature_pipeline():
    """
    Run the complete feature pipeline.
    
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*60)
    print("🚀 FEATURE PIPELINE STARTED")
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")
    
    # Step 1: Fetch raw data from APIs
    print("Step 1/4: Fetching data from APIs...")
    raw_data = fetch_all_current_data()
    
    if raw_data is None:
        print("✗ Failed to fetch data. Pipeline aborted.")
        return False
    
    # Step 2: Create features
    print("\nStep 2/4: Engineering features...")
    features = create_features_from_raw_data(raw_data)
    
    # Convert to DataFrame
    df = pd.DataFrame([features])
    
    print(f"✓ Created {len(df.columns)} features")
    print(f"  Columns: {', '.join(df.columns[:10])}...")
    
    # Step 3: Connect to Hopsworks
    print("\nStep 3/4: Connecting to Hopsworks...")
    project = connect_to_hopsworks()
    
    if project is None:
        print("✗ Failed to connect to Hopsworks. Pipeline aborted.")
        print("\n⚠️  Make sure you have:")
        print("   1. Created a Hopsworks account")
        print("   2. Added HOPSWORKS_API_KEY to .env file")
        print("   3. Set HOPSWORKS_PROJECT_NAME in .env file")
        return False
    
    # Step 4: Insert features into Hopsworks
    print("\nStep 4/4: Inserting features into feature store...")
    success = insert_features(
        project=project,
        df=df,
        feature_group_name="aqi_features",
        version=1,
        description="Hourly AQI prediction features for Karachi"
    )
    
    if success:
        print("\n" + "="*60)
        print("✅ FEATURE PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60 + "\n")
        
        # Print summary
        print("Summary:")
        print(f"  • Timestamp: {features['timestamp']}")
        print(f"  • City: {features['city']}")
        print(f"  • Temperature: {features['temperature']}°C")
        print(f"  • Humidity: {features['humidity']}%")
        print(f"  • AQI: {features['aqi']}")
        print(f"  • PM2.5: {features['pm2_5']} μg/m³")
        print(f"  • Features created: {len(df.columns)}")
        
        return True
    else:
        print("\n✗ Failed to insert features. Pipeline failed.")
        return False


if __name__ == "__main__":
    try:
        success = run_feature_pipeline()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
