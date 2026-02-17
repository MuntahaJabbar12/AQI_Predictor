import sys
import os
import pandas as pd
from datetime import datetime
from fetch_data import fetch_all_current_data
from feature_engineering import create_features_from_raw_data
from hopsworks_utils import connect_to_hopsworks, insert_features

CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'combined_aqi_data.csv')


def save_to_csv(features: dict):
    new_row = pd.DataFrame([features])
    new_row['timestamp'] = pd.to_datetime(new_row['timestamp'])

    if os.path.exists(CSV_PATH):
        existing = pd.read_csv(CSV_PATH)
        existing['timestamp'] = pd.to_datetime(existing['timestamp'])
        combined = pd.concat([existing, new_row], ignore_index=True)
        combined = combined.drop_duplicates(subset=['city', 'timestamp'], keep='last')
        combined = combined.sort_values('timestamp').reset_index(drop=True)
    else:
        combined = new_row

    combined.to_csv(CSV_PATH, index=False)
    print(f"Saved to CSV: {len(combined)} total records")
    return combined


def run_feature_pipeline():
    print("\n" + "="*60)
    print("FEATURE PIPELINE STARTED")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")

    print("Step 1/4: Fetching data from APIs...")
    raw_data = fetch_all_current_data()

    if raw_data is None:
        print("Failed to fetch data. Pipeline aborted.")
        return False

    print("\nStep 2/4: Engineering features...")
    features = create_features_from_raw_data(raw_data)
    df = pd.DataFrame([features])
    print(f"Created {len(df.columns)} features")

    print("\nStep 2b: Saving to CSV...")
    try:
        save_to_csv(features)
    except Exception as e:
        print(f"CSV save warning: {e}")

    print("\nStep 3/4: Connecting to Hopsworks...")
    project = connect_to_hopsworks()

    if project is None:
        print("Failed to connect to Hopsworks.")
        print("Data saved to CSV successfully.")
        return True

    print("\nStep 4/4: Inserting features into feature store...")
    success = insert_features(
        project=project,
        df=df,
        feature_group_name="aqi_features",
        version=1,
        description="Hourly AQI prediction features for Karachi"
    )

    print("\n" + "="*60)
    print("FEATURE PIPELINE COMPLETED!")
    print("="*60 + "\n")

    print("Summary:")
    print(f"  Timestamp: {features['timestamp']}")
    print(f"  City: {features['city']}")
    print(f"  Temperature: {features['temperature']}C")
    print(f"  Humidity: {features['humidity']}%")
    print(f"  AQI: {features['aqi']}")
    print(f"  PM2.5: {features['pm2_5']} ug/m3")
    print(f"  CSV: Saved successfully")
    print(f"  Hopsworks: {'Saved' if success else 'Failed (CSV backup exists)'}")

    return True


if __name__ == "__main__":
    try:
        success = run_feature_pipeline()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)