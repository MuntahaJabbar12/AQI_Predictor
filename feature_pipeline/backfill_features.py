"""
Backfill Historical Features
=============================
This script backfills historical data to create training dataset.

It fetches historical weather data and attempts to get pollution data
for the past N days to create a robust training dataset.

Note: Due to OpenWeather API free tier limitations, historical pollution
data may not be available. In that case, we'll collect current data
over time to build our dataset.
"""

import sys
import pandas as pd
from datetime import datetime, timedelta
import time
from fetch_data import fetch_historical_weather, fetch_pollution_data, LATITUDE, LONGITUDE, CITY_NAME
from feature_engineering import create_time_features
from hopsworks_utils import connect_to_hopsworks, insert_features


def backfill_features(days_back: int = 30):
    """
    Backfill historical features.
    
    Args:
        days_back: Number of days to backfill
    
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*60)
    print("🔄 HISTORICAL DATA BACKFILL STARTED")
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📅 Backfilling last {days_back} days")
    print("="*60 + "\n")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    print(f"Date range: {start_str} to {end_str}\n")
    
    # Fetch historical weather data
    print("Step 1/3: Fetching historical weather data...")
    weather_data = fetch_historical_weather(LATITUDE, LONGITUDE, start_str, end_str)
    
    if weather_data is None:
        print("✗ Failed to fetch historical weather data.")
        return False
    
    # Convert to DataFrame
    df_weather = pd.DataFrame({
        'timestamp': pd.to_datetime(weather_data['time']),
        'temperature': weather_data['temperature_2m'],
        'humidity': weather_data['relative_humidity_2m'],
        'pressure': weather_data['pressure_msl'],
        'wind_speed': weather_data['wind_speed_10m']
    })
    
    print(f"✓ Fetched {len(df_weather)} hourly weather records")
    
    # Add time features
    print("\nStep 2/3: Engineering features...")
    
    df_weather['city'] = CITY_NAME
    
    # Add time-based features
    for idx, row in df_weather.iterrows():
        time_features = create_time_features(row['timestamp'])
        for key, value in time_features.items():
            df_weather.at[idx, key] = value
    
    # Note about pollution data
    print("\n⚠️  Note about historical pollution data:")
    print("OpenWeather free tier has limited historical pollution data.")
    print("We'll add placeholder values for now and collect real data going forward.")
    print("The model will be trained on real data as it accumulates.\n")
    
    # Add placeholder pollution data (will be replaced with real data over time)
    pollution_cols = ['pm2_5', 'pm10', 'co', 'no2', 'so2', 'o3', 'aqi']
    for col in pollution_cols:
        df_weather[col] = 0.0  # Placeholder
    
    # Try to get at least current pollution data for the most recent hour
    print("Fetching current pollution data for latest timestamp...")
    current_pollution = fetch_pollution_data(LATITUDE, LONGITUDE)
    
    if current_pollution:
        # Update the most recent row with actual pollution data
        latest_idx = df_weather['timestamp'].idxmax()
        df_weather.at[latest_idx, 'pm2_5'] = current_pollution['pm2_5']
        df_weather.at[latest_idx, 'pm10'] = current_pollution['pm10']
        df_weather.at[latest_idx, 'co'] = current_pollution['co']
        df_weather.at[latest_idx, 'no2'] = current_pollution['no2']
        df_weather.at[latest_idx, 'so2'] = current_pollution['so2']
        df_weather.at[latest_idx, 'o3'] = current_pollution['o3']
        df_weather.at[latest_idx, 'aqi'] = current_pollution['aqi']
        print(f"✓ Added current pollution data (AQI: {current_pollution['aqi']})")
    
    # Add derived features
    df_weather['pm_ratio'] = df_weather['pm2_5'] / df_weather['pm10'].replace(0, 1)
    df_weather['temp_humidity_interaction'] = df_weather['temperature'] * df_weather['humidity']
    
    print(f"✓ Created features for {len(df_weather)} records")
    
    # Connect to Hopsworks
    print("\nStep 3/3: Uploading to Hopsworks...")
    project = connect_to_hopsworks()
    
    if project is None:
        print("✗ Failed to connect to Hopsworks.")
        return False
    
    # Insert features
    success = insert_features(
        project=project,
        df=df_weather,
        feature_group_name="aqi_features",
        version=1,
        description="Historical AQI prediction features for Karachi"
    )
    
    if success:
        print("\n" + "="*60)
        print("✅ BACKFILL COMPLETED SUCCESSFULLY!")
        print("="*60 + "\n")
        
        print("Summary:")
        print(f"  • Records inserted: {len(df_weather)}")
        print(f"  • Date range: {start_str} to {end_str}")
        print(f"  • Features per record: {len(df_weather.columns)}")
        print(f"\n💡 Recommendation:")
        print("   Run the hourly pipeline (run_pipeline.py) to collect real")
        print("   pollution data going forward. As data accumulates, the model")
        print("   will have more real training data to work with.")
        
        return True
    else:
        print("\n✗ Failed to insert features.")
        return False


def backfill_recent_pollution_only(hours_back: int = 24):
    """
    Alternative approach: Only backfill recent hours where we might get pollution data.
    
    Args:
        hours_back: Number of hours to attempt backfill
    """
    print("\n" + "="*60)
    print("🔄 RECENT DATA COLLECTION")
    print(f"Attempting to collect last {hours_back} hours of data")
    print("="*60 + "\n")
    
    all_data = []
    
    # This approach won't work well with free tier as OpenWeather
    # doesn't provide historical pollution data
    print("⚠️  Note: This requires a paid OpenWeather plan for historical data.")
    print("Instead, run the hourly pipeline regularly to build your dataset.\n")
    
    return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Backfill historical AQI features')
    parser.add_argument('--days', type=int, default=30, 
                       help='Number of days to backfill (default: 30)')
    
    args = parser.parse_args()
    
    try:
        success = backfill_features(days_back=args.days)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Backfill failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
