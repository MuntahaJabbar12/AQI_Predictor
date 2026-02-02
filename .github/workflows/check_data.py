"""
View Data in Hopsworks Feature Store
=====================================
This script lets you view the actual data stored in Hopsworks.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_pipeline.hopsworks_utils import connect_to_hopsworks, get_features_for_training
import pandas as pd

def view_feature_store_data():
    """
    View data from Hopsworks Feature Store.
    """
    print("\n" + "="*60)
    print("👀 VIEWING HOPSWORKS DATA")
    print("="*60 + "\n")
    
    # Connect to Hopsworks
    project = connect_to_hopsworks()
    if project is None:
        return
    
    # Get feature store
    fs = project.get_feature_store()
    
    # Get feature group
    try:
        fg = fs.get_feature_group(name="aqi_features", version=1)
        print(f"✓ Found feature group: {fg.name}")
        print(f"  Version: {fg.version}")
        print(f"  Primary keys: {fg.primary_key}")
        print(f"  Event time: {fg.event_time}")
        
        # Get schema
        print(f"\n📋 Schema ({len(fg.features)} features):")
        for feature in fg.features:
            print(f"  • {feature.name}: {feature.type}")
        
        # Read data
        print(f"\n📥 Reading data from feature store...")
        df = fg.read()
        
        print(f"\n✅ DATA LOADED SUCCESSFULLY!")
        print(f"="*60)
        
        # Show statistics
        print(f"\n📊 Data Statistics:")
        print(f"  • Total records: {len(df)}")
        print(f"  • Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  • Features: {len(df.columns)}")
        
        # Show records with real AQI data
        df_real = df[df['aqi'] > 0]
        print(f"\n  • Records with real AQI: {len(df_real)} ({len(df_real)/len(df)*100:.1f}%)")
        print(f"  • Records with placeholder AQI: {len(df) - len(df_real)}")
        
        # Show first few rows
        print(f"\n📄 First 5 Records:")
        print("="*60)
        print(df.head().to_string())
        
        # Show last few rows
        print(f"\n📄 Last 5 Records:")
        print("="*60)
        print(df.tail().to_string())
        
        # Show AQI distribution
        if len(df_real) > 0:
            print(f"\n📈 AQI Statistics (real data only):")
            print(f"  • Mean: {df_real['aqi'].mean():.2f}")
            print(f"  • Min: {df_real['aqi'].min():.2f}")
            print(f"  • Max: {df_real['aqi'].max():.2f}")
            print(f"  • Median: {df_real['aqi'].median():.2f}")
        
        # Show recent records
        print(f"\n🕐 Most Recent 10 Records:")
        print("="*60)
        recent = df.nlargest(10, 'timestamp')[['timestamp', 'city', 'aqi', 'pm2_5', 'temperature', 'humidity']]
        print(recent.to_string())
        
        # Save to CSV for inspection
        print(f"\n💾 Saving data to CSV for inspection...")
        df.to_csv('hopsworks_data_export.csv', index=False)
        print(f"✓ Data saved to: hopsworks_data_export.csv")
        print(f"  You can open this file in Excel or any text editor")
        
        return df
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    df = view_feature_store_data()
    
    if df is not None:
        print("\n" + "="*60)
        print("✅ SUCCESS!")
        print("="*60)
        print("\nYour data is in Hopsworks! The UI just shows the schema.")
        print("Check the CSV file for full data inspection.")
    else:
        print("\n" + "="*60)
        print("❌ FAILED")
        print("="*60)
        print("\nCouldn't retrieve data. Check error messages above.")
