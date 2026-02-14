"""
Data Merger
===========
Combines 3 months historical data with your real-time Hopsworks data

This gives you: Historical depth + Real-time freshness!
"""

import pandas as pd
import sys
import os

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_pipeline.hopsworks_utils import connect_to_hopsworks, get_features_for_training

def load_historical_data(filename='historical_aqi_data.csv'):
    """Load the downloaded historical data."""
    print("\n📂 Loading historical data...")
    
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        print("   Run download_historical_data.py first!")
        return None
    
    df = pd.read_csv(filename)
    
    # FIX: Make historical timestamps UTC-aware to match Hopsworks data
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    print(f"✅ Loaded {len(df)} historical records")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


def load_realtime_data():
    """Load your real-time data from Hopsworks."""
    print("\n📡 Loading real-time data from Hopsworks...")
    
    try:
        project = connect_to_hopsworks()
        if project:
            df = get_features_for_training(project)
            if df is not None and len(df) > 0:
                df['timestamp'] = pd.to_datetime(df['timestamp'])  # already tz-aware
                df_real = df[df['aqi'] > 0]  # Only real data
                
                print(f"✅ Loaded {len(df_real)} real-time records")
                print(f"   Date range: {df_real['timestamp'].min()} to {df_real['timestamp'].max()}")
                
                return df_real
        
        print("⚠️ Could not load from Hopsworks")
        return None
        
    except Exception as e:
        print(f"❌ Error loading Hopsworks data: {str(e)}")
        return None


def merge_datasets(historical_df, realtime_df):
    """
    Intelligently merge historical and real-time data.
    
    Strategy:
    1. Use historical for bulk (90 days)
    2. Append real-time on top (most recent)
    3. Remove duplicates (keep real-time version)
    """
    print("\n🔄 Merging datasets...")
    
    # Ensure same columns
    common_cols = list(set(historical_df.columns) & set(realtime_df.columns))
    
    if 'timestamp' not in common_cols:
        print("❌ No timestamp column found!")
        return None
    
    print(f"   Common columns: {len(common_cols)}")
    
    # Select common columns
    hist_subset = historical_df[common_cols].copy()
    real_subset = realtime_df[common_cols].copy()
    
    # Mark source
    hist_subset['data_source'] = 'historical'
    real_subset['data_source'] = 'realtime'
    
    # Combine
    combined = pd.concat([hist_subset, real_subset], ignore_index=True)
    
    # Sort by timestamp
    combined = combined.sort_values('timestamp')
    
    # Remove duplicates (keep realtime version)
    combined = combined.drop_duplicates(subset=['timestamp'], keep='last')
    
    # Reset index
    combined = combined.reset_index(drop=True)
    
    print(f"\n✅ Merged successfully!")
    print(f"   Total records: {len(combined)}")
    print(f"   Historical: {len(combined[combined['data_source'] == 'historical'])}")
    print(f"   Real-time: {len(combined[combined['data_source'] == 'realtime'])}")
    print(f"   Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")
    
    return combined


def save_combined_data(df, filename='combined_aqi_data.csv'):
    """Save the merged dataset."""
    df.to_csv(filename, index=False)
    
    print(f"\n💾 Saved combined data to: {filename}")
    
    # Summary statistics
    print("\n📊 Dataset Summary:")
    print(f"   Total records: {len(df)}")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
    print(f"   AQI range: {df['aqi'].min():.1f} - {df['aqi'].max():.1f}")
    print(f"   Average AQI: {df['aqi'].mean():.1f}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    
    # Data quality check
    print("\n🔍 Data Quality:")
    real_count = len(df[df['data_source'] == 'realtime'])
    hist_count = len(df[df['data_source'] == 'historical'])
    
    print(f"   Real-time data: {real_count} records ({real_count/len(df)*100:.1f}%)")
    print(f"   Historical data: {hist_count} records ({hist_count/len(df)*100:.1f}%)")
    
    if real_count > 0:
        print(f"   ✅ Contains your collected data!")
    
    return filename


def main():
    print("\n" + "="*60)
    print("🔄 DATA MERGER - Historical + Real-time")
    print("="*60)
    
    # Step 1: Load historical data
    historical_df = load_historical_data('historical_aqi_data.csv')
    
    if historical_df is None:
        print("\n❌ Cannot proceed without historical data")
        print("   Run: python download_historical_data.py")
        return
    
    # Step 2: Load real-time data
    realtime_df = load_realtime_data()
    
    if realtime_df is None:
        print("\n⚠️ No real-time data found")
        print("   Continuing with historical data only...")
        combined_df = historical_df
    else:
        # Step 3: Merge
        combined_df = merge_datasets(historical_df, realtime_df)
    
    if combined_df is None:
        print("\n❌ Merge failed!")
        return
    
    # Step 4: Save
    filename = save_combined_data(combined_df)
    
    print("\n" + "="*60)
    print("✅ MERGE COMPLETE!")
    print("="*60)
    
    print("\n🎯 Next Steps:")
    print(f"   1. Your combined data: {filename}")
    print(f"   2. Update train_model.py to use this file")
    print(f"   3. Run: cd training_pipeline && python train_model.py")
    print(f"   4. Deploy updated dashboard!")
    
    print(f"\n📊 You now have {len(combined_df)} records!")
    print(f"   Duration: {(combined_df['timestamp'].max() - combined_df['timestamp'].min()).days} days")
    print(f"   Meets 3-month requirement: {'✅ YES!' if (combined_df['timestamp'].max() - combined_df['timestamp'].min()).days >= 90 else '❌ NO'}")


if __name__ == "__main__":
    main()
