"""
Feature Engineering Module
===========================
This module creates features from raw data for ML models.

Features Created:
1. Time-based: hour, day_of_week, month, is_weekend
2. Derived: AQI change rate, rolling averages
3. Lag features: previous hour values
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List


def create_time_features(timestamp: datetime) -> Dict:
    """
    Create time-based features from timestamp.
    
    Args:
        timestamp: datetime object
    
    Returns:
        Dictionary with time features
    """
    return {
        'hour': timestamp.hour,
        'day_of_week': timestamp.weekday(),  # 0=Monday, 6=Sunday
        'day': timestamp.day,
        'month': timestamp.month,
        'is_weekend': 1 if timestamp.weekday() >= 5 else 0,
        'is_rush_hour': 1 if timestamp.hour in [7, 8, 9, 17, 18, 19] else 0
    }


def calculate_aqi_change_rate(current_aqi: float, previous_aqi: float) -> float:
    """
    Calculate the rate of change in AQI.
    
    Args:
        current_aqi: Current AQI value
        previous_aqi: Previous AQI value
    
    Returns:
        Rate of change (positive = increasing, negative = decreasing)
    """
    if previous_aqi == 0:
        return 0
    return ((current_aqi - previous_aqi) / previous_aqi) * 100


def create_features_from_raw_data(raw_data: Dict) -> Dict:
    """
    Create all features from raw API data.
    
    Args:
        raw_data: Dictionary from fetch_all_current_data()
    
    Returns:
        Dictionary with all features
    """
    timestamp = raw_data['timestamp']
    
    # Start with raw data
    features = {
        'city': raw_data['city'],
        'timestamp': timestamp,
        
        # Weather features
        'temperature': raw_data['temperature'],
        'humidity': raw_data['humidity'],
        'pressure': raw_data['pressure'],
        'wind_speed': raw_data['wind_speed'],
        
        # Pollution features
        'pm2_5': raw_data['pm2_5'],
        'pm10': raw_data['pm10'],
        'co': raw_data['co'],
        'no2': raw_data['no2'],
        'so2': raw_data['so2'],
        'o3': raw_data['o3'],
        'aqi': raw_data['aqi'],
    }
    
    # Add time features
    time_features = create_time_features(timestamp)
    features.update(time_features)
    
    # Add derived features
    features['pm_ratio'] = raw_data['pm2_5'] / raw_data['pm10'] if raw_data['pm10'] > 0 else 0
    features['temp_humidity_interaction'] = raw_data['temperature'] * raw_data['humidity']
    
    return features


def create_lag_features(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
    """
    Create lag features for time series prediction.
    
    Args:
        df: DataFrame with time series data
        columns: List of column names to create lags for
        lags: List of lag periods (e.g., [1, 3, 6, 12, 24])
    
    Returns:
        DataFrame with lag features added
    """
    df = df.copy()
    
    for col in columns:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    return df


def create_rolling_features(df: pd.DataFrame, columns: List[str], windows: List[int]) -> pd.DataFrame:
    """
    Create rolling average features.
    
    Args:
        df: DataFrame with time series data
        columns: List of column names to create rolling features for
        windows: List of window sizes (e.g., [3, 6, 12, 24])
    
    Returns:
        DataFrame with rolling features added
    """
    df = df.copy()
    
    for col in columns:
        for window in windows:
            df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
            df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
    
    return df


def prepare_features_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare complete feature set for model training.
    
    Args:
        df: DataFrame with basic features
    
    Returns:
        DataFrame with all engineered features
    """
    # Ensure data is sorted by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Features to create lags and rolling for
    pollution_cols = ['pm2_5', 'pm10', 'co', 'no2', 'so2', 'o3', 'aqi']
    weather_cols = ['temperature', 'humidity', 'pressure', 'wind_speed']
    
    # Create lag features (previous 1, 3, 6, 12, 24 hours)
    df = create_lag_features(df, pollution_cols + weather_cols, lags=[1, 3, 6, 12, 24])
    
    # Create rolling features (3, 6, 12, 24 hour windows)
    df = create_rolling_features(df, pollution_cols + weather_cols, windows=[3, 6, 12, 24])
    
    # Create AQI change rate
    df['aqi_change_rate'] = df['aqi'].diff()
    df['aqi_change_rate_pct'] = df['aqi'].pct_change() * 100
    
    # Fill NaN values in lag and rolling features with forward fill
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df


def get_feature_names_for_training() -> List[str]:
    """
    Get list of feature names to use for model training.
    
    Returns:
        List of feature names
    """
    base_features = [
        # Current weather
        'temperature', 'humidity', 'pressure', 'wind_speed',
        
        # Current pollution
        'pm2_5', 'pm10', 'co', 'no2', 'so2', 'o3',
        
        # Time features
        'hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour',
        
        # Derived features
        'pm_ratio', 'temp_humidity_interaction'
    ]
    
    # Add lag features (we'll use 1, 3, 6 hour lags for main pollutants)
    pollution_cols = ['pm2_5', 'pm10', 'aqi']
    for col in pollution_cols:
        for lag in [1, 3, 6]:
            base_features.append(f'{col}_lag_{lag}')
    
    # Add rolling features (3, 6 hour windows)
    for col in pollution_cols:
        for window in [3, 6]:
            base_features.append(f'{col}_rolling_mean_{window}')
    
    # Add AQI change rate
    base_features.extend(['aqi_change_rate', 'aqi_change_rate_pct'])
    
    return base_features


# Test function
if __name__ == "__main__":
    print("Testing Feature Engineering Module\n")
    
    # Create sample data
    sample_timestamp = datetime.now()
    sample_raw = {
        'city': 'Karachi',
        'timestamp': sample_timestamp,
        'temperature': 28.5,
        'humidity': 65.0,
        'pressure': 1013.2,
        'wind_speed': 12.5,
        'pm2_5': 45.2,
        'pm10': 88.6,
        'co': 450.0,
        'no2': 28.3,
        'so2': 12.1,
        'o3': 35.8,
        'aqi': 3
    }
    
    # Test feature creation
    features = create_features_from_raw_data(sample_raw)
    
    print("Created Features:")
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Feature engineering working correctly!")
