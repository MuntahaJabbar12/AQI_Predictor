"""
Data Fetching Module
====================
This module fetches weather and air quality data from external APIs.

APIs Used:
1. Open-Meteo: Free weather data (no API key needed)
2. OpenWeather: Air pollution data (free tier: 1,000 calls/day)
"""

import os
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional
import time

# Load environment variables (optional - GitHub Actions provides them directly)
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass  # In GitHub Actions, env vars are already available

# Configuration
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
LATITUDE = float(os.getenv('LATITUDE', 24.8607))
LONGITUDE = float(os.getenv('LONGITUDE', 67.0011))
CITY_NAME = os.getenv('CITY_NAME', 'Karachi')


def fetch_weather_data(lat: float = LATITUDE, lon: float = LONGITUDE) -> Optional[Dict]:
    """
    Fetch current weather data from Open-Meteo API.
    
    Args:
        lat: Latitude
        lon: Longitude
    
    Returns:
        Dictionary with weather data or None if error
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m",
            "timezone": "auto"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        current = data.get('current', {})
        
        weather_data = {
            'timestamp': datetime.fromisoformat(current.get('time')),
            'temperature': current.get('temperature_2m'),
            'humidity': current.get('relative_humidity_2m'),
            'pressure': current.get('pressure_msl'),
            'wind_speed': current.get('wind_speed_10m')
        }
        
        print(f"✓ Weather data fetched successfully for {CITY_NAME}")
        return weather_data
        
    except Exception as e:
        print(f"✗ Error fetching weather data: {str(e)}")
        return None


def fetch_pollution_data(lat: float = LATITUDE, lon: float = LONGITUDE) -> Optional[Dict]:
    """
    Fetch current air pollution data from OpenWeather API.
    
    Args:
        lat: Latitude
        lon: Longitude
    
    Returns:
        Dictionary with pollution data or None if error
    """
    if not OPENWEATHER_API_KEY or OPENWEATHER_API_KEY == 'your_openweather_api_key_here':
        print("✗ OpenWeather API key not configured. Please add it to .env file")
        return None
    
    try:
        url = "http://api.openweathermap.org/data/2.5/air_pollution"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": OPENWEATHER_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if 'list' not in data or len(data['list']) == 0:
            print("✗ No pollution data available")
            return None
        
        pollution = data['list'][0]
        components = pollution.get('components', {})
        aqi = pollution.get('main', {}).get('aqi', 0)
        
        pollution_data = {
            'timestamp': datetime.fromtimestamp(pollution.get('dt')),
            'aqi': aqi,
            'pm2_5': components.get('pm2_5', 0),
            'pm10': components.get('pm10', 0),
            'co': components.get('co', 0),
            'no2': components.get('no2', 0),
            'so2': components.get('so2', 0),
            'o3': components.get('o3', 0)
        }
        
        print(f"✓ Pollution data fetched successfully for {CITY_NAME} (AQI: {aqi})")
        return pollution_data
        
    except Exception as e:
        print(f"✗ Error fetching pollution data: {str(e)}")
        return None


def fetch_historical_weather(lat: float, lon: float, start_date: str, end_date: str) -> Optional[Dict]:
    """
    Fetch historical weather data from Open-Meteo API.
    
    Args:
        lat: Latitude
        lon: Longitude
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    
    Returns:
        Dictionary with historical weather data or None if error
    """
    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m",
            "timezone": "auto"
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        hourly = data.get('hourly', {})
        
        print(f"✓ Historical weather data fetched from {start_date} to {end_date}")
        return hourly
        
    except Exception as e:
        print(f"✗ Error fetching historical weather: {str(e)}")
        return None


def fetch_all_current_data() -> Optional[Dict]:
    """
    Fetch both weather and pollution data for current time.
    
    Returns:
        Combined dictionary with all data or None if error
    """
    print(f"\n{'='*50}")
    print(f"Fetching current data for {CITY_NAME}")
    print(f"Location: {LATITUDE}°N, {LONGITUDE}°E")
    print(f"{'='*50}\n")
    
    weather = fetch_weather_data()
    time.sleep(1)  # Rate limiting - be nice to APIs
    pollution = fetch_pollution_data()
    
    if weather is None or pollution is None:
        print("\n✗ Failed to fetch complete data")
        return None
    
    # Combine the data
    combined_data = {
        'city': CITY_NAME,
        'latitude': LATITUDE,
        'longitude': LONGITUDE,
        'timestamp': pollution['timestamp'],  # Use pollution timestamp as primary
        'temperature': weather['temperature'],
        'humidity': weather['humidity'],
        'pressure': weather['pressure'],
        'wind_speed': weather['wind_speed'],
        'aqi': pollution['aqi'],
        'pm2_5': pollution['pm2_5'],
        'pm10': pollution['pm10'],
        'co': pollution['co'],
        'no2': pollution['no2'],
        'so2': pollution['so2'],
        'o3': pollution['o3']
    }
    
    print(f"\n{'='*50}")
    print("✓ All data fetched successfully!")
    print(f"{'='*50}\n")
    
    return combined_data


# Test function
if __name__ == "__main__":
    print("Testing Data Fetching Module\n")
    
    # Test current data
    data = fetch_all_current_data()
    
    if data:
        print("\nSample Data:")
        print(f"Timestamp: {data['timestamp']}")
        print(f"Temperature: {data['temperature']}°C")
        print(f"Humidity: {data['humidity']}%")
        print(f"AQI: {data['aqi']}")
        print(f"PM2.5: {data['pm2_5']} μg/m³")
