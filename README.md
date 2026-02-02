# 🌍 AQI Predictor - Karachi Air Quality Forecasting

Predict Air Quality Index (AQI) for the next 3 days in Karachi using machine learning and real-time data.

## 🎯 Project Overview

This project implements an end-to-end machine learning pipeline that:
- Fetches hourly weather and air quality data from APIs
- Stores features in Hopsworks Feature Store
- Trains ML models to predict AQI for next 72 hours
- Displays predictions on an interactive dashboard
- Runs automatically using GitHub Actions (hourly data collection, daily model training)

## 🏗️ Architecture

```
APIs (OpenWeather + Open-Meteo)
          ↓
Feature Pipeline (runs hourly)
          ↓
Hopsworks Feature Store
          ↓
Training Pipeline (runs daily)
          ↓
Model Registry (Hopsworks)
          ↓
Streamlit Dashboard
```

## 🛠️ Tech Stack

- **Language**: Python 3.9+
- **Feature Store**: Hopsworks
- **ML Libraries**: Scikit-learn, XGBoost, LightGBM
- **Dashboard**: Streamlit
- **Automation**: GitHub Actions
- **APIs**: OpenWeather API, Open-Meteo API

## 📋 Prerequisites

- Python 3.9 or higher
- Git
- OpenWeather API key (free tier)
- Hopsworks account (free tier)
- GitHub account

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/AQI_Predictor.git
cd AQI_Predictor
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
# OpenWeather API
OPENWEATHER_API_KEY=your_openweather_api_key_here

# Hopsworks
HOPSWORKS_API_KEY=your_hopsworks_api_key_here
HOPSWORKS_PROJECT_NAME=aqi_predictor

# Location
CITY_NAME=Karachi
LATITUDE=24.8607
LONGITUDE=67.0011
```

### 5. Run Feature Pipeline (First Time)

```bash
# Backfill historical data (last 60 days)
python feature_pipeline/backfill_features.py
```

### 6. Train Initial Model

```bash
python training_pipeline/train_model.py
```

### 7. Launch Dashboard

```bash
streamlit run app/dashboard.py
```

## 📁 Project Structure

```
AQI_Predictor/
│
├── notebooks/                    # Jupyter notebooks for EDA
│   └── eda_analysis.ipynb
│
├── feature_pipeline/             # Data collection and feature engineering
│   ├── __init__.py
│   ├── fetch_data.py            # Fetch from APIs
│   ├── feature_engineering.py   # Create features
│   ├── hopsworks_utils.py       # Hopsworks integration
│   ├── backfill_features.py     # Historical data collection
│   └── run_pipeline.py          # Hourly pipeline
│
├── training_pipeline/            # Model training
│   ├── __init__.py
│   ├── train_model.py           # Main training script
│   ├── evaluate_model.py        # Model evaluation
│   └── model_utils.py           # Helper functions
│
├── app/                          # Web dashboard
│   ├── dashboard.py             # Streamlit app
│   ├── components/              # UI components
│   └── utils.py                 # Helper functions
│
├── .github/                      # CI/CD workflows
│   └── workflows/
│       ├── feature_pipeline.yml # Hourly data collection
│       └── training_pipeline.yml# Daily model training
│
├── .env                          # Environment variables (not committed)
├── .gitignore                    # Git ignore file
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🔑 Getting API Keys

### OpenWeather API
1. Go to [OpenWeatherMap](https://openweathermap.org/api)
2. Sign up for a free account
3. Navigate to API Keys section
4. Copy your API key
5. Free tier: 1,000 calls/day

### Hopsworks
1. Go to [Hopsworks](https://app.hopsworks.ai/)
2. Sign up for free account
3. Create a new project: "aqi_predictor"
4. Go to Settings → API Keys
5. Generate new API key
6. Copy the key

## 📊 Features

### Input Features
- **Weather**: Temperature, Humidity, Pressure, Wind Speed
- **Pollutants**: PM2.5, PM10, CO, NO2, SO2, O3
- **Time**: Hour, Day of Week, Month, Is Weekend
- **Derived**: AQI change rate, Rolling averages, Lag features

### Target Variable
- **AQI**: Air Quality Index (0-500 scale)

## 🤖 Models

The project experiments with multiple models:
- Linear Regression (baseline)
- Random Forest Regressor
- Gradient Boosting (XGBoost, LightGBM)
- Ridge/Lasso Regression
- LSTM (optional, for time series)

## 📈 Evaluation Metrics

- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (R-squared)
- **MAPE** (Mean Absolute Percentage Error)

## 🔄 Automation

### Feature Pipeline (Hourly)
Runs every hour via GitHub Actions to:
- Fetch latest weather and pollution data
- Engineer features
- Store in Hopsworks

### Training Pipeline (Daily)
Runs daily at 2 AM to:
- Fetch updated training data
- Retrain models
- Evaluate and compare with previous best
- Update model in registry if improved

## 🎨 Dashboard Features

- Current AQI with health category
- 72-hour AQI forecast
- Historical trends visualization
- Feature importance (SHAP values)
- Health recommendations
- Pollutant breakdown

## 🧪 Testing Locally

```bash
# Test feature pipeline
python feature_pipeline/run_pipeline.py

# Test training pipeline
python training_pipeline/train_model.py

# Test dashboard
streamlit run app/dashboard.py
```

## 🚢 Deployment

### Dashboard Deployment (Streamlit Cloud)
1. Push code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your repository
4. Add secrets (API keys) in dashboard settings
5. Deploy!

## 📚 Resources

- [Project Documentation](docs/)
- [API Documentation](docs/api.md)
- [Model Training Report](docs/model_report.md)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

Your Name
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

## 🙏 Acknowledgments

- OpenWeather API for pollution data
- Open-Meteo for weather data
- Hopsworks for feature store
- Streamlit for dashboard framework

---

**Note**: This is an educational project demonstrating end-to-end ML pipeline development. Predictions should not be used for critical decision-making without proper validation.
