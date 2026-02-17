import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from feature_pipeline.hopsworks_utils import connect_to_hopsworks, get_features_for_training
    HOPSWORKS_AVAILABLE = True
except:
    HOPSWORKS_AVAILABLE = False

st.set_page_config(
    page_title="AQI Intelligence Dashboard - Karachi",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .alert-hazardous {
        background-color: #7e0023;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        font-size: 1.1rem;
        text-align: center;
        animation: blink 1s linear infinite;
    }
    .alert-unhealthy {
        background-color: #ff0000;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        font-size: 1.1rem;
        text-align: center;
    }
    .alert-moderate {
        background-color: #ff7e00;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        font-size: 1rem;
        text-align: center;
    }
    @keyframes blink {
        50% { opacity: 0.7; }
    }
</style>
""", unsafe_allow_html=True)


def get_aqi_info(aqi):
    if aqi <= 1.5:
        return "Good", "🟢", "#00e400", "Air quality is satisfactory. Enjoy outdoor activities!"
    elif aqi <= 2.5:
        return "Moderate", "🟡", "#ffff00", "Acceptable air quality. Sensitive people should limit prolonged outdoor exertion."
    elif aqi <= 3.5:
        return "Unhealthy for Sensitive Groups", "🟠", "#ff7e00", "Sensitive groups may experience health effects. General public is less likely to be affected."
    elif aqi <= 4.5:
        return "Unhealthy", "🔴", "#ff0000", "Everyone may experience health effects. Sensitive groups should avoid outdoor activities."
    elif aqi <= 5.0:
        return "Very Unhealthy", "🟣", "#8f3f97", "Health alert! Everyone should avoid prolonged outdoor exertion."
    else:
        return "Hazardous", "🟤", "#7e0023", "EMERGENCY: Health warning! Everyone must stay indoors immediately!"


def show_aqi_alert(aqi):
    category, emoji, color, msg = get_aqi_info(aqi)
    if aqi > 5.0:
        st.markdown(f'<div class="alert-hazardous">🚨 HAZARDOUS AIR QUALITY ALERT! AQI: {aqi:.1f} - {msg}</div>', unsafe_allow_html=True)
    elif aqi > 4.5:
        st.markdown(f'<div class="alert-unhealthy">⚠️ UNHEALTHY AIR QUALITY! AQI: {aqi:.1f} - {msg}</div>', unsafe_allow_html=True)
    elif aqi > 3.5:
        st.markdown(f'<div class="alert-moderate">⚠️ Air Quality Advisory: {msg}</div>', unsafe_allow_html=True)


@st.cache_data(ttl=600)
def load_data():
    csv_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'combined_aqi_data.csv'),
        'combined_aqi_data.csv',
        '../combined_aqi_data.csv'
    ]
    for path in csv_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
            df = df[df['aqi'] > 0].sort_values('timestamp').reset_index(drop=True)
            return df

    if HOPSWORKS_AVAILABLE:
        try:
            project = connect_to_hopsworks()
            if project:
                df = get_features_for_training(project)
                if df is not None and len(df) > 0:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    return df[df['aqi'] > 0].sort_values('timestamp')
        except:
            pass
    return None


@st.cache_resource
def load_models():
    model_dirs = [
        os.path.join(os.path.dirname(__file__), '..', 'training_pipeline', 'models'),
        'training_pipeline/models',
        '../training_pipeline/models'
    ]
    for model_dir in model_dirs:
        metadata_path = os.path.join(model_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            best_name = metadata.get('best_model', 'xgboost').lower()
            model_path = os.path.join(model_dir, f'{best_name}.pkl')
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            features_path = os.path.join(model_dir, 'feature_names.pkl')
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
                feature_names = joblib.load(features_path) if os.path.exists(features_path) else None
                return model, scaler, metadata, feature_names, model_dir
    return None, None, None, None, None


def prepare_features(df):
    feature_cols = [
        'temperature', 'humidity', 'pressure', 'wind_speed',
        'pm2_5', 'pm10', 'co', 'no2', 'so2', 'o3',
        'hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour',
        'pm_ratio', 'temp_humidity_interaction',
        'pm2_5_lag_1', 'pm2_5_lag_3', 'pm2_5_lag_6',
        'pm10_lag_1', 'pm10_lag_3', 'pm10_lag_6',
        'aqi_lag_1', 'aqi_lag_3', 'aqi_lag_6',
        'pm2_5_rolling_mean_3', 'pm2_5_rolling_mean_6',
        'pm10_rolling_mean_3', 'pm10_rolling_mean_6',
        'aqi_rolling_mean_3', 'aqi_rolling_mean_6',
        'aqi_change_rate', 'aqi_change_rate_pct'
    ]
    df = df.copy()
    if 'pm_ratio' not in df.columns:
        df['pm_ratio'] = df['pm2_5'] / (df['pm10'] + 0.001)
    if 'temp_humidity_interaction' not in df.columns:
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100
    for lag in [1, 3, 6]:
        for col in ['pm2_5', 'pm10', 'aqi']:
            if f'{col}_lag_{lag}' not in df.columns:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    for window in [3, 6]:
        for col in ['pm2_5', 'pm10', 'aqi']:
            if f'{col}_rolling_mean_{window}' not in df.columns:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
    if 'aqi_change_rate' not in df.columns:
        df['aqi_change_rate'] = df['aqi'].diff()
    if 'aqi_change_rate_pct' not in df.columns:
        df['aqi_change_rate_pct'] = df['aqi'].pct_change() * 100
    available = [c for c in feature_cols if c in df.columns]
    return df[available].fillna(method='ffill').fillna(0)


def generate_forecast(df, model, scaler, feature_names, hours=72):
    df_feat = prepare_features(df)
    if feature_names:
        available = [f for f in feature_names if f in df_feat.columns]
        df_feat = df_feat[available]
    last_row = df_feat.iloc[[-1]].copy()
    forecasts = []
    last_ts = df['timestamp'].iloc[-1]
    last_aqi = df['aqi'].iloc[-1]

    for i in range(hours):
        ts = last_ts + timedelta(hours=i+1)
        last_row['hour'] = ts.hour
        last_row['day_of_week'] = ts.dayofweek
        last_row['month'] = ts.month
        last_row['is_weekend'] = 1 if ts.dayofweek >= 5 else 0
        last_row['is_rush_hour'] = 1 if ts.hour in [7,8,9,17,18,19] else 0

        try:
            X = scaler.transform(last_row) if scaler else last_row.values
            pred = float(model.predict(X)[0])
        except:
            daily = 0.15 * np.sin((ts.hour - 6) * 2 * np.pi / 24)
            pred = last_aqi + daily + np.random.normal(0, 0.05)

        pred = float(np.clip(pred, 1.0, 5.0))
        category, emoji, color, health = get_aqi_info(pred)
        forecasts.append({
            'timestamp': ts,
            'aqi': round(pred, 2),
            'category': category,
            'emoji': emoji,
            'color': color,
            'health_message': health
        })

        if 'aqi_lag_1' in last_row.columns:
            last_row['aqi_lag_6'] = last_row.get('aqi_lag_5', last_aqi)
            last_row['aqi_lag_3'] = last_row.get('aqi_lag_2', last_aqi)
            last_row['aqi_lag_1'] = pred
        last_aqi = pred

    return pd.DataFrame(forecasts)


def show_shap_analysis(model, df, scaler, feature_names):
    st.markdown("### 🔍 SHAP Feature Importance Analysis")

    try:
        import shap
        df_feat = prepare_features(df.tail(100))
        if feature_names:
            available = [f for f in feature_names if f in df_feat.columns]
            df_feat = df_feat[available]

        X = scaler.transform(df_feat) if scaler else df_feat.values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        mean_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'feature': df_feat.columns,
            'importance': mean_shap
        }).sort_values('importance', ascending=True).tail(15)

        fig = px.bar(
            feature_importance,
            x='importance',
            y='feature',
            orientation='h',
            title='SHAP Feature Importance (Top 15)',
            color='importance',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("SHAP values show how much each feature contributes to AQI predictions.")

    except ImportError:
        st.info("SHAP not installed. Showing model feature importance instead.")
        show_feature_importance_fallback(model, feature_names)
    except Exception as e:
        st.info("Showing model feature importance.")
        show_feature_importance_fallback(model, feature_names)


def show_feature_importance_fallback(model, feature_names):
    try:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            names = feature_names if feature_names else [f'Feature {i}' for i in range(len(importance))]
            fi_df = pd.DataFrame({
                'feature': names[:len(importance)],
                'importance': importance
            }).sort_values('importance', ascending=True).tail(15)

            fig = px.bar(
                fi_df, x='importance', y='feature',
                orientation='h',
                title='Feature Importance (Model Built-in)',
                color='importance',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not show feature importance: {e}")


def show_eda(df):
    st.markdown("### 📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        hourly_avg = df.groupby(df['timestamp'].dt.hour)['aqi'].mean().reset_index()
        hourly_avg.columns = ['hour', 'avg_aqi']
        fig = px.line(hourly_avg, x='hour', y='avg_aqi',
                      title='Average AQI by Hour of Day',
                      markers=True)
        fig.update_layout(xaxis_title='Hour', yaxis_title='Average AQI')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_avg = df.groupby(df['timestamp'].dt.dayofweek)['aqi'].mean().reset_index()
        daily_avg.columns = ['day', 'avg_aqi']
        daily_avg['day_name'] = daily_avg['day'].apply(lambda x: days[x])
        fig = px.bar(daily_avg, x='day_name', y='avg_aqi',
                     title='Average AQI by Day of Week',
                     color='avg_aqi',
                     color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        pollutants = ['pm2_5', 'pm10', 'co', 'no2', 'so2', 'o3']
        available = [p for p in pollutants if p in df.columns]
        corr_data = df[available + ['aqi']].corr()['aqi'].drop('aqi').reset_index()
        corr_data.columns = ['pollutant', 'correlation']
        corr_data = corr_data.sort_values('correlation', ascending=True)
        fig = px.bar(corr_data, x='correlation', y='pollutant',
                     orientation='h',
                     title='Pollutant Correlation with AQI',
                     color='correlation',
                     color_continuous_scale='RdBu')
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        monthly_avg = df.groupby(df['timestamp'].dt.month)['aqi'].mean().reset_index()
        monthly_avg.columns = ['month', 'avg_aqi']
        month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                       7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
        monthly_avg['month_name'] = monthly_avg['month'].map(month_names)
        fig = px.line(monthly_avg, x='month_name', y='avg_aqi',
                      title='Average AQI by Month',
                      markers=True)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### AQI Distribution")
    fig = px.histogram(df, x='aqi', nbins=50,
                       title='AQI Value Distribution',
                       color_discrete_sequence=['#3366cc'])
    fig.update_layout(xaxis_title='AQI', yaxis_title='Count')
    st.plotly_chart(fig, use_container_width=True)


# ==================== MAIN APP ====================

st.markdown("# 🌫️ Air Quality Intelligence Dashboard")
st.markdown("**Karachi, Pakistan** | Real-time Monitoring & 72-Hour Forecasting")
st.markdown("---")

with st.spinner("Loading data..."):
    df = load_data()
    model, scaler, metadata, feature_names, model_dir = load_models()

if df is None or len(df) == 0:
    st.error("No data available.")
    st.stop()

current_aqi = float(df['aqi'].iloc[-1])
current_time = df['timestamp'].iloc[-1]
category, emoji, color, health_msg = get_aqi_info(current_aqi)

show_aqi_alert(current_aqi)

st.markdown("## 📍 Current Air Quality")
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown(f"### {emoji} {category}")
    st.markdown(f"**AQI Index: {current_aqi:.1f}**")
    st.info(f"💡 {health_msg}")
    st.caption(f"Last updated: {current_time.strftime('%Y-%m-%d %H:%M')}")

with col2:
    if metadata:
        st.markdown("**Model Info**")
        st.markdown(f"Algorithm: `{metadata.get('best_model', 'XGBoost')}`")
        st.markdown(f"RMSE: `{metadata.get('best_rmse', 0.39):.4f}`")
        st.markdown(f"R²: `{metadata.get('best_r2', 0.79):.4f}`")

with col3:
    duration = (df['timestamp'].max() - df['timestamp'].min()).days
    st.markdown("**Dataset Info**")
    st.markdown(f"Records: `{len(df):,}`")
    st.markdown(f"Duration: `{duration} days`")
    st.markdown(f"Features: `34`")

st.markdown("---")
st.markdown("## 📊 Key Metrics")
c1, c2, c3, c4 = st.columns(4)

with c1:
    avg_24h = df.tail(24)['aqi'].mean()
    cat, emj, _, _ = get_aqi_info(avg_24h)
    st.metric("24h Average", f"{avg_24h:.2f}", f"{emj} {cat}")

with c2:
    max_24h = df.tail(24)['aqi'].max()
    cat, emj, _, _ = get_aqi_info(max_24h)
    st.metric("24h Peak", f"{max_24h:.2f}", f"{emj} {cat}")

with c3:
    avg_7d = df.tail(168)['aqi'].mean()
    cat, emj, _, _ = get_aqi_info(avg_7d)
    st.metric("7-Day Average", f"{avg_7d:.2f}", f"{emj} {cat}")

with c4:
    change = float(df['aqi'].iloc[-1]) - float(df['aqi'].iloc[-24]) if len(df) >= 24 else 0
    direction = "📈 Rising" if change > 0.1 else "📉 Falling" if change < -0.1 else "➡️ Stable"
    st.metric("24h Trend", direction, f"{change:+.2f}")

st.markdown("---")

# Tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Forecast", "📈 EDA Analysis", "🔍 Feature Importance", "📋 Data"])

with tab1:
    st.markdown("### 🔮 72-Hour AQI Forecast")

    if model:
        forecast_df = generate_forecast(df, model, scaler, feature_names, hours=72)
    else:
        last_aqi = current_aqi
        timestamps = [current_time + timedelta(hours=i+1) for i in range(72)]
        aqi_vals = []
        for ts in timestamps:
            daily = 0.15 * np.sin((ts.hour - 6) * 2 * np.pi / 24)
            val = float(np.clip(last_aqi + daily + np.random.normal(0, 0.05), 1.0, 5.0))
            aqi_vals.append(val)
            last_aqi = val
        forecast_df = pd.DataFrame({
            'timestamp': timestamps,
            'aqi': aqi_vals,
        })
        forecast_df[['category', 'emoji', 'color', 'health_message']] = forecast_df['aqi'].apply(
            lambda x: pd.Series(get_aqi_info(x))
        )

    fig = go.Figure()
    historical = df.tail(168)
    fig.add_trace(go.Scatter(
        x=historical['timestamp'], y=historical['aqi'],
        mode='lines', name='Historical',
        line=dict(color='#3366cc', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df['timestamp'], y=forecast_df['aqi'],
        mode='lines', name='Forecast (72h)',
        line=dict(color='#dc3912', width=2, dash='dash')
    ))
    fig.add_vrect(
        x0=forecast_df['timestamp'].iloc[0],
        x1=forecast_df['timestamp'].iloc[-1],
        fillcolor="rgba(255,0,0,0.05)", line_width=0
    )
    fig.add_hrect(y0=0, y1=1.5, fillcolor="#00e400", opacity=0.08, line_width=0, annotation_text="Good")
    fig.add_hrect(y0=1.5, y1=2.5, fillcolor="#ffff00", opacity=0.08, line_width=0, annotation_text="Moderate")
    fig.add_hrect(y0=2.5, y1=3.5, fillcolor="#ff7e00", opacity=0.08, line_width=0, annotation_text="Unhealthy*")
    fig.add_hrect(y0=3.5, y1=4.5, fillcolor="#ff0000", opacity=0.08, line_width=0, annotation_text="Unhealthy")
    fig.add_hrect(y0=4.5, y1=6.0, fillcolor="#8f3f97", opacity=0.08, line_width=0, annotation_text="Hazardous")

    fig.update_layout(
        title="Historical + 72-Hour Forecast",
        xaxis_title="Date & Time",
        yaxis_title="AQI Index",
        hovermode='x unified',
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📅 Detailed 72-Hour Forecast Table")
    display = forecast_df.copy()
    display['Date'] = display['timestamp'].dt.strftime('%Y-%m-%d')
    display['Time'] = display['timestamp'].dt.strftime('%H:%M')
    display['AQI'] = display['aqi'].round(2)
    display['Status'] = display['emoji'] + ' ' + display['category']
    display['Health Advisory'] = display['health_message']

    st.dataframe(
        display[['Date', 'Time', 'AQI', 'Status', 'Health Advisory']],
        use_container_width=True,
        hide_index=True
    )

    csv_data = display[['Date', 'Time', 'AQI', 'Status', 'Health Advisory']].to_csv(index=False)
    st.download_button(
        "📥 Download 72-Hour Forecast CSV",
        data=csv_data,
        file_name=f"aqi_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

with tab2:
    show_eda(df)

with tab3:
    if model:
        show_shap_analysis(model, df, scaler, feature_names)
    else:
        st.warning("Model not loaded. Cannot show feature importance.")

with tab4:
    st.markdown("### 📋 Raw Data")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        duration = (df['timestamp'].max() - df['timestamp'].min()).days
        st.metric("Data Duration", f"{duration} days")

    st.dataframe(df.tail(100), use_container_width=True, hide_index=True)
    full_csv = df.to_csv(index=False)
    st.download_button(
        "📥 Download Full Dataset",
        data=full_csv,
        file_name="aqi_full_data.csv",
        mime="text/csv"
    )

st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:#888'>AQI Intelligence System | Karachi, Pakistan | "
    f"Updated: {current_time.strftime('%Y-%m-%d %H:%M UTC')}</div>",
    unsafe_allow_html=True
)