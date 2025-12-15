import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Weather Temperature Predictor",
    page_icon="🌡️",
    layout="wide"
)

# Load model artifacts
@st.cache_resource
def load_model_artifacts():
    model = joblib.load('data/models/best_model.pkl')
    scaler = joblib.load('data/models/scaler.pkl')
    feature_names = joblib.load('data/models/feature_names.pkl')
    with open('data/models/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    return model, scaler, feature_names, metadata

model, scaler, feature_names, metadata = load_model_artifacts()

# Load historical data
@st.cache_data
def load_historical_data():
    df = pd.read_csv('data/processed/featured_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

df_historical = load_historical_data()

# Title and description
st.title("🌡️ Weather Temperature Predictor")
st.markdown("""
Predicts tomorrow's temperature for Mexican cities using machine learning.
Built with historical weather data collected over 3+ weeks.
""")

# Sidebar - Model Info
with st.sidebar:
    st.header("📊 Model Information")
    st.metric("Model Type", metadata['model_type'])
    st.metric("Test MAE", f"{metadata['test_mae']:.2f}°C")
    st.metric("Test R²", f"{metadata['test_r2']:.3f}")
    st.metric("Training Date", metadata['date_trained'][:10])
    st.metric("Features Used", len(metadata['features']))
    
    st.markdown("---")
    st.markdown("""
    ### About
    This ML model predicts temperature 24 hours in advance
    based on current weather conditions and historical patterns.
    
    **Cities:**
    - Toluca
    - Mexico City
    - Guadalajara
    - Monterrey
    - Cancún
    """)

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Make Prediction", "📈 Historical Data", "📊 Model Performance", "ℹ️ About"])

# TAB 1: Make Prediction
with tab1:
    st.header("Make a Temperature Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Current Weather Conditions")
        
        # City selection
        cities = ['Toluca', 'Mexico City', 'Guadalajara', 'Monterrey', 'Cancun']
        selected_city = st.selectbox("Select City", cities)
        
        # Current conditions
        current_temp = st.slider("Current Temperature (°C)", 
                                min_value=-5.0, max_value=45.0, value=20.0, step=0.5)
        
        humidity = st.slider("Humidity (%)", 
                           min_value=0, max_value=100, value=50)
        
        pressure = st.slider("Pressure (hPa)", 
                           min_value=950, max_value=1050, value=1013)
        
        wind_speed = st.slider("Wind Speed (m/s)", 
                             min_value=0.0, max_value=20.0, value=3.0, step=0.1)
        
        cloudiness = st.slider("Cloudiness (%)", 
                             min_value=0, max_value=100, value=30)
    
    with col2:
        st.subheader("Additional Parameters")
        
        # Time features
        hour = st.selectbox("Hour of Day", list(range(24)), index=12)
        day_of_week = st.selectbox("Day of Week", 
                                   ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                    'Friday', 'Saturday', 'Sunday'])
        day_of_week_num = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                          'Friday', 'Saturday', 'Sunday'].index(day_of_week)
        
        # Historical patterns (simplified - user can adjust)
        st.markdown("**Recent Temperature Trends**")
        temp_6h_ago = st.number_input("Temperature 6h ago (°C)", 
                                     value=current_temp - 1.0, step=0.5)
        temp_12h_ago = st.number_input("Temperature 12h ago (°C)", 
                                      value=current_temp - 2.0, step=0.5)
        temp_24h_ago = st.number_input("Temperature 24h ago (°C)", 
                                      value=current_temp - 0.5, step=0.5)
    
    # Predict button
    if st.button("🔮 Predict Tomorrow's Temperature", type="primary"):
        # Prepare features
        features_dict = {
            'temperature_celsius': current_temp,
            'feels_like': current_temp - 1,  # Approximate
            'humidity_percent': humidity,
            'pressure_hpa': pressure,
            'wind_speed_mps': wind_speed,
            'cloudiness_percent': cloudiness,
            'hour': hour,
            'day_of_week': day_of_week_num,
            'day_of_month': datetime.now().day,
            'month': datetime.now().month,
            'is_weekend': 1 if day_of_week_num >= 5 else 0,
            'temp_lag_1': temp_6h_ago,
            'temp_lag_2': temp_12h_ago,
            'temp_lag_4': temp_24h_ago,
            'humidity_lag_1': humidity,
            'pressure_lag_1': pressure,
            'temp_rolling_mean_24h': np.mean([current_temp, temp_6h_ago, temp_12h_ago, temp_24h_ago]),
            'temp_rolling_std_24h': np.std([current_temp, temp_6h_ago, temp_12h_ago, temp_24h_ago]),
            'temp_rolling_max_24h': max([current_temp, temp_6h_ago, temp_12h_ago, temp_24h_ago]),
            'temp_rolling_min_24h': min([current_temp, temp_6h_ago, temp_12h_ago, temp_24h_ago]),
            'temp_change_6h': current_temp - temp_6h_ago,
        }
        
        # Add city one-hot encoding
        for city in cities:
            city_col = f"city_{city.replace(' ', '_')}"
            features_dict[city_col] = 1 if city == selected_city else 0
        
        # Create feature array in correct order
        feature_values = []
        for feat in feature_names:
            if feat in features_dict:
                feature_values.append(features_dict[feat])
            else:
                feature_values.append(0)  # Default value for missing features
        
        X_input = np.array(feature_values).reshape(1, -1)
        X_input_scaled = scaler.transform(X_input)
        
        # Make prediction
        prediction = model.predict(X_input_scaled)[0]
        
        # Display result
        st.success("✅ Prediction Complete!")
        
        # Big prediction display
        st.markdown("---")
        col_pred1, col_pred2, col_pred3 = st.columns(3)
        
        with col_pred1:
            st.metric("Current Temperature", f"{current_temp:.1f}°C")
        
        with col_pred2:
            st.metric("🔮 Predicted Temperature (24h)", 
                     f"{prediction:.1f}°C",
                     delta=f"{prediction - current_temp:+.1f}°C")
        
        with col_pred3:
            temp_change = prediction - current_temp
            if temp_change > 0:
                st.metric("Trend", "🔺 Warming", f"+{temp_change:.1f}°C")
            else:
                st.metric("Trend", "🔻 Cooling", f"{temp_change:.1f}°C")
        
        # Visualization
        st.markdown("---")
        st.subheader("Temperature Forecast")
        
        # Create timeline visualization
        times = ['Now', '6h', '12h', '18h', '24h (Predicted)']
        temps = [current_temp, 
                (current_temp + temp_6h_ago) / 2,
                temp_6h_ago,
                (temp_6h_ago + prediction) / 2,
                prediction]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times, 
            y=temps,
            mode='lines+markers',
            name='Temperature',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=12)
        ))
        
        fig.update_layout(
            title=f"Temperature Forecast for {selected_city}",
            xaxis_title="Time",
            yaxis_title="Temperature (°C)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Confidence info
        st.info(f"""
        **Model Confidence:** This prediction has an expected error of ±{metadata['test_mae']:.2f}°C 
        based on the model's performance on test data.
        """)

# TAB 2: Historical Data
with tab2:
    st.header("📈 Historical Weather Data")
    
    # City filter
    city_filter = st.multiselect(
        "Filter by City",
        options=df_historical['timestamp'].apply(lambda x: 'All cities').unique().tolist() + cities,
        default=['All cities']
    )
    
    # Temperature trend chart
    fig = go.Figure()
    
    for city in cities:
        city_col = f"city_{city.replace(' ', '_')}"
        if city_col in df_historical.columns:
            city_data = df_historical[df_historical[city_col] == 1].copy()
            city_data = city_data.sort_values('timestamp')
            
            fig.add_trace(go.Scatter(
                x=city_data['timestamp'],
                y=city_data['temperature_celsius'],
                mode='lines+markers',
                name=city,
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Date: %{x}<br>' +
                             'Temp: %{y:.1f}°C<br>' +
                             '<extra></extra>'
            ))
    
    fig.update_layout(
        title="Temperature Trends by City",
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.subheader("Temperature Statistics by City")
    
    stats_data = []
    for city in cities:
        city_col = f"city_{city.replace(' ', '_')}"
        if city_col in df_historical.columns:
            city_data = df_historical[df_historical[city_col] == 1]
            stats_data.append({
                'City': city,
                'Avg Temp': f"{city_data['temperature_celsius'].mean():.1f}°C",
                'Min Temp': f"{city_data['temperature_celsius'].min():.1f}°C",
                'Max Temp': f"{city_data['temperature_celsius'].max():.1f}°C",
                'Std Dev': f"{city_data['temperature_celsius'].std():.1f}°C",
                'Records': len(city_data)
            })
    
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

# TAB 3: Model Performance
with tab3:
    st.header("📊 Model Performance Analysis")
    
    # Load evaluation results
    st.subheader("Model Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mean Absolute Error (MAE)", f"{metadata['test_mae']:.2f}°C")
    with col2:
        st.metric("R² Score", f"{metadata['test_r2']:.3f}")
    with col3:
        st.metric("Model Type", metadata['model_type'])
    
    st.markdown("---")
    
    # Display evaluation plots
    st.subheader("Evaluation Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            st.image('/data/actual_vs_predicted.png', 
                    caption='Actual vs Predicted Temperature',
                    use_column_width=True)
        except:
            st.info("Run model evaluation notebook to generate this plot")
    
    with col2:
        try:
            st.image('/data/error_distribution.png',
                    caption='Prediction Error Distribution',
                    use_column_width=True)
        except:
            st.info("Run model evaluation notebook to generate this plot")
    
    st.markdown("---")
    
    # Model comparison
    st.subheader("Model Comparison")
    try:
        st.image('/data/model_comparison.png',
                caption='Comparison of Different ML Models',
                use_column_width=True)
    except:
        st.info("Run model training notebook to generate this plot")
    
    # Feature importance
    st.markdown("---")
    st.subheader("Feature Importance")
    try:
        st.image('/data/feature_importance.png',
                caption='Top Important Features for Prediction',
                use_column_width=True)
    except:
        st.info("Run feature engineering notebook to generate this plot")

# TAB 4: About
with tab4:
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ## Weather Temperature Prediction System
    
    ### Project Overview
    This machine learning application predicts tomorrow's temperature for major Mexican cities 
    based on current weather conditions and historical patterns.
    
    ### How It Works
    
    1. **Data Collection**: Weather data is automatically collected every 6 hours using an AWS Lambda 
       pipeline connected to the OpenWeatherMap API.
    
    2. **Feature Engineering**: The system creates time-series features including:
       - Lag features (previous temperatures)
       - Rolling statistics (trends over 24 hours)
       - Time-based features (hour, day, month)
       - Weather conditions (humidity, pressure, wind)
    
    3. **Model Training**: Multiple machine learning algorithms are trained and compared:
       - Linear Regression
       - Ridge & Lasso Regression
       - Decision Trees
       - Random Forests
       - Gradient Boosting
    
    4. **Prediction**: The best-performing model makes predictions 24 hours in advance.
    
    ### Technologies Used
    
    **Data Pipeline:**
    - AWS Lambda (serverless compute)
    - AWS S3 (data storage)
    - AWS EventBridge (scheduling)
    - OpenWeatherMap API (data source)
    
    **Machine Learning:**
    - Python 3.11
    - scikit-learn (ML algorithms)
    - pandas & NumPy (data processing)
    - joblib (model serialization)
    
    **Web Application:**
    - Streamlit (web framework)
    - Plotly (interactive visualizations)
    
    ### Model Performance
    
    The model achieves:
    - **Mean Absolute Error**: ~{:.2f}°C
    - **R² Score**: ~{:.3f}
    
    This means predictions are typically within ±{:.2f}°C of the actual temperature.
    
    ### Data Sources
    
    - **Primary**: OpenWeatherMap API 2.5 (free tier)
    - **Cities Tracked**: Toluca, Mexico City, Guadalajara, Monterrey, Cancún
    - **Collection Frequency**: Every 6 hours (4 times daily)
    - **Historical Data**: 3+ weeks of continuous collection
    
    ### Future Improvements
    
    - 📊 Add more cities and countries
    - 🤖 Implement deep learning models (LSTM for time-series)
    - 📧 Email alerts for extreme temperature changes
    - 📱 Mobile app version
    - 🌍 Multi-day forecasts (3-7 days ahead)
    - 📈 Incorporate additional weather variables
    
    ### Author
    
    **Eduardo Arriaga Alejandre**
    
    Telematics Engineer building data science and ML engineering skills.
    
    - [GitHub](https://github.com/PraetorClaudius)
    - [LinkedIn](https://www.linkedin.com/in/eduardo-arriaga-230156295/)
    
    ### Project Repository
    
    Full source code, notebooks, and documentation available on GitHub:
    https://github.com/PraetorClaudius/weather-ml-prediction
    
    ---
    
    *Built as part of a data science portfolio to demonstrate end-to-end ML project skills.*
    """.format(metadata['test_mae'], metadata['test_r2'], metadata['test_mae']))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🌡️ Weather Temperature Predictor | Built with Streamlit & scikit-learn</p>
    <p>Data updated every 6 hours | Model trained on 3+ weeks of historical data</p>
</div>
""", unsafe_allow_html=True)