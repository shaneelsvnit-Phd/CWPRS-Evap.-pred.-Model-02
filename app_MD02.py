import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# --- 1. Page Configuration ---
st.set_page_config(page_title="CWPRS Reservoir Evaporation Predictor", page_icon="💧", layout="wide")

# --- 2. Dummy Model Training ---
# In production, replace this function with: model = joblib.load('random_forest_model.pkl')
@st.cache_resource
def load_model():
    # Generating synthetic hydrological data for training [cite: 25]
    np.random.seed(42)
    X_train = pd.DataFrame({
        'Air_Temperature_C': np.random.uniform(10, 50, 1000),
        'Humidity_pct': np.random.uniform(10, 90, 1000),
        'Wind_Speed_kmh': np.random.uniform(0, 30, 1000),
        'Solar_Radiation_kWh': np.random.uniform(2, 8, 1000)
    })
    
    # Simulating non-linear relationships [cite: 37]
    y_train = (X_train['Air_Temperature_C'] * 0.15) - (X_train['Humidity_pct'] * 0.05) + (X_train['Wind_Speed_kmh'] * 0.05) + (X_train['Solar_Radiation_kWh'] * 0.2)
    y_train = np.maximum(y_train, 0.1) # Ensure no negative evaporation
    
    # Train the Random Forest model [cite: 26]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

rf_model = load_model()

# --- 3. Sidebar Inputs ---
st.sidebar.header("⚙️ Input Parameters")
# Setting inputs based on model parameters [cite: 30, 31, 32, 33, 34]
air_temp = st.sidebar.slider("Air Temperature (°C)", min_value=10.0, max_value=50.0, value=30.0, step=0.1)
humidity = st.sidebar.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=40.0, step=1.0)
wind_speed = st.sidebar.slider("Wind Speed (km/h)", min_value=0.0, max_value=50.0, value=15.0, step=0.1)
solar_rad = st.sidebar.slider("Solar Radiation (kWh/m²)", min_value=0.0, max_value=12.0, value=6.0, step=0.1)
surface_area = st.sidebar.number_input("Reservoir Surface Area (km²)", min_value=1.0, max_value=100.0, value=5.0, step=0.5)

# --- 4. Main App Interface ---
st.title("💧 CWPRS Reservoir Evaporation Predictor - CWPRS")
st.markdown("""
**AI-Powered Hydrological Forecasting** By Shaneel S. Sao, R.A. (Engineering) under Guidance of Dr. M Selva Balan & Smt. Anuja Rajgoplan.  
*Description: Uses a Random Forest Machine Learning model to estimate daily water loss based on meteorological parameters. Developed for CWPRS AI Project Implementation Cell.*
""")
st.divider()

# Organize layout into two columns 
col1, col2 = st.columns(2)

with col1:
    st.subheader("Current Conditions")
    # Displaying input data in a clean table format
    input_display = pd.DataFrame({
        "Value": [air_temp, humidity, wind_speed, solar_rad, surface_area]
    }, index=["Air_Temperature_C", "Humidity_pct", "Wind_Speed_kmh", "Solar_Radiation_kWh", "Surface_Area_km2"])
    st.dataframe(input_display, use_container_width=True)

with col2:
    st.subheader("Prediction")
    # Preparing data for the model
    input_data = pd.DataFrame({
        'Air_Temperature_C': [air_temp],
        'Humidity_pct': [humidity],
        'Wind_Speed_kmh': [wind_speed],
        'Solar_Radiation_kWh': [solar_rad]
    })

    if st.button("Calculate Loss", type="primary"):
        # Run prediction
        daily_loss_mm = rf_model.predict(input_data)[0]
        
        # Calculate Volumetric Loss (km² * mm = Million Liters)
        volumetric_loss_ml = surface_area * daily_loss_mm 

        st.success(f"Estimated Evaporation Loss: **{daily_loss_mm:.2f} mm/day**")
        st.info(f"Total Volumetric Loss: **{volumetric_loss_ml:.2f} Million Liters/day**")
