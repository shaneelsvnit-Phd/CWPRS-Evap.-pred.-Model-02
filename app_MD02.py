import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="SHANEEL Evaporation Predictor", 
    page_icon="💧", 
    layout="wide"
)

# --- 2. Custom CSS for Styling (Optional, to mimic the React look) ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .big-value {
        font-size: 3rem;
        font-weight: 800;
        color: #2563eb;
    }
    .big-value-cyan {
        font-size: 2.5rem;
        font-weight: 800;
        color: #0891b2;
    }
    .unit {
        font-size: 1.2rem;
        font-weight: 500;
        color: #94a3b8;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. Sidebar - Input Parameters ---
st.sidebar.header("⚙️ Input Parameters")

temp = st.sidebar.slider("Air Temperature (°C)", min_value=10.0, max_value=50.0, value=30.0, step=1.0)
humidity = st.sidebar.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=40.0, step=1.0)
wind = st.sidebar.slider("Wind Speed (km/h)", min_value=0.0, max_value=50.0, value=15.0, step=1.0)
solar = st.sidebar.slider("Solar Radiation (kWh/m²)", min_value=0.0, max_value=12.0, value=6.0, step=0.5)
area = st.sidebar.slider("Reservoir Area (km²)", min_value=1.0, max_value=50.0, value=5.0, step=1.0)

# --- 4. Logic & Calculations ---
# Formula: Math.max(0.1, (Temp * 0.15) - (Humidity * 0.05) + (Wind * 0.05) + (Solar * 0.2))
raw_loss = (temp * 0.15) - (humidity * 0.05) + (wind * 0.05) + (solar * 0.2)
daily_loss = max(0.1, raw_loss)
total_volume_loss = daily_loss * area

# Determine Intensity Status
if daily_loss > 4:
    status = "HIGH"
    status_color = "red"
elif daily_loss > 2.5:
    status = "MODERATE"
    status_color = "orange"
else:
    status = "LOW"
    status_color = "green"

# --- 5. Main App Interface ---
st.title("💧 SHANEEL Evaporation Predictor")
st.markdown("**AI-Powered Hydrological Forecasting (Interactive Simulation)**")
st.divider()

# Layout Columns
col1, col2 = st.columns(2)

with col1:
    # Predicted Daily Loss Card
    st.markdown(f"""
        <div class="metric-card">
            <h3 style="color:#475569; margin-top:0;">Predicted Daily Loss</h3>
            <div class="big-value">{daily_loss:.2f} <span class="unit">mm/day</span></div>
        </div>
    """, unsafe_allow_html=True)

    # Total Volumetric Loss Card
    st.markdown(f"""
        <div class="metric-card">
            <h3 style="color:#475569; margin-top:0;">Total Volumetric Loss</h3>
            <div class="big-value-cyan">{total_volume_loss:.2f} <span class="unit">Million Liters</span></div>
            <p style="color:#94a3b8; font-size:0.8rem; margin-bottom:0;">Based on current surface area.</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    # Visual Representation / Status Card
    st.markdown(f"""
        <div class="metric-card" style="text-align: center; min-height: 280px; display: flex; flex-direction: column; justify-content: center;">
            <h3 style="color:#475569; margin-top:0;">Evaporation Intensity</h3>
            <h1 style="font-size: 5rem; margin: 10px 0;">{ "☀️" if status == "HIGH" else "⛅" if status == "MODERATE" else "☁️"}</h1>
            <p style="font-size: 1.2rem; color: #64748b;">
                Intensity Status: <strong style="color: {status_color};">{status}</strong>
            </p>
        </div>
    """, unsafe_allow_html=True)
