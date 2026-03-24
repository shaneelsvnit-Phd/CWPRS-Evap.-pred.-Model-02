import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

import streamlit as st

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="SHANEEL Evaporation Predictor", 
    page_icon="💧", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Custom CSS for Exact UI Matching ---
# This CSS mimics the Tailwind styling from the mock model
st.markdown("""
    <style>
    /* Main background and font */
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Card Styling */
    .ui-card {
        background-color: #ffffff;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        border: 1px solid #f1f5f9;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    
    /* Typography inside cards */
    .card-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: #475569;
        margin-bottom: 0.5rem;
    }
    .val-blue {
        font-size: 3rem;
        font-weight: 900;
        color: #2563eb;
        line-height: 1.2;
    }
    .val-cyan {
        font-size: 2.25rem;
        font-weight: 800;
        color: #0891b2;
        line-height: 1.2;
    }
    .unit-text {
        font-size: 1.25rem;
        font-weight: 500;
        color: #94a3b8;
    }
    .sub-text {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 0.5rem;
    }
    
    /* Water Reservoir Graphic Styling */
    .reservoir-container {
        width: 12rem;
        height: 12rem;
        border: 4px solid #cbd5e1;
        border-bottom-left-radius: 1.5rem;
        border-bottom-right-radius: 1.5rem;
        border-top-left-radius: 0.125rem;
        border-top-right-radius: 0.125rem;
        display: flex;
        align-items: flex-end;
        justify-content: center;
        background-color: #f0f9ff;
        margin: 0 auto;
        position: relative;
        overflow: hidden;
    }
    
    /* Hide default Streamlit top padding */
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. Sidebar - Input Parameters ---
st.sidebar.markdown("<h2 style='color: #1e3a8a; font-weight: bold;'>⚙️ Input Parameters</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

temp = st.sidebar.slider("Air Temperature (°C)", min_value=10.0, max_value=50.0, value=30.0, step=1.0)
humidity = st.sidebar.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=40.0, step=1.0)
wind = st.sidebar.slider("Wind Speed (km/h)", min_value=0.0, max_value=50.0, value=15.0, step=1.0)
solar = st.sidebar.slider("Solar Radiation (kWh/m²)", min_value=0.0, max_value=12.0, value=6.0, step=0.5)
area = st.sidebar.slider("Reservoir Area (km²)", min_value=1.0, max_value=50.0, value=5.0, step=1.0)

# --- 4. Logic & Mock Calculations ---
# Mock Formula: max(0.1, (Temp * 0.15) - (Humidity * 0.05) + (Wind * 0.05) + (Solar * 0.2))
raw_loss = (temp * 0.15) - (humidity * 0.05) + (wind * 0.05) + (solar * 0.2)
daily_loss = max(0.1, raw_loss)
total_volume_loss = daily_loss * area

# Determine Intensity Status & Visuals
if daily_loss > 4.0:
    status, color, water_level = "HIGH", "#ef4444", "30%" # Red, Low water
elif daily_loss > 2.5:
    status, color, water_level = "MODERATE", "#f97316", "55%" # Orange, Med water
else:
    status, color, water_level = "LOW", "#22c55e", "85%" # Green, High water

# --- 5. Main App Interface ---
st.markdown("""
    <h1 style='color: #0f172a; font-weight: 800; display: flex; align-items: center; gap: 10px;'>
        <span style='color: #3b82f6;'>💧</span> SHANEEL Evaporation Predictor
    </h1>
    <p style='color: #64748b; margin-top: -10px; margin-bottom: 20px;'>
        AI-Powered Hydrological Forecasting (Interactive Simulation)
    </p>
""", unsafe_allow_html=True)

# Layout: Two columns for the main content
col1, col2 = st.columns([1.2, 1])

with col1:
    # Card 1: Predicted Daily Loss
    st.markdown(f"""
        <div class="ui-card">
            <div style="position: absolute; top: 10px; right: 15px; font-size: 4rem; opacity: 0.05;">☀️</div>
            <div class="card-title">Predicted Daily Loss</div>
            <div class="val-blue">{daily_loss:.2f} <span class="unit-text">mm/day</span></div>
        </div>
    """, unsafe_allow_html=True)

    # Card 2: Total Volumetric Loss
    st.markdown(f"""
        <div class="ui-card">
            <div style="position: absolute; top: 10px; right: 15px; font-size: 4rem; opacity: 0.05;">🌊</div>
            <div class="card-title">Total Volumetric Loss</div>
            <div class="val-cyan">{total_volume_loss:.2f} <span class="unit-text" style="font-size: 1rem;">Million Liters</span></div>
            <div class="sub-text">Based on current surface area.</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    # Card 3: Visual Representation / Reservoir Graphic
    st.markdown(f"""
        <div class="ui-card" style="text-align: center; min-height: 320px; display: flex; flex-direction: column; justify-content: space-between;">
            <div class="card-title" style="text-align: left;">Evaporation Intensity</div>
            
            <div class="reservoir-container">
                <div style="position: absolute; top: 10%; width: 100%; display: flex; justify-content: space-evenly; color: #93c5fd; font-weight: bold; font-size: 1.5rem; opacity: {min(daily_loss / 5, 1)};">
                    <span>↑</span><span style="margin-top: 10px;">↑</span><span>↑</span>
                </div>
                
                <div style="width: 100%; height: {water_level}; background: linear-gradient(to top, #2563eb, #22d3ee); transition: height 0.5s ease-in-out;"></div>
            </div>
            
            <p style="margin-top: 1.5rem; color: #64748b; font-weight: 500;">
                Intensity Status: <span style="color: {color}; font-weight: 700; margin-left: 5px;">{status}</span>
            </p>
        </div>
    """, unsafe_allow_html=True)
