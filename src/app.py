"""
Main Streamlit application for EV Battery SOH and RUL Analysis
"""
import streamlit as st
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    SOH_MODEL_PATH, RUL_MODEL_PATH, SOH_FEATURES, RUL_FEATURES,
    TOTAL_CYCLES_EOL, BATTERY_STATUS_DISCHARGE, SOC_FULL
)
from src._pages._upload_page import render_upload_page
from src._pages._soh_page import render_soh_page
from src._pages._rul_page import render_rul_page
from src.models.model_loader import load_models

# Set page config
st.set_page_config(
    page_title="Battery SOH & RUL Forecast",
    layout="wide",
    page_icon="🔋"
)

# Load models
soh_model, rul_model = load_models(SOH_MODEL_PATH, RUL_MODEL_PATH)

# Sidebar Navigation
st.sidebar.title("🔋 Navigation")
page = st.sidebar.radio("Go to", ["Upload Data", "SOH Forecast", "RUL Forecast"])

# Main application logic
if page == "Upload Data":
    render_upload_page()
elif page == "SOH Forecast":
    render_soh_page(soh_model)
elif page == "RUL Forecast":
    render_rul_page(rul_model)

def main():
    """Main application entry point - for testing purposes"""
    pass

if __name__ == "__main__":
    main()