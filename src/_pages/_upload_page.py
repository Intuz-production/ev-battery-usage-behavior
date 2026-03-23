"""
Upload page for battery dataset
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    SOH_FEATURES, RUL_FEATURES, TOTAL_CYCLES_EOL,
    BATTERY_STATUS_DISCHARGE, SOC_FULL
)

def render_upload_page():
    """Render the data upload page"""
    st.title("📂 Upload Battery Dataset")

    uploaded_file = st.file_uploader("Upload your battery dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # Data Relevance Check
            missing_soh_features = [f for f in SOH_FEATURES if f not in df.columns]
            missing_rul_features = [f for f in RUL_FEATURES if f not in df.columns]

            if not missing_soh_features and not missing_rul_features:
                st.success("✅ Your dataset contains all required features for SOH and RUL prediction!")
            else:
                if missing_soh_features:
                    st.error(f"❌ Missing required columns for SOH prediction: {missing_soh_features}")
                if missing_rul_features:
                    st.error(f"❌ Missing required columns for RUL prediction: {missing_rul_features}")

            # Check required columns
            features = SOH_FEATURES
            targets = ['SOH']
            missing_features = [f for f in features if f not in df.columns]

            if missing_features:
                st.error(f"❌ Missing required columns: {missing_features}")
            else:
                st.session_state["features"] = features
                st.session_state["targets"] = targets
                st.success("✅ Dataset loaded! Switch to 'SOH Forecast' or 'RUL Forecast' pages from the sidebar.")

                # --- Additional processing ---
                # RUL Calculation
                df = df[(df["BSt"] == BATTERY_STATUS_DISCHARGE)]

                # Calculate RUL in cycles
                df['RUL_cycles'] = TOTAL_CYCLES_EOL - df['CyCnt']
                df['RUL_cycles'] = df['RUL_cycles'].clip(lower=0)
                df_sorted = df.sort_values('CyCnt')

                Q_nominal = df['RCap'].max()

                # Filter for SOC = 100 and battery in discharge state
                df = df[(df["Soc"] == SOC_FULL) & (df["BSt"] == BATTERY_STATUS_DISCHARGE) & (df['RCap'] != 0)]

                # Calculate SOH only for filtered data
                df["SOH"] = (df["RCap"] / Q_nominal) * 100

                # Ensure RUL_cycles is present in filtered dataframe
                if 'RUL_cycles' not in df.columns:
                    df['RUL_cycles'] = TOTAL_CYCLES_EOL - df['CyCnt']
                    df['RUL_cycles'] = df['RUL_cycles'].clip(lower=0)

                # Store final processed DataFrame in session
                st.session_state["df"] = df
                st.session_state["rul_features"] = RUL_FEATURES

                # Preview
                st.write("### 📊 Dataset Preview")
                st.dataframe(df.head())

        except Exception as e:
            st.error(f"⚠️ An error occurred during data processing: {e}")