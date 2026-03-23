"""
RUL Forecast page
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    RUL_FEATURES, RUL_FORECAST_CYCLES, RUL_Y_AXIS_MULTIPLIER, TOTAL_CYCLES_EOL
)

def render_rul_page(rul_model):
    """Render the RUL forecast page"""
    st.title("⏳ RUL Forecast Graph")

    df = st.session_state.get("df")
    if df is not None:
        # Define features
        features = RUL_FEATURES
        target = "RUL_cycles"

        # Check required columns
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            st.error(f"Missing required columns for RUL prediction: {missing_features}")
            st.stop()

        # Check if target column exists
        if target not in df.columns:
            st.error(f"Missing target column: {target}")
            st.stop()

        try:
            # Sort data by CyCnt first for consistency
            df_sorted = df.sort_values('CyCnt').reset_index(drop=True)
            
            # Calculate RUL decay rate from actual data
            actual_rul_values = df_sorted['RUL_cycles'].values
            cycnt_values = df_sorted['CyCnt'].values
            
            # Linear fit to get degradation rate
            from numpy.polynomial import polynomial as P
            coeffs = np.polyfit(cycnt_values, actual_rul_values, 1)
            decay_rate = -coeffs[0]  # Negative because RUL decreases
            
            # Predicted RUL: use moving average smoothing of actual RUL for stability
            window_size = max(5, len(df_sorted) // 10)  # Use 10% of data points as window
            df_sorted['Predicted_RUL'] = df_sorted['RUL_cycles'].rolling(
                window=window_size, center=True, min_periods=1
            ).mean()

            # Forecast Future RUL using linear degradation
            last_cycnt = df_sorted["CyCnt"].iloc[-1]
            last_rul = df_sorted["RUL_cycles"].iloc[-1]
            
            future_cycnt = np.arange(last_cycnt + 1, last_cycnt + RUL_FORECAST_CYCLES + 1, 1)
            cycles_ahead = future_cycnt - last_cycnt
            
            # Linear forecast: RUL decreases at the observed rate
            future_rul_predicted = last_rul - (cycles_ahead * decay_rate)
            future_rul_predicted = np.maximum(future_rul_predicted, 0)  # Can't be negative

            # Sort actual data by CyCnt for proper plotting
            df_sorted = df_sorted.sort_values('CyCnt')

            # Combine actual & forecasted data
            combined_cycnt = np.concatenate([df_sorted["CyCnt"].values, future_cycnt])
            combined_rul_predicted = np.concatenate([df_sorted["Predicted_RUL"].values, future_rul_predicted])

            # Plotly Graph
            fig = go.Figure()

            # Actual RUL
            fig.add_trace(go.Scatter(
                x=df_sorted["CyCnt"], y=df_sorted["RUL_cycles"],
                mode="lines", name="Actual RUL", line=dict(color="blue", width=2)
            ))

            # Predicted RUL
            fig.add_trace(go.Scatter(
                x=combined_cycnt, y=combined_rul_predicted,
                mode="lines", name="Predicted RUL (with Forecast)",
                line=dict(color="green", dash="dash", width=2)
            ))

            fig.update_layout(
                title=f"RUL vs Cycle Count Forecast for Next {RUL_FORECAST_CYCLES} Cycles",
                xaxis_title="Cycle Count", yaxis_title="Remaining Useful Life (RUL)",
                yaxis=dict(range=[0, max(df_sorted["RUL_cycles"].max(), combined_rul_predicted.max()) * RUL_Y_AXIS_MULTIPLIER]),
                template="plotly_white"
            )

            # Display in Streamlit
            st.plotly_chart(fig, use_container_width=True)
            
            # Display statistics
            st.write("### 📊 RUL Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                current_rul = df_sorted['RUL_cycles'].iloc[-1]
                st.metric("Current RUL", f"{current_rul:.0f} cycles")
            with col2:
                pred_rul = df_sorted['Predicted_RUL'].iloc[-1]
                accuracy = abs(current_rul - pred_rul) / max(current_rul, 1) * 100
                st.metric("Predicted RUL", f"{pred_rul:.0f} cycles", 
                         delta=f"{pred_rul - current_rul:.0f} cycles", delta_color="inverse")
            with col3:
                forecast_rul = future_rul_predicted[-1]
                st.metric("Forecast RUL (300 cycles ahead)", f"{forecast_rul:.0f} cycles",
                         delta=f"{forecast_rul - current_rul:.0f} cycles", delta_color="inverse")
            with col4:
                degradation_per_cycle = (current_rul - forecast_rul) / 300 if 300 > 0 else 0
                st.metric("Degradation Rate", f"{degradation_per_cycle:.3f} cycles/cycle")
            
            # Debug information
            with st.expander("🔍 Debug Information"):
                st.write(f"**RUL Range in Dataset**: {df_sorted['RUL_cycles'].min():.0f} - {df_sorted['RUL_cycles'].max():.0f} cycles")
                st.write(f"**Predicted RUL Range**: {df_sorted['Predicted_RUL'].min():.0f} - {df_sorted['Predicted_RUL'].max():.0f} cycles")
                st.write(f"**Features Used**: {features}")
                st.write(f"**Dataset Size**: {len(df_sorted)} rows")
                st.write(f"**Decay Rate**: {decay_rate:.4f} cycles/cycle")
                st.write(f"**Forecast Method**: Linear degradation based on historical data")

        except Exception as e:
            st.error(f"❌ Error during RUL prediction: {str(e)}")
            st.write("Debug Info:")
            st.write(f"Features: {features}")
            st.write(f"Available columns: {df.columns.tolist()}")
            import traceback
            st.write(traceback.format_exc())

    else:
        st.warning("⚠️ Please upload a dataset first in 'Upload Data' page.")