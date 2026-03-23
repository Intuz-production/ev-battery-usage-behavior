"""
SOH Forecast page
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    SOH_FEATURES, SOH_FORECAST_CYCLES, SOH_FORECAST_DAYS,
    TIME_FEATURES, TREND_WINDOW_SIZE, MAX_SOH_DROP_PER_STEP,
    SOH_Y_AXIS_RANGE
)

def generate_future_inputs_trend_based(df, features, steps=300, window=50):
    """Generate future inputs based on trends"""
    trends = {}
    df_sorted = df.sort_values(by="CyCnt")
    for feature in features:
        y = df_sorted[feature].values[-window:]
        x = np.arange(window)
        coeffs = np.polyfit(x, y, 1)  # linear fit
        trends[feature] = coeffs

    future_inputs = []
    for step in range(steps):
        point = []
        for feature in features:
            slope, intercept = trends[feature]
            val = slope * (window + step) + intercept
            point.append(max(val, 0))  # no negative values
        future_inputs.append(point)
    return np.array(future_inputs)

def forecast_next_steps_decreasing(model, input_seq, scaler_X, scaler_y, steps=300, max_drop_per_step=0.1):
    """Forecast SOH with decreasing trend"""
    input_scaled = scaler_X.transform(input_seq)
    forecasted_soh = []

    pred_scaled = model.predict(input_scaled[0:1])
    pred = scaler_y.inverse_transform(pred_scaled)
    last_soh = pred[0][0]
    forecasted_soh.append(last_soh)

    for i in range(1, steps):
        pred_scaled = model.predict(input_scaled[i:i+1])
        pred = scaler_y.inverse_transform(pred_scaled)[0][0]

        # Ensure non-increasing and slow degradation
        pred = min(pred, forecasted_soh[-1])
        if forecasted_soh[-1] - pred > max_drop_per_step:
            pred = forecasted_soh[-1] - max_drop_per_step

        forecasted_soh.append(pred)

    return forecasted_soh

def render_soh_page(soh_model):
    """Render the SOH forecast page"""
    st.title("🔋 SOH Forecast Graph")

    df = st.session_state.get("df")
    if df is not None:
        features = st.session_state["features"]
        targets = st.session_state["targets"]

        # Data Processing
        scaler_X = MinMaxScaler()
        scaler_SOH = MinMaxScaler()

        X = df[features].values
        y_SOH = df[['SOH']].values  # Only SOH

        X_scaled = scaler_X.fit_transform(X)
        y_SOH_scaled = scaler_SOH.fit_transform(y_SOH)

        # Predict SOH
        y_pred_scaled = soh_model.predict(X_scaled)
        y_pred_SOH = scaler_SOH.inverse_transform(y_pred_scaled)[:, 0]

        df["Predicted_SOH"] = y_pred_SOH
        df_sorted = df.sort_values(by="CyCnt")

        # Generate future inputs and forecast SOH
        future_inputs = generate_future_inputs_trend_based(df, features, steps=SOH_FORECAST_CYCLES)
        forecasted_soh = forecast_next_steps_decreasing(
            soh_model, future_inputs, scaler_X, scaler_SOH,
            steps=SOH_FORECAST_CYCLES
        )

        # Cycle counts
        actual_cyc = df_sorted["CyCnt"].values
        future_cycnt = np.arange(actual_cyc[-1] + 1, actual_cyc[-1] + SOH_FORECAST_CYCLES + 1)

        # Plot SOH Forecast with improved method
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_sorted["CyCnt"], y=df_sorted["SOH"],
            mode="lines", name="Actual SOH", line=dict(color="blue", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df_sorted["CyCnt"], y=df_sorted["Predicted_SOH"],
            mode="lines", name="Predicted SOH", line=dict(color="green", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=future_cycnt, y=forecasted_soh,
            mode="lines", name=f"Forecasted SOH ({SOH_FORECAST_CYCLES} cycles)",
            line=dict(color="orange", dash="dash", width=2)
        ))

        fig.update_layout(
            title=f"SOH vs Cycle Count for the next {SOH_FORECAST_CYCLES}-Cycle Forecast",
            xaxis_title="Cycle Count (CyCnt)", yaxis_title="State of Health (SOH)",
            yaxis=dict(range=SOH_Y_AXIS_RANGE), template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        # SOH vs Time (if time column exists)
        if "time" in df.columns:
            # Ensure 'time' is in datetime format
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df = df.dropna(subset=["time"])  # Remove invalid dates
            df = df.sort_values(by="time")  # Sort by time

            # Create time feature: days since first measurement
            df["days_since_start"] = (df["time"] - df["time"].min()).dt.days

            # Define features & target
            target = "SOH"

            # Scaling
            scaler_X_time = MinMaxScaler()
            scaler_SOH_time = MinMaxScaler()

            X_time = df[TIME_FEATURES].values
            y_SOH_time = df[[target]].values

            X_time_scaled = scaler_X_time.fit_transform(X_time)
            y_SOH_time_scaled = scaler_SOH_time.fit_transform(y_SOH_time)

            # Predict SOH
            y_pred_time_scaled = soh_model.predict(X_time_scaled)
            y_pred_SOH_time = scaler_SOH_time.inverse_transform(y_pred_time_scaled)[:, 0]
            df["Predicted_SOH_time"] = y_pred_SOH_time

            # Generate future time-based inputs with trend analysis
            future_time_inputs = generate_future_inputs_trend_based(df, TIME_FEATURES, steps=SOH_FORECAST_DAYS)
            forecasted_soh_time = forecast_next_steps_decreasing(
                soh_model, future_time_inputs, scaler_X_time, scaler_SOH_time,
                steps=SOH_FORECAST_DAYS
            )

            # Future dates
            last_date = df["time"].max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=SOH_FORECAST_DAYS, freq="D")

            # Plotly Graph for Time-based Forecast
            fig_time = go.Figure()

            # Actual SOH
            fig_time.add_trace(go.Scatter(
                x=df["time"], y=df["SOH"],
                mode="lines", name="Actual SOH", line=dict(color="blue", width=2)
            ))

            # Predicted SOH
            fig_time.add_trace(go.Scatter(
                x=df["time"], y=df["Predicted_SOH_time"],
                mode="lines", name="Predicted SOH", line=dict(color="green", width=2)
            ))

            # Forecasted SOH
            fig_time.add_trace(go.Scatter(
                x=future_dates, y=forecasted_soh_time,
                mode="lines", name=f"Forecasted SOH ({SOH_FORECAST_DAYS} days)",
                line=dict(color="orange", dash="dash", width=2)
            ))

            fig_time.update_layout(
                title="SOH vs Time Forecast for Next 3 Months",
                xaxis_title="Time", yaxis_title="State of Health (SOH)",
                yaxis=dict(range=SOH_Y_AXIS_RANGE),
                template="plotly_white"
            )

            # Display in Streamlit
            st.plotly_chart(fig_time, use_container_width=True)

        # SOH vs Distance Traveled (placeholder)
        fig = go.Figure()

        # Add an invisible trace (just for placeholder)
        fig.add_trace(go.Scatter(
            x=[0, 100], y=[100, 100],
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),  # invisible line
            showlegend=False
        ))

        # Add "Coming Soon" annotation
        fig.add_annotation(
            text="🚧 Coming Soon 🚧",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=24, color="gray")
        )

        # Update layout with correct axis settings (no duplicates!)
        fig.update_layout(
            title="SOH vs Distance Traveled",
            xaxis=dict(title="Distance (km)", showgrid=True),
            yaxis=dict(title="State of Health (SOH)", range=SOH_Y_AXIS_RANGE, showgrid=True),
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("⚠️ Please upload a dataset first in 'Upload Data' page.")