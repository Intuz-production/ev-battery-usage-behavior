import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Set page config
st.set_page_config(page_title="Battery SOH & RUL Forecast", layout="wide")

# Load models
soh_model_path = "models/best_soh_model.keras"
rul_model_path = "models/best_rul_model.keras"

soh_model = load_model(soh_model_path)
rul_model = load_model(rul_model_path)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload Data", "SOH Forecast", "RUL Forecast"])

# File Upload Page
if page == "Upload Data":
    st.title("📂 Upload Battery Dataset")
    
    uploaded_file = st.file_uploader("Upload your battery dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # Data Relevance Check
            required_soh_features = ['RCap', 'Volt', 'BCur']
            required_rul_features = ['CyCnt', 'BCur']
            
            missing_soh_features = [f for f in required_soh_features if f not in df.columns]
            missing_rul_features = [f for f in required_rul_features if f not in df.columns]
            
            if not missing_soh_features and not missing_rul_features:
                st.success("✅ Your dataset contains all required features for SOH and RUL prediction!")
            else:
                if missing_soh_features:
                    st.error(f"❌ Missing required columns for SOH prediction: {missing_soh_features}")
                if missing_rul_features:
                    st.error(f"❌ Missing required columns for RUL prediction: {missing_rul_features}")

            # Check required columns
            features = ['RCap', 'Volt', 'BCur']
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
                TOTAL_CYCLES_EOL = 2000
                df = df[(df["BSt"] == 770)]

                # Calculate RUL in cycles
                df['RUL_cycles'] = TOTAL_CYCLES_EOL - df['CyCnt']
                df['RUL_cycles'] = df['RUL_cycles'].clip(lower=0)
                df_sorted = df.sort_values('CyCnt')

                Q_nominal = df['RCap'].max()

                # Filter for SOC = 100 and battery in discharge state
                df = df[(df["Soc"] == 100) & (df["BSt"] == 770) & (df['RCap'] != 0)]

                # Calculate SOH only for filtered data
                df["SOH"] = (df["RCap"] / Q_nominal) * 100

                # Store final processed DataFrame in session
                st.session_state["df"] = df  

                # Preview
                st.write("### 📊 Dataset Preview")
                st.dataframe(df.head()) 

        except Exception as e:
            st.error(f"⚠️ An error occurred during data processing: {e}")


# SOH Forecast Page
elif page == "SOH Forecast":
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

        # ----- NEW SOH FORECASTING METHODS -----
        
        # Function to generate future inputs based on trends
        def generate_future_inputs_trend_based(df, features, steps=300, window=50):
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

        # Function for SOH forecasting that ensures decreasing trend
        def forecast_next_steps_decreasing(model, input_seq, scaler_X, scaler_y, steps=300, max_drop_per_step=0.1):
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

        # Generate future inputs and forecast SOH
        future_inputs = generate_future_inputs_trend_based(df, features, steps=300)
        forecasted_soh = forecast_next_steps_decreasing(soh_model, future_inputs, scaler_X, scaler_SOH, steps=300)

        # Cycle counts
        actual_cyc = df_sorted["CyCnt"].values
        future_cycnt = np.arange(actual_cyc[-1] + 1, actual_cyc[-1] + 301)

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
            mode="lines", name="Forecasted SOH (300 cycles)", 
            line=dict(color="orange", dash="dash", width=2)
        ))

        fig.update_layout(
            title="SOH vs Cycle Count for the next 300-Cycle Forecast",
            xaxis_title="Cycle Count (CyCnt)", yaxis_title="State of Health (SOH)",
            yaxis=dict(range=[0, 120]), template="plotly_white"
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
            time_features = ["days_since_start", "Volt", "BCur"]
            target = "SOH"

            # Scaling
            scaler_X_time = MinMaxScaler()
            scaler_SOH_time = MinMaxScaler()

            X_time = df[time_features].values
            y_SOH_time = df[[target]].values

            X_time_scaled = scaler_X_time.fit_transform(X_time)
            y_SOH_time_scaled = scaler_SOH_time.fit_transform(y_SOH_time)

            # Predict SOH
            y_pred_time_scaled = soh_model.predict(X_time_scaled)
            y_pred_SOH_time = scaler_SOH_time.inverse_transform(y_pred_time_scaled)[:, 0]
            df["Predicted_SOH_time"] = y_pred_SOH_time

            # Generate future time-based inputs with trend analysis
            future_time_inputs = generate_future_inputs_trend_based(df, time_features, steps=90)
            forecasted_soh_time = forecast_next_steps_decreasing(soh_model, future_time_inputs, 
                                                               scaler_X_time, scaler_SOH_time, steps=90)

            # Future dates
            last_date = df["time"].max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=90, freq="D")

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
                mode="lines", name="Forecasted SOH (90 days)", 
                line=dict(color="orange", dash="dash", width=2)
            ))

            fig_time.update_layout(
                title="SOH vs Time Forecast for Next 3 Months",
                xaxis_title="Time", yaxis_title="State of Health (SOH)",
                yaxis=dict(range=[0, 120]),
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
            yaxis=dict(title="State of Health (SOH)", range=[0, 120], showgrid=True),
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("⚠️ Please upload a dataset first in 'Upload Data' page.")

# RUL Forecast Page
elif page == "RUL Forecast":
    st.title("⏳ RUL Forecast Graph")

    df = st.session_state.get("df")
    if df is not None:
        # Define features
        features = ["CyCnt", "BCur", "Volt"]
        target = "RUL_cycles"

        # Check required columns
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            st.error(f"Missing required columns for RUL prediction: {missing_features}")
            st.stop()

        # Scaling
        scaler_X = MinMaxScaler()
        scaler_RUL = MinMaxScaler()

        X = df[features].values
        y_RUL = df[[target]].values

        X_scaled = scaler_X.fit_transform(X)
        y_RUL_scaled = scaler_RUL.fit_transform(y_RUL)

        # Predict RUL
        y_pred_scaled = rul_model.predict(X_scaled)
        y_pred_RUL = scaler_RUL.inverse_transform(y_pred_scaled)[:, 0]

        df["Predicted_RUL"] = y_pred_RUL

        # Forecast Future RUL
        last_cycnt = df["CyCnt"].iloc[-1]
        future_cycnt = np.arange(last_cycnt + 1, last_cycnt + 301, 1)
        future_X = np.tile(X[-1], (300, 1))  
        future_X[:, 0] = future_cycnt  
        future_X_scaled = scaler_X.transform(future_X)
        future_y_pred_scaled = rul_model.predict(future_X_scaled)
        future_rul_predicted = scaler_RUL.inverse_transform(future_y_pred_scaled)[:, 0]

        # Combine actual & forecasted data
        combined_cycnt = np.concatenate([df["CyCnt"], future_cycnt])
        combined_rul_predicted = np.concatenate([df["Predicted_RUL"], future_rul_predicted])

        # Plotly Graph
        fig = go.Figure()

        # Actual RUL
        fig.add_trace(go.Scatter(
            x=df["CyCnt"], y=df["RUL_cycles"],
            mode="lines", name="Actual RUL", line=dict(color="blue", width=2)
        ))

        # Predicted RUL
        fig.add_trace(go.Scatter(
            x=combined_cycnt, y=combined_rul_predicted,
            mode="lines", name="Predicted RUL (with Forecast)",
            line=dict(color="green", dash="dash", width=2)
        ))

        fig.update_layout(
            title="RUL vs Cycle Count Forecast for Next 300 Cycles",
            xaxis_title="Cycle Count", yaxis_title="Remaining Useful Life (RUL)",
            yaxis=dict(range=[0, df["RUL_cycles"].max() * 1.1]),
            template="plotly_white"
        )

        # Display in Streamlit
        st.plotly_chart(fig,use_container_width=True)

    else:
        st.warning("⚠️ Please upload a dataset first in 'Upload Data' page.")