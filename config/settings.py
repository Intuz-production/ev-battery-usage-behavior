"""
Configuration file for EV Battery Usage Behavior Analysis
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Model paths
SOH_MODEL_PATH = MODELS_DIR / "best_soh_model.keras"
RUL_MODEL_PATH = MODELS_DIR / "best_rul_model.keras"

# Battery parameters
TOTAL_CYCLES_EOL = int(os.getenv("TOTAL_CYCLES_EOL", "2000"))

# Forecasting parameters
SOH_FORECAST_CYCLES = int(os.getenv("SOH_FORECAST_CYCLES", "300"))
SOH_FORECAST_DAYS = int(os.getenv("SOH_FORECAST_DAYS", "90"))
RUL_FORECAST_CYCLES = int(os.getenv("RUL_FORECAST_CYCLES", "300"))

# Feature configurations
SOH_FEATURES = ['RCap', 'Volt', 'BCur']
RUL_FEATURES = ['CyCnt', 'BCur', 'Volt']
TIME_FEATURES = ['days_since_start', 'Volt', 'BCur']

# Data filtering parameters
BATTERY_STATUS_DISCHARGE = 770
SOC_FULL = 100

# Model parameters
TREND_WINDOW_SIZE = int(os.getenv("TREND_WINDOW_SIZE", "50"))
MAX_SOH_DROP_PER_STEP = float(os.getenv("MAX_SOH_DROP_PER_STEP", "0.1"))

# Plot parameters
SOH_Y_AXIS_RANGE = [0, 120]
RUL_Y_AXIS_MULTIPLIER = 1.1