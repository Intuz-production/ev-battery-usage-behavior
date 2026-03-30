## About Intuz
This library is maintained by [Intuz](https://www.intuz.com) — an AI-first software development company specializing in [machine learning development services](https://www.intuz.com/machine-learning-development-services) and [custom ai development](https://www.intuz.com/custom-ai-development).
<br><br>

# EV Battery Usage Behaviour — Use Cases

## 1. 🔋 Fleet Battery Health Assessment

**Problem**
Fleet operators lack real-time visibility into individual battery degradation rates across their EV fleet, making it impossible to schedule predictive maintenance and resulting in unexpected vehicle downtime and high replacement costs.

**Solution**
The system uploads fleet telemetry data (voltage, current, capacity, cycle counts) and provides instant SOH calculation showing what percentage of battery capacity remains. The upload page validates data integrity and automatically calculates SOH as:
```
SOH = (RemainingCapacity / MaxCapacity) × 100
```

**Outcome**
Battery health is now quantifiable and actionable — fleet managers can immediately identify which vehicles need maintenance before failure occurs, reducing emergency downtime costs.

---

## 2. 🚨 Early Anomaly Detection & Safety Risk Mitigation

**Problem**
Batteries with abnormal degradation patterns cannot be automatically detected, allowing unsafe or defective batteries to remain in service longer than safe.

**Solution**
The system monitors SOH and RUL forecasts against expected degradation curves. By applying a `MAX_SOH_DROP_PER_STEP` constraint (0.1% per step), the system flags batteries with abnormally steep degradation. Pre-trained models capture normal degradation signatures — deviations indicate manufacturing defects, abuse, or environmental stress.

**Outcome**
Anomalous batteries are identified before becoming safety risks. Early detection prevents potential vehicle malfunctions, improves customer safety, and reduces warranty claims from defective units.

---

## 3. 📈 Battery Health & Lifecycle Forecasting

**Problem**
Fleet operators need to know both current battery health and when to replace batteries to prevent costly failures.

**Solution**
Upload battery data. The system calculates:
- **SOH** — percentage of capacity remaining
- **RUL** — cycles remaining until end-of-life

Both metrics are forecasted **300 cycles ahead** using AI models to visualise degradation trends.

**Outcome**
See battery health status and remaining lifespan in one view. Plan maintenance before failures occur. Replace batteries at the optimal time — not too early, not too late.

# EV Battery Usage Behavior Analysis

A comprehensive Streamlit-based dashboard for monitoring and forecasting Electric Vehicle (EV) battery State of Health (SOH) and Remaining Useful Life (RUL) using deep learning models.

## Description

This application provides a complete solution for EV battery health monitoring and predictive analytics. It enables users to upload battery datasets and gain insights into:
- Current State of Health (SOH) assessment
- SOH degradation forecasting over cycle counts and time periods
- Remaining Useful Life (RUL) predictions
- Interactive visualizations for data-driven decision making

Built with open-source technologies, the dashboard leverages TensorFlow deep neural networks for accurate battery health predictions.

## Features

- **📂 Data Upload & Validation**: Secure CSV upload with automatic data validation
- **🔋 SOH Analysis**: Real-time SOH calculation and multi-horizon forecasting
- **⏳ RUL Prediction**: Cycle-based remaining life estimation with trend analysis
- **📊 Interactive Visualizations**: Plotly-powered charts for comprehensive insights
- **🔧 Configurable Parameters**: Environment-based configuration for customization
- **🏗️ Modular Architecture**: Clean, maintainable codebase with proper separation of concerns

### Required Dataset Columns

| Column Name     | Description                              | Data Type     | Required For      | Sample Value        |
|------------------|------------------------------------------|----------------|-------------------|---------------------|
| `CyCnt`          | Cycle count                              | `int`          | SOH, RUL          | 150                 |
| `Volt`           | Battery voltage (V)                      | `float`        | SOH, RUL          | 364.2               |
| `BCur`           | Battery current (A)                      | `float`        | SOH, RUL          | -23.5               |
| `RCap`           | Remaining capacity (Ah)                  | `float`        | SOH Calculation   | 40.3                |
| `Soc`            | State of charge (%)                      | `int`          | Data Filtering    | 100                 |
| `BSt`            | Battery status code                      | `int`          | Data Filtering    | 770                 |
| `time`           | Timestamp of data collection             | `datetime`     | Time-based Forecast| 2024-05-10 10:30:00 |
| `SOH`            | State of Health (%)                      | `float`        | Target (Optional) | 91.7                |
| `RUL_cycles`     | Remaining Useful Life in cycles          | `int`          | Target (Optional) | 1850                |

> **Note**: SOH and RUL columns are calculated automatically if not present in the dataset.

## Project Structure

```
├── config/                 # Configuration files
│   ├── __init__.py
│   └── settings.py         # Application settings and constants
├── data/                   # Sample datasets
│   └── VehicleBattery.csv  # Sample battery data
├── docs/                   # Documentation
├── models/                 # Pre-trained ML models
│   ├── best_soh_model.keras
│   └── best_rul_model.keras
├── src/                    # Source code
│   ├── __init__.py
│   ├── app.py             # Main application entry point
│   ├── models/            # Model loading utilities
│   │   ├── __init__.py
│   │   └── model_loader.py
│   └── pages/             # Page components
│       ├── __init__.py
│       ├── upload_page.py
│       ├── soh_page.py
│       └── rul_page.py
├── tests/                  # Unit tests
├── .env.example           # Environment variables template
├── .gitignore            # Git ignore rules
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd poc-ev-battery-usage-behavior
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment** (optional)
   ```bash
   cp .env.example .env
   # Edit .env with your custom settings
   ```

5. **Verify model files**
   Ensure the following model files exist in the `models/` directory:
   - `best_soh_model.keras`
   - `best_rul_model.keras`

### Running the Application

```bash
streamlit run src/app.py
```

The application will be available at `http://localhost:8501`

## Configuration

The application uses environment variables for configuration. Copy `.env.example` to `.env` and modify as needed:

```bash
# Battery Parameters
TOTAL_CYCLES_EOL=2000

# Forecasting Parameters
SOH_FORECAST_CYCLES=300
SOH_FORECAST_DAYS=90
RUL_FORECAST_CYCLES=300

# Model Parameters
TREND_WINDOW_SIZE=50
MAX_SOH_DROP_PER_STEP=0.1
```

## Usage

1. **Upload Data**: Start by uploading your battery dataset in CSV format
2. **SOH Forecast**: Analyze current State of Health and view cycle/time-based forecasts
3. **RUL Forecast**: Review Remaining Useful Life predictions and future projections

## Dependencies

All dependencies are open-source and listed in `requirements.txt`:

- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning utilities
- **tensorflow**: Deep learning framework
- **plotly**: Interactive visualizations
- **python-dotenv**: Environment variable management

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Maintain modular code structure
- Document functions and classes

## License

This project is open-source and available under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
- Check the documentation in `docs/`
- Review existing issues on GitHub
- Create a new issue with detailed information
