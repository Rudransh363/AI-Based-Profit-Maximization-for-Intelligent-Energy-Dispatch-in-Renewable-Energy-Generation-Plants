# ⚡ AI-Based Intelligent Energy Dispatch System

**Final Year Major Project — Green Day Ahead Market (GDAM) Solar Energy Optimization**

An intelligent system that forecasts solar generation and electricity prices, then optimizes battery-aware energy dispatch to maximize profit for a renewable solar energy plant operating in India's GDAM.

---

## Architecture

```
┌──────────────┐    ┌──────────────────┐    ┌────────────────────┐
│  NASA POWER   │───▶│  Solar Forecast   │───▶│                    │
│  Weather Data │    │  (XGBoost)        │    │  Dispatch          │
└──────────────┘    └──────────────────┘    │  Optimization      │
                                             │  (PuLP + CBC)      │──▶ Profit
┌──────────────┐    ┌──────────────────┐    │                    │
│  GDAM Market  │───▶│  Price Forecast   │───▶│  Battery-Aware     │
│  Snapshots    │    │  (XGBoost)        │    │  Scheduling        │
└──────────────┘    └──────────────────┘    └────────────────────┘
```

## Project Structure

```
├── data/                   # Saved models and processed data
├── models/
│   ├── solar_forecast.py   # XGBoost solar generation forecasting
│   └── price_forecast.py   # XGBoost GDAM price forecasting
├── optimization/
│   └── dispatch.py         # PuLP-based battery dispatch optimizer
├── dashboard/
│   └── app.py              # Streamlit interactive dashboard
├── utils/
│   └── preprocessing.py    # Data loading, cleaning, feature engineering
├── notebooks/              # Jupyter notebooks (exploration)
├── requirements.txt
└── README.md
```

## Setup Instructions

### 1. Prerequisites

- Python 3.9+
- pip

### 2. Install Dependencies

```bash
cd "Major Project"
pip install -r requirements.txt
```

### 3. Run the Dashboard

```bash
streamlit run dashboard/app.py
```

### 4. Upload Data

In the dashboard sidebar:
1. Upload the **NASA POWER CSV** file
2. Upload all **GDAM Market Snapshot Excel** files

The system will automatically preprocess data, train models, and run optimization.

---

## Datasets

### NASA POWER (Weather/Solar)
- Source: NASA POWER API
- Location: Jaipur, Rajasthan (26.91°N, 75.79°E)
- Parameters: Irradiance (GHI), Temperature, Relative Humidity
- Resolution: Hourly
- Note: Irradiance is synthesized using a clear-sky model when real data is unavailable

### GDAM Market Snapshots
- Source: Indian Energy Exchange (IEX) — Green Day Ahead Market
- Key Column: MCP (Market Clearing Price, Rs/MWh)
- Resolution: Hourly (24 time blocks per day)

---

## System Components

### 1. Solar Generation Forecasting
- **Model:** XGBoost Regressor
- **Features:** Irradiance, temperature, humidity, hour, weekday, lag features, rolling averages
- **Output:** Hourly solar generation (MW)
- **Evaluation:** MAE, RMSE

### 2. GDAM Price Forecasting
- **Model:** XGBoost Regressor
- **Features:** Lag prices (1h–48h), rolling averages, hour, weekday
- **Output:** Hourly market clearing price (Rs/MWh)
- **Evaluation:** MAE, RMSE

### 3. Battery Dispatch Optimization
- **Solver:** PuLP with CBC
- **Objective:** Maximize total profit
- **Decisions per hour:**
  - Energy sold to grid
  - Energy stored in battery
  - Battery discharge amount
  - Energy purchased from grid
- **Constraints:** Battery capacity, SOC limits, charge/discharge rates, power balance

### 4. Streamlit Dashboard
- Dataset upload and preview
- Solar forecast visualizations
- Price forecast visualizations
- Dispatch schedule with battery SOC
- Profit analysis and breakdown

---

## Assumptions

1. Solar plant capacity: 10 MW (configurable)
2. Battery capacity: 20 MWh (configurable via dashboard)
3. Charge/discharge efficiency: 92%/95%
4. Buy price premium: 10% above market price
5. Irradiance synthesized using clear-sky model for Jaipur coordinates
6. Solar generation estimated from irradiance × panel efficiency (18%)

## Limitations

- Uses synthetic irradiance (NASA POWER data shows -999 for future dates)
- No real plant generation data — estimated from physics model
- Single-node optimization (no grid topology)
- No degradation modeling for battery
- Assumes perfect forecast in optimization (no stochastic element)

## Future Scope

- Integration with real-time IEX API
- Actual plant SCADA data for solar generation
- Multi-day rolling horizon optimization
- Battery degradation cost modeling
- Probabilistic forecasting with confidence intervals
- Real-time deployment with scheduling

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Forecasting | XGBoost, scikit-learn |
| Optimization | PuLP (CBC solver) |
| Dashboard | Streamlit, Matplotlib |
| Data Processing | Pandas, NumPy |
