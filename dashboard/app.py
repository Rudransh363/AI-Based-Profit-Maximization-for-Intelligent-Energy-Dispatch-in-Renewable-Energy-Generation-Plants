"""Streamlit dashboard for AI Energy Dispatch System."""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.preprocessing import (
    preprocess_nasa, preprocess_gdam, load_nasa_power, synthesize_irradiance,
    reindex_hourly, estimate_solar_generation, add_time_features,
    load_gdam_files, PLANT_CAPACITY_MW,
)
from models.solar_forecast import train_solar_model, prepare_solar_features
from models.price_forecast import train_price_model, prepare_price_features
from optimization.dispatch import optimize_dispatch, get_dispatch_summary, DEFAULT_PARAMS

st.set_page_config(page_title="AI Energy Dispatch System", layout="wide", page_icon="⚡")

st.title("⚡ AI-Based Intelligent Energy Dispatch System")
st.caption("Green Day Ahead Market (GDAM) — Solar Energy Plant Optimization")

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("📁 Data Upload")
nasa_file = st.sidebar.file_uploader("Upload NASA POWER CSV", type=["csv"])
gdam_files = st.sidebar.file_uploader("Upload GDAM Market Snapshots", type=["xlsx"], accept_multiple_files=True)

st.sidebar.header("🔋 Battery Parameters")
battery_capacity = st.sidebar.slider("Battery Capacity (MWh)", 5.0, 50.0, 20.0, 1.0)
charge_rate = st.sidebar.slider("Max Charge Rate (MW)", 1.0, 15.0, 5.0, 0.5)
discharge_rate = st.sidebar.slider("Max Discharge Rate (MW)", 1.0, 15.0, 5.0, 0.5)
initial_soc = st.sidebar.slider("Initial SOC (MWh)", 0.0, battery_capacity, battery_capacity / 2, 0.5)
plant_demand = st.sidebar.slider("Plant Own Demand (MW)", 0.0, 5.0, 2.0, 0.5)

forecast_hours = st.sidebar.slider("Forecast/Optimize Hours", 24, 168, 48, 24)

if not nasa_file or not gdam_files:
    st.info("👈 Upload both NASA POWER CSV and GDAM Market Snapshot files to get started.")

    st.markdown("---")
    st.subheader("System Architecture")
    st.markdown("""
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
    """)
    st.stop()


# ── Data Processing ──────────────────────────────────────────────────────────
@st.cache_data
def process_nasa(file_bytes):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(file_bytes)
        f.flush()
        return preprocess_nasa(f.name)


@st.cache_data
def process_gdam(file_bytes_list):
    import tempfile
    paths = []
    for fb in file_bytes_list:
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            f.write(fb)
            f.flush()
            paths.append(f.name)
    return preprocess_gdam(paths)


with st.spinner("Processing datasets..."):
    nasa_df = process_nasa(nasa_file.getvalue())
    gdam_df = process_gdam([f.getvalue() for f in gdam_files])

st.success(f"Data loaded — NASA: {len(nasa_df)} hours | GDAM: {len(gdam_df)} hours")

# ── Tab Layout ───────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "☀️ Solar Forecast",
    "💰 Price Forecast",
    "🔋 Dispatch Optimization",
    "📊 Profit Analysis",
    "📋 Data Explorer",
])

# ── TAB 1: Solar Forecast ────────────────────────────────────────────────────
with tab1:
    st.subheader("Solar Generation Forecast")

    with st.spinner("Training solar forecast model..."):
        solar_results = train_solar_model(nasa_df, test_size=0.2)

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{solar_results['mae']:.4f} MW")
    col2.metric("RMSE", f"{solar_results['rmse']:.4f} MW")
    col3.metric("Plant Capacity", f"{PLANT_CAPACITY_MW} MW")

    # Forecast vs Actual plot
    fig, ax = plt.subplots(figsize=(12, 4))
    test_idx = solar_results["X_test"].index
    ax.plot(test_idx, solar_results["y_test"].values, label="Actual", alpha=0.8, linewidth=1)
    ax.plot(test_idx, solar_results["y_pred"], label="Predicted", alpha=0.8, linewidth=1, linestyle="--")
    ax.set_xlabel("Time")
    ax.set_ylabel("Generation (MW)")
    ax.set_title("Solar Generation — Actual vs Predicted (Test Set)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Daily generation pattern
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    hourly_avg = nasa_df.groupby("hour")["solar_generation_mw"].mean()
    ax2.bar(hourly_avg.index, hourly_avg.values, color="orange", alpha=0.7)
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Avg Generation (MW)")
    ax2.set_title("Average Solar Generation by Hour")
    ax2.set_xticks(range(0, 24))
    plt.tight_layout()
    st.pyplot(fig2)

    # Feature importance
    fi = pd.Series(
        solar_results["model"].feature_importances_,
        index=solar_results["features"]
    ).sort_values(ascending=True).tail(10)
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    fi.plot(kind="barh", ax=ax3, color="teal")
    ax3.set_title("Top 10 Feature Importances — Solar Model")
    plt.tight_layout()
    st.pyplot(fig3)


# ── TAB 2: Price Forecast ────────────────────────────────────────────────────
with tab2:
    st.subheader("GDAM Electricity Price Forecast")

    target_col = "MCP (Rs/MWh)"
    if target_col not in gdam_df.columns:
        candidates = [c for c in gdam_df.columns if "MCP" in c.upper() or "price" in c.lower()]
        if candidates:
            target_col = candidates[0]
        else:
            st.error("Could not find MCP/price column in GDAM data")
            st.stop()

    with st.spinner("Training price forecast model..."):
        price_results = train_price_model(gdam_df, target_col=target_col, test_size=0.2)

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"₹{price_results['mae']:.2f}/MWh")
    col2.metric("RMSE", f"₹{price_results['rmse']:.2f}/MWh")
    col3.metric("Avg Price", f"₹{gdam_df[target_col].mean():.2f}/MWh")

    fig, ax = plt.subplots(figsize=(12, 4))
    test_idx = price_results["X_test"].index
    ax.plot(test_idx, price_results["y_test"].values, label="Actual", alpha=0.8, linewidth=1)
    ax.plot(test_idx, price_results["y_pred"], label="Predicted", alpha=0.8, linewidth=1, linestyle="--")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price (Rs/MWh)")
    ax.set_title("GDAM Market Clearing Price — Actual vs Predicted (Test Set)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Hourly price pattern
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    hourly_price = gdam_df.groupby("hour")[target_col].mean()
    ax2.bar(hourly_price.index, hourly_price.values, color="steelblue", alpha=0.7)
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Avg Price (Rs/MWh)")
    ax2.set_title("Average GDAM Price by Hour")
    ax2.set_xticks(range(0, 24))
    plt.tight_layout()
    st.pyplot(fig2)

    # Feature importance
    fi = pd.Series(
        price_results["model"].feature_importances_,
        index=price_results["features"]
    ).sort_values(ascending=True).tail(10)
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    fi.plot(kind="barh", ax=ax3, color="coral")
    ax3.set_title("Top 10 Feature Importances — Price Model")
    plt.tight_layout()
    st.pyplot(fig3)


# ── TAB 3: Dispatch Optimization ─────────────────────────────────────────────
with tab3:
    st.subheader("Battery-Aware Dispatch Optimization")

    # Align data on common time range
    common_start = max(nasa_df.index.min(), gdam_df.index.min())
    common_end = min(nasa_df.index.max(), gdam_df.index.max())

    nasa_common = nasa_df.loc[common_start:common_end]
    gdam_common = gdam_df.loc[common_start:common_end]

    # Use the last N hours for optimization (simulating forward dispatch)
    n_hours = min(forecast_hours, len(nasa_common), len(gdam_common))

    solar_gen = nasa_common["solar_generation_mw"].values[-n_hours:]
    market_prices = gdam_common[target_col].values[-n_hours:]

    params = {
        "battery_capacity_mwh": battery_capacity,
        "battery_initial_soc": initial_soc,
        "charge_rate_max_mw": charge_rate,
        "discharge_rate_max_mw": discharge_rate,
        "plant_demand_mw": plant_demand,
    }

    with st.spinner("Running dispatch optimization..."):
        dispatch_results, total_profit, status = optimize_dispatch(
            solar_gen, market_prices, hours=n_hours, params=params
        )
        summary = get_dispatch_summary(dispatch_results, total_profit)

    if status != "Optimal":
        st.warning(f"Optimization status: {status}")
    else:
        st.success(f"Optimization complete — Status: {status}")

    # Key metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Profit", f"₹{summary['total_profit_rs']:,.0f}")
    c2.metric("Revenue", f"₹{summary['total_revenue_rs']:,.0f}")
    c3.metric("Buy Cost", f"₹{summary['total_buy_cost_rs']:,.0f}")
    c4.metric("Penalty", f"₹{summary['total_penalty_rs']:,.0f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Energy Sold", f"{summary['total_energy_sold_mwh']:.1f} MWh")
    c6.metric("Energy Bought", f"{summary['total_energy_bought_mwh']:.1f} MWh")
    c7.metric("Hours Selling", f"{summary['hours_selling']}")
    c8.metric("Hours Buying", f"{summary['hours_buying']}")

    # Dispatch schedule visualization
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Solar generation
    axes[0].fill_between(dispatch_results["hour"], dispatch_results["solar_generation_mw"],
                         alpha=0.5, color="orange", label="Solar Generation")
    axes[0].set_ylabel("MW")
    axes[0].set_title("Solar Generation")
    axes[0].legend(loc="upper right")

    # Price
    axes[1].plot(dispatch_results["hour"], dispatch_results["price_rs_per_mwh"],
                 color="steelblue", linewidth=1.5)
    axes[1].set_ylabel("Rs/MWh")
    axes[1].set_title("Market Price")

    # Dispatch actions
    axes[2].bar(dispatch_results["hour"], dispatch_results["energy_sold_mw"],
                alpha=0.7, color="green", label="Sold")
    axes[2].bar(dispatch_results["hour"], -dispatch_results["energy_bought_mw"],
                alpha=0.7, color="red", label="Bought")
    axes[2].bar(dispatch_results["hour"], dispatch_results["battery_charge_mw"],
                alpha=0.5, color="blue", label="Charging", bottom=dispatch_results["energy_sold_mw"])
    axes[2].set_ylabel("MW")
    axes[2].set_title("Dispatch Actions")
    axes[2].legend(loc="upper right")
    axes[2].axhline(y=0, color="black", linewidth=0.5)

    # Battery SOC
    axes[3].fill_between(dispatch_results["hour"], dispatch_results["battery_soc_mwh"],
                         alpha=0.6, color="purple")
    axes[3].axhline(y=battery_capacity * 0.95, color="red", linestyle="--", alpha=0.5, label="Max SOC")
    axes[3].axhline(y=battery_capacity * 0.10, color="red", linestyle="--", alpha=0.5, label="Min SOC")
    axes[3].set_ylabel("MWh")
    axes[3].set_xlabel("Hour")
    axes[3].set_title("Battery State of Charge")
    axes[3].legend(loc="upper right")

    plt.tight_layout()
    st.pyplot(fig)

    # Dispatch table
    st.subheader("Hourly Dispatch Schedule")
    display_cols = [
        "hour", "solar_generation_mw", "price_rs_per_mwh",
        "energy_sold_mw", "energy_bought_mw",
        "battery_charge_mw", "battery_discharge_mw", "battery_soc_mwh",
        "net_profit_rs",
    ]
    st.dataframe(
        dispatch_results[display_cols].round(2),
        use_container_width=True,
        height=400,
    )


# ── TAB 4: Profit Analysis ──────────────────────────────────────────────────
with tab4:
    st.subheader("Profit & Revenue Analysis")

    # Hourly profit
    fig, ax = plt.subplots(figsize=(14, 4))
    colors = ["green" if x >= 0 else "red" for x in dispatch_results["net_profit_rs"]]
    ax.bar(dispatch_results["hour"], dispatch_results["net_profit_rs"], color=colors, alpha=0.7)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Net Profit (Rs)")
    ax.set_title("Hourly Net Profit")
    ax.axhline(y=0, color="black", linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig)

    # Cumulative profit
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    cumulative = dispatch_results["net_profit_rs"].cumsum()
    ax2.plot(dispatch_results["hour"], cumulative, color="darkgreen", linewidth=2)
    ax2.fill_between(dispatch_results["hour"], cumulative, alpha=0.2, color="green")
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Cumulative Profit (Rs)")
    ax2.set_title("Cumulative Profit Over Time")
    plt.tight_layout()
    st.pyplot(fig2)

    # Revenue breakdown
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    breakdown = {
        "Revenue": summary["total_revenue_rs"],
        "Buy Cost": summary["total_buy_cost_rs"],
        "Penalty": summary["total_penalty_rs"],
    }
    labels = [k for k, v in breakdown.items() if v > 0]
    values = [v for v in breakdown.values() if v > 0]
    if values:
        ax3.pie(values, labels=labels, autopct="%1.1f%%", colors=["#2ecc71", "#e74c3c", "#f39c12"])
        ax3.set_title("Cost Breakdown")
    else:
        ax3.text(0.5, 0.5, "No costs incurred", ha="center", va="center")
    plt.tight_layout()
    st.pyplot(fig3)

    # Summary table
    st.subheader("Optimization Summary")
    summary_df = pd.DataFrame({
        "Metric": [
            "Total Profit", "Total Revenue", "Total Buy Cost", "Total Penalty",
            "Energy Sold", "Energy Bought", "Avg Battery SOC",
            "Hours Selling", "Hours Buying", "Hours Charging", "Hours Discharging",
        ],
        "Value": [
            f"₹{summary['total_profit_rs']:,.2f}",
            f"₹{summary['total_revenue_rs']:,.2f}",
            f"₹{summary['total_buy_cost_rs']:,.2f}",
            f"₹{summary['total_penalty_rs']:,.2f}",
            f"{summary['total_energy_sold_mwh']:.2f} MWh",
            f"{summary['total_energy_bought_mwh']:.2f} MWh",
            f"{summary['avg_soc_mwh']:.2f} MWh",
            str(summary["hours_selling"]),
            str(summary["hours_buying"]),
            str(summary["hours_charging"]),
            str(summary["hours_discharging"]),
        ],
    })
    st.table(summary_df)


# ── TAB 5: Data Explorer ────────────────────────────────────────────────────
with tab5:
    st.subheader("Raw Data Explorer")

    data_choice = st.radio("Select dataset", ["NASA POWER (Weather)", "GDAM (Market)"])

    if data_choice == "NASA POWER (Weather)":
        st.dataframe(nasa_df.head(200), use_container_width=True, height=400)
        st.markdown(f"**Total rows:** {len(nasa_df)} | **Columns:** {', '.join(nasa_df.columns[:10])}")
    else:
        st.dataframe(gdam_df.head(200), use_container_width=True, height=400)
        st.markdown(f"**Total rows:** {len(gdam_df)} | **Columns:** {', '.join(gdam_df.columns[:10])}")
