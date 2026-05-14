"""Battery-aware energy dispatch optimization using PuLP."""

import pandas as pd
import numpy as np
from pulp import (
    LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, value, PULP_CBC_CMD
)
import logging

logger = logging.getLogger(__name__)

# Default battery and plant parameters
DEFAULT_PARAMS = {
    "battery_capacity_mwh": 20.0,    # total battery capacity
    "battery_initial_soc": 10.0,     # starting state of charge (MWh)
    "charge_rate_max_mw": 5.0,       # max charging rate per hour
    "discharge_rate_max_mw": 5.0,    # max discharging rate per hour
    "charge_efficiency": 0.92,       # round-trip charging efficiency
    "discharge_efficiency": 0.95,    # discharging efficiency
    "soc_min_pct": 0.10,             # minimum SOC (% of capacity)
    "soc_max_pct": 0.95,             # maximum SOC (% of capacity)
    "buy_price_premium": 1.10,       # buying price = market price * premium
    "penalty_per_mwh": 500.0,        # penalty for unmet demand (Rs/MWh)
    "plant_demand_mw": 2.0,          # plant's own consumption (MW)
}


def optimize_dispatch(
    solar_forecast: np.ndarray,
    price_forecast: np.ndarray,
    hours: int = None,
    params: dict = None,
) -> pd.DataFrame:
    """
    Optimize hourly dispatch decisions to maximize profit.

    For each hour, decide:
    - How much energy to sell to grid
    - How much to store in battery
    - How much to discharge from battery
    - How much to buy from grid (if deficit)

    Returns DataFrame with hourly dispatch schedule.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    if hours is None:
        hours = min(len(solar_forecast), len(price_forecast))

    solar = solar_forecast[:hours]
    prices = price_forecast[:hours]

    logger.info(f"Optimizing dispatch for {hours} hours")

    # Derived limits
    soc_min = p["battery_capacity_mwh"] * p["soc_min_pct"]
    soc_max = p["battery_capacity_mwh"] * p["soc_max_pct"]

    # Create optimization problem
    prob = LpProblem("EnergyDispatch", LpMaximize)

    # Decision variables for each hour
    sell = [LpVariable(f"sell_{t}", lowBound=0) for t in range(hours)]
    charge = [LpVariable(f"charge_{t}", lowBound=0, upBound=p["charge_rate_max_mw"]) for t in range(hours)]
    discharge = [LpVariable(f"discharge_{t}", lowBound=0, upBound=p["discharge_rate_max_mw"]) for t in range(hours)]
    buy = [LpVariable(f"buy_{t}", lowBound=0) for t in range(hours)]
    soc = [LpVariable(f"soc_{t}", lowBound=soc_min, upBound=soc_max) for t in range(hours)]
    unmet = [LpVariable(f"unmet_{t}", lowBound=0) for t in range(hours)]

    # Objective: maximize profit
    revenue = lpSum(sell[t] * float(prices[t]) for t in range(hours))
    buy_cost = lpSum(buy[t] * float(prices[t]) * p["buy_price_premium"] for t in range(hours))
    penalty = lpSum(unmet[t] * p["penalty_per_mwh"] for t in range(hours))

    prob += revenue - buy_cost - penalty

    for t in range(hours):
        solar_t = float(solar[t])
        demand = p["plant_demand_mw"]

        # Power balance: supply = demand
        # supply: solar + battery_discharge + grid_buy
        # demand: plant_demand + grid_sell + battery_charge + unmet_slack
        prob += (
            solar_t + discharge[t] * p["discharge_efficiency"] + buy[t]
            == demand + sell[t] + charge[t] + unmet[t]
        )

        # Sell cannot exceed total available supply
        prob += sell[t] <= solar_t + discharge[t] * p["discharge_efficiency"]

        # Battery SOC dynamics
        if t == 0:
            prob += soc[t] == p["battery_initial_soc"] + charge[t] * p["charge_efficiency"] - discharge[t]
        else:
            prob += soc[t] == soc[t - 1] + charge[t] * p["charge_efficiency"] - discharge[t]

    # Solve
    solver = PULP_CBC_CMD(msg=0)
    prob.solve(solver)

    status = LpStatus[prob.status]
    total_profit = value(prob.objective) if prob.status == 1 else 0
    logger.info(f"Optimization status: {status}, Total Profit: Rs {total_profit:,.2f}")

    # Build results DataFrame
    results = pd.DataFrame({
        "hour": range(hours),
        "solar_generation_mw": solar[:hours],
        "price_rs_per_mwh": prices[:hours],
        "energy_sold_mw": [value(sell[t]) for t in range(hours)],
        "energy_bought_mw": [value(buy[t]) for t in range(hours)],
        "battery_charge_mw": [value(charge[t]) for t in range(hours)],
        "battery_discharge_mw": [value(discharge[t]) for t in range(hours)],
        "battery_soc_mwh": [value(soc[t]) for t in range(hours)],
        "unmet_demand_mw": [value(unmet[t]) for t in range(hours)],
    })

    results["revenue_rs"] = results["energy_sold_mw"] * results["price_rs_per_mwh"]
    results["buy_cost_rs"] = results["energy_bought_mw"] * results["price_rs_per_mwh"] * p["buy_price_premium"]
    results["penalty_rs"] = results["unmet_demand_mw"] * p["penalty_per_mwh"]
    results["net_profit_rs"] = results["revenue_rs"] - results["buy_cost_rs"] - results["penalty_rs"]

    return results, total_profit, status


def get_dispatch_summary(results: pd.DataFrame, total_profit: float) -> dict:
    """Summarize dispatch optimization results."""
    return {
        "total_profit_rs": total_profit,
        "total_revenue_rs": results["revenue_rs"].sum(),
        "total_buy_cost_rs": results["buy_cost_rs"].sum(),
        "total_penalty_rs": results["penalty_rs"].sum(),
        "total_energy_sold_mwh": results["energy_sold_mw"].sum(),
        "total_energy_bought_mwh": results["energy_bought_mw"].sum(),
        "avg_soc_mwh": results["battery_soc_mwh"].mean(),
        "peak_soc_mwh": results["battery_soc_mwh"].max(),
        "hours_selling": (results["energy_sold_mw"] > 0.01).sum(),
        "hours_buying": (results["energy_bought_mw"] > 0.01).sum(),
        "hours_charging": (results["battery_charge_mw"] > 0.01).sum(),
        "hours_discharging": (results["battery_discharge_mw"] > 0.01).sum(),
    }
