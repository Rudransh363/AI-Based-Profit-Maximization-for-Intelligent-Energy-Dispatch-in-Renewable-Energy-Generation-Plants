"""Data loading and preprocessing utilities for GDAM and NASA POWER datasets."""

import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Location: Jaipur, Rajasthan (from NASA POWER header)
LATITUDE = 26.9124
LONGITUDE = 75.7873
PLANT_CAPACITY_MW = 10  # assumed 10 MW solar plant


def load_nasa_power(filepath: str) -> pd.DataFrame:
    """Load NASA POWER CSV, skip metadata header, parse datetime."""
    logger.info(f"Loading NASA POWER data from {filepath}")

    # Count header lines (lines before -END HEADER-)
    skip = 0
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            if "-END HEADER-" in line:
                skip = i + 1
                break

    df = pd.read_csv(filepath, skiprows=skip)
    df.columns = df.columns.str.strip()

    df["datetime"] = pd.to_datetime(
        df["YEAR"].astype(str) + "-" + df["MO"].astype(str).str.zfill(2) + "-" +
        df["DY"].astype(str).str.zfill(2) + " " + df["HR"].astype(str).str.zfill(2) + ":00:00"
    )
    df = df.set_index("datetime").sort_index()
    df = df[["ALLSKY_SFC_SW_DWN", "T2M", "RH2M"]]

    # Replace -999 with NaN
    df = df.replace(-999.0, np.nan)

    logger.info(f"NASA data loaded: {len(df)} rows, range: {df.index.min()} to {df.index.max()}")
    return df


def synthesize_irradiance(df: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic clear-sky irradiance when real data is missing (-999)."""
    if df["ALLSKY_SFC_SW_DWN"].isna().sum() == 0:
        logger.info("Irradiance data present, no synthesis needed")
        return df

    logger.info("Synthesizing irradiance using clear-sky model (real data unavailable)")

    hours = df.index.hour
    day_of_year = df.index.dayofyear
    lat_rad = np.radians(LATITUDE)

    # Solar declination angle
    declination = 23.45 * np.sin(np.radians(360 / 365 * (day_of_year - 81)))
    decl_rad = np.radians(declination)

    # Hour angle (solar noon = 0)
    solar_hour = hours - 12
    hour_angle_rad = np.radians(15 * solar_hour)

    # Solar elevation angle
    sin_elevation = (
        np.sin(lat_rad) * np.sin(decl_rad) +
        np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hour_angle_rad)
    )
    sin_elevation = np.clip(sin_elevation, 0, 1)

    # Clear sky GHI model (simplified)
    max_ghi = 1000  # W/m^2 peak
    ghi = max_ghi * sin_elevation

    # Add some realistic variation using humidity (higher humidity = more clouds)
    if "RH2M" in df.columns:
        cloud_factor = 1 - 0.3 * (df["RH2M"].fillna(50) / 100)
        ghi = ghi * cloud_factor

    # Add small random noise for realism
    np.random.seed(42)
    noise = np.random.normal(1.0, 0.05, len(ghi))
    ghi = np.clip(ghi * noise, 0, None)

    df["ALLSKY_SFC_SW_DWN"] = ghi.round(2)
    logger.info("Irradiance synthesis complete")
    return df


def estimate_solar_generation(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate solar generation from irradiance (no real plant data available)."""
    irradiance = df["ALLSKY_SFC_SW_DWN"].fillna(0)

    # Normalize: peak irradiance ~1000 W/m^2 maps to plant capacity
    peak_irradiance = 1000.0
    df["solar_generation_mw"] = (irradiance / peak_irradiance * PLANT_CAPACITY_MW).clip(lower=0).round(4)
    df["solar_generation_mw"] = df["solar_generation_mw"].clip(upper=PLANT_CAPACITY_MW)

    logger.info(f"Solar generation estimated: peak={df['solar_generation_mw'].max():.2f} MW")
    return df


def load_gdam_files(filepaths: list) -> pd.DataFrame:
    """Load and combine multiple GDAM Market Snapshot Excel files."""
    logger.info(f"Loading {len(filepaths)} GDAM file(s)")
    all_dfs = []

    for fp in filepaths:
        logger.info(f"Reading {os.path.basename(fp)}")
        raw = pd.read_excel(fp, header=None)

        # Find the header row containing 'Date' and 'Hour'
        header_row = None
        for i in range(min(10, len(raw))):
            row_vals = raw.iloc[i].astype(str).str.strip().tolist()
            if "Date" in row_vals and "Hour" in row_vals:
                header_row = i
                break

        if header_row is None:
            logger.warning(f"Could not find header in {fp}, skipping")
            continue

        df = pd.read_excel(fp, header=header_row)
        df.columns = df.columns.str.strip()

        # Drop fully empty rows
        df = df.dropna(how="all")

        # Keep only rows with valid Date and Hour
        df = df.dropna(subset=["Date", "Hour"])

        all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No valid GDAM data found")

    combined = pd.concat(all_dfs, ignore_index=True)

    # Parse datetime
    combined["Date"] = pd.to_datetime(combined["Date"], format="%d-%m-%Y", errors="coerce")
    combined = combined.dropna(subset=["Hour"])
    combined["Hour"] = pd.to_numeric(combined["Hour"], errors="coerce")
    combined = combined.dropna(subset=["Hour"])
    combined["Hour"] = combined["Hour"].astype(int)
    combined["datetime"] = combined["Date"] + pd.to_timedelta(combined["Hour"] - 1, unit="h")

    # Convert price columns to numeric
    for col in combined.columns:
        if col not in ["Date", "Hour", "datetime"]:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined = combined.set_index("datetime").sort_index()

    # Remove exact duplicates
    combined = combined[~combined.index.duplicated(keep="first")]

    logger.info(f"GDAM data loaded: {len(combined)} rows, range: {combined.index.min()} to {combined.index.max()}")
    return combined


def reindex_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to continuous hourly timestamps and fill gaps."""
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")
    missing_count = len(full_idx) - len(df)
    if missing_count > 0:
        logger.info(f"Filling {missing_count} missing hourly timestamps")

    df = df.reindex(full_idx)
    df.index.name = "datetime"

    # Forward fill then interpolate for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features useful for forecasting."""
    df["hour"] = df.index.hour
    df["weekday"] = df.index.weekday
    df["month"] = df.index.month
    df["day_of_year"] = df.index.dayofyear
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    # Cyclical encoding of hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df


def preprocess_nasa(filepath: str) -> pd.DataFrame:
    """Full preprocessing pipeline for NASA POWER data."""
    df = load_nasa_power(filepath)
    df = synthesize_irradiance(df)
    df = reindex_hourly(df)
    df = estimate_solar_generation(df)
    df = add_time_features(df)
    return df


def preprocess_gdam(filepaths: list) -> pd.DataFrame:
    """Full preprocessing pipeline for GDAM data."""
    df = load_gdam_files(filepaths)
    df = reindex_hourly(df)
    df = add_time_features(df)
    return df
