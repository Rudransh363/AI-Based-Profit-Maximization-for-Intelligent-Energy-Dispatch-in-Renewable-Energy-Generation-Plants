"""Solar generation forecasting using XGBoost."""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import logging

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def prepare_solar_features(df: pd.DataFrame) -> tuple:
    """Prepare features and target for solar forecasting."""
    feature_cols = []

    # Base weather features
    for col in ["ALLSKY_SFC_SW_DWN", "T2M", "RH2M"]:
        if col in df.columns:
            feature_cols.append(col)

    # Time features
    for col in ["hour", "weekday", "month", "day_of_year", "hour_sin", "hour_cos", "is_weekend"]:
        if col in df.columns:
            feature_cols.append(col)

    # Lag features for solar generation
    target = "solar_generation_mw"
    if target in df.columns:
        for lag in [1, 2, 3, 24]:
            col_name = f"solar_lag_{lag}"
            df[col_name] = df[target].shift(lag)
            feature_cols.append(col_name)

        # Rolling averages
        df["solar_roll_6h"] = df[target].rolling(6, min_periods=1).mean()
        df["solar_roll_24h"] = df[target].rolling(24, min_periods=1).mean()
        feature_cols.extend(["solar_roll_6h", "solar_roll_24h"])

    df = df.dropna(subset=feature_cols + [target])

    X = df[feature_cols]
    y = df[target]
    return X, y, feature_cols


def train_solar_model(df: pd.DataFrame, test_size: float = 0.2) -> dict:
    """Train XGBoost model for solar generation forecasting."""
    logger.info("Training solar generation forecast model")

    X, y, feature_cols = prepare_solar_features(df.copy())

    # Chronological split (no shuffle for time series)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    logger.info(f"Solar Model — MAE: {mae:.4f} MW, RMSE: {rmse:.4f} MW")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "solar_model.joblib")
    joblib.dump({"model": model, "features": feature_cols}, model_path)
    logger.info(f"Model saved to {model_path}")

    return {
        "model": model,
        "features": feature_cols,
        "mae": mae,
        "rmse": rmse,
        "y_test": y_test,
        "y_pred": y_pred,
        "X_test": X_test,
    }


def predict_solar(df: pd.DataFrame, model_path: str = None) -> np.ndarray:
    """Load saved model and predict solar generation."""
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, "solar_model.joblib")

    saved = joblib.load(model_path)
    model = saved["model"]
    feature_cols = saved["features"]

    X, _, _ = prepare_solar_features(df.copy())
    X = X[feature_cols]

    predictions = model.predict(X)
    return np.clip(predictions, 0, None)
