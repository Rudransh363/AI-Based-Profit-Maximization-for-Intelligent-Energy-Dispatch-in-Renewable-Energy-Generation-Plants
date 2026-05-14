"""GDAM electricity price forecasting using XGBoost."""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import logging

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def prepare_price_features(df: pd.DataFrame, target_col: str = "MCP (Rs/MWh)") -> tuple:
    """Prepare features and target for price forecasting."""
    feature_cols = []

    # Time features
    for col in ["hour", "weekday", "month", "day_of_year", "hour_sin", "hour_cos", "is_weekend"]:
        if col in df.columns:
            feature_cols.append(col)

    # Lag features
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        col_name = f"price_lag_{lag}"
        df[col_name] = df[target_col].shift(lag)
        feature_cols.append(col_name)

    # Rolling statistics
    df["price_roll_6h"] = df[target_col].rolling(6, min_periods=1).mean()
    df["price_roll_24h"] = df[target_col].rolling(24, min_periods=1).mean()
    df["price_roll_24h_std"] = df[target_col].rolling(24, min_periods=1).std().fillna(0)
    df["price_diff_1h"] = df[target_col].diff(1)
    feature_cols.extend(["price_roll_6h", "price_roll_24h", "price_roll_24h_std", "price_diff_1h"])

    df = df.dropna(subset=feature_cols + [target_col])

    X = df[feature_cols]
    y = df[target_col]
    return X, y, feature_cols


def train_price_model(df: pd.DataFrame, target_col: str = "MCP (Rs/MWh)", test_size: float = 0.2) -> dict:
    """Train XGBoost model for GDAM price forecasting."""
    logger.info("Training GDAM price forecast model")

    X, y, feature_cols = prepare_price_features(df.copy(), target_col)

    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    logger.info(f"Price Model — MAE: {mae:.2f} Rs/MWh, RMSE: {rmse:.2f} Rs/MWh")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "price_model.joblib")
    joblib.dump({"model": model, "features": feature_cols, "target_col": target_col}, model_path)
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


def predict_price(df: pd.DataFrame, model_path: str = None) -> np.ndarray:
    """Load saved model and predict prices."""
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, "price_model.joblib")

    saved = joblib.load(model_path)
    model = saved["model"]
    feature_cols = saved["features"]
    target_col = saved["target_col"]

    X, _, _ = prepare_price_features(df.copy(), target_col)
    X = X[feature_cols]

    return model.predict(X)
