"""
Hybrid Model Training Script
============================
Train LightGBM + LSTM and LightGBM + GRU hybrid models using real Karnataka
SLDC load data and real Bengaluru weather data.

Default real dataset:
- `data/karnataka_realdata.csv`

Usage:
    python train_hybrid_models.py
    python train_hybrid_models.py --data data/karnataka_realdata.csv --epochs 150
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, GRU, Input, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

np.random.seed(42)
tf.random.set_seed(42)

DEFAULT_DATA_CANDIDATES = [
    "data/karnataka_realdata.csv",
    "backend/data/karnataka_realdata.csv",
]

REAL_SIGNAL_COLUMNS = [
    "temperature_celsius",
    "humidity_percent",
    "precipitation_mm",
    "weather_code",
    "is_holiday",
]

KARNATAKA_HOLIDAYS = [
    "2020-01-15", "2020-01-26", "2020-03-25", "2020-04-02", "2020-08-15", "2020-10-02", "2020-11-01", "2020-11-14", "2020-12-25",
    "2021-01-14", "2021-01-26", "2021-04-13", "2021-04-21", "2021-08-15", "2021-10-02", "2021-11-01", "2021-11-04", "2021-12-25",
    "2022-01-14", "2022-01-26", "2022-04-02", "2022-04-10", "2022-08-15", "2022-10-02", "2022-11-01", "2022-10-24", "2022-12-25",
    "2023-01-15", "2023-01-26", "2023-03-22", "2023-03-30", "2023-08-15", "2023-10-02", "2023-11-01", "2023-11-12", "2023-12-25",
    "2024-01-15", "2024-01-26", "2024-04-09", "2024-04-17", "2024-08-15", "2024-10-02", "2024-11-01", "2024-10-31", "2024-12-25",
    "2025-01-14", "2025-01-26", "2025-03-30", "2025-04-06", "2025-08-15", "2025-10-02", "2025-11-01", "2025-10-20", "2025-12-25",
    "2026-01-14", "2026-01-26", "2026-03-19", "2026-03-27", "2026-08-15", "2026-10-02", "2026-11-01", "2026-11-08", "2026-12-25",
]

class FeatureEngineer:
    """Feature engineering driven by the true source cadence."""

    def __init__(self, samples_per_day: int):
        self.samples_per_day = samples_per_day
        self.lag_periods = sorted(set([1, 2, 3, samples_per_day//4, samples_per_day//2, samples_per_day, samples_per_day*2, samples_per_day*7]))
        half_day = max(2, samples_per_day // 2)
        self.rolling_windows = sorted(set([max(2, half_day // 2), half_day, samples_per_day]))
        self.exogenous_rolling_windows = sorted(set([max(2, half_day // 2), half_day]))

    def create_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
        """Create leakage-safe temporal and exogenous features."""
        df = df.copy()

        df["hour"] = df.index.hour
        df["dayofweek"] = df.index.dayofweek
        df["dayofmonth"] = df.index.day
        df["month"] = df.index.month
        df["dayofyear"] = df.index.dayofyear
        df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

        if self.samples_per_day > 24:
            minute_slot = (df.index.hour * 60 + df.index.minute) / (24 * 60)
            df["minute"] = df.index.minute
            df["time_slot"] = np.arange(len(df.index)) % self.samples_per_day
        else:
            minute_slot = df.index.hour / 24.0
            df["minute"] = 0
            df["time_slot"] = df.index.hour

        df["hour_sin"] = np.sin(2 * np.pi * minute_slot)
        df["hour_cos"] = np.cos(2 * np.pi * minute_slot)
        df["day_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 366)
        df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 366)

        feature_cols = [
            "hour",
            "minute",
            "dayofweek",
            "dayofmonth",
            "month",
            "dayofyear",
            "is_weekend",
            "time_slot",
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "month_sin",
            "month_cos",
            "dayofyear_sin",
            "dayofyear_cos",
        ]
        
        holiday_dates = pd.to_datetime(KARNATAKA_HOLIDAYS)
        df["is_karnataka_holiday"] = df.index.normalize().isin(holiday_dates).astype(int)
        feature_cols.append("is_karnataka_holiday")

        for lag in self.lag_periods:
            df[f"lag_{lag}"] = df["load"].shift(lag)
            feature_cols.append(f"lag_{lag}")

        for window in self.rolling_windows:
            shifted = df["load"].shift(1)
            for stat in ["mean", "std", "min", "max"]:
                col = f"rolling_{stat}_{window}"
                if stat == "mean":
                    df[col] = shifted.rolling(window=window).mean()
                elif stat == "std":
                    df[col] = shifted.rolling(window=window).std()
                elif stat == "min":
                    df[col] = shifted.rolling(window=window).min()
                else:
                    df[col] = shifted.rolling(window=window).max()
                feature_cols.append(col)

        prev_load = df["load"].shift(1)
        df["diff_1"] = prev_load.diff(1)
        df[f"diff_{self.samples_per_day}"] = prev_load.diff(self.samples_per_day)
        feature_cols.extend(["diff_1", f"diff_{self.samples_per_day}"])

        for col in REAL_SIGNAL_COLUMNS:
            if col not in df.columns:
                continue

            if col == "is_holiday":
                df[col] = (
                    df[col].astype(str).str.strip().str.lower().map(
                        {"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0}
                    )
                ).fillna(pd.to_numeric(df[col], errors="coerce")).fillna(0).astype(int)
                feature_cols.append(col)
                continue

            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[f"{col}_lag_1"] = df[col].shift(1)
            feature_cols.extend([col, f"{col}_lag_1"])

            for window in self.exogenous_rolling_windows:
                roll_col = f"{col}_rolling_mean_{window}"
                df[roll_col] = df[col].shift(1).rolling(window=window).mean()
                feature_cols.append(roll_col)
                
        # Temperature-load interaction (heat/cool stress)
        if "temperature_celsius" in df.columns:
            df["temp_sq"] = df["temperature_celsius"] ** 2
            df["heat_stress"] = (df["temperature_celsius"] - 28).clip(lower=0)  # cooling load above 28°C
            df["cold_stress"] = (18 - df["temperature_celsius"]).clip(lower=0)  # heating load below 18°C
            df["temp_hour_interaction"] = df["temperature_celsius"] * df["hour_sin"]
            feature_cols.extend(["temp_sq", "heat_stress", "cold_stress", "temp_hour_interaction"])

        # Humidity discomfort index
        if "humidity_percent" in df.columns and "temperature_celsius" in df.columns:
            df["discomfort_index"] = df["temperature_celsius"] - 0.55 * (1 - df["humidity_percent"] / 100) * (df["temperature_celsius"] - 14.5)
            df["temp_humidity_interaction"] = df["temperature_celsius"] * df["humidity_percent"] / 100
            feature_cols.extend(["discomfort_index", "temp_humidity_interaction"])

        # Seasonal dummies (Karnataka: Summer=Mar-May, Monsoon=Jun-Sep, Winter=Oct-Feb)
        df["is_summer"] = df.index.month.isin([3, 4, 5]).astype(int)
        df["is_monsoon"] = df.index.month.isin([6, 7, 8, 9]).astype(int)
        df["is_winter"] = df.index.month.isin([10, 11, 12, 1, 2]).astype(int)
        df["season_hour_sin"] = df["is_summer"] * df["hour_sin"] - df["is_monsoon"] * df["hour_cos"]
        feature_cols.extend(["is_summer", "is_monsoon", "is_winter", "season_hour_sin"])

        # Peak period flags (Bengaluru BESCOM morning and evening peaks)
        df["is_morning_peak"] = ((df.index.hour >= 9) & (df.index.hour <= 12)).astype(int)
        df["is_evening_peak"] = ((df.index.hour >= 18) & (df.index.hour <= 22)).astype(int)
        df["is_off_peak"] = ((df.index.hour >= 0) & (df.index.hour <= 5)).astype(int)
        feature_cols.extend(["is_morning_peak", "is_evening_peak", "is_off_peak"])

        return df, feature_cols


def resolve_data_path(requested_path: str | None) -> str:
    """Resolve the most appropriate real dataset path."""
    if requested_path and os.path.exists(requested_path):
        return requested_path

    search_paths = [requested_path] if requested_path else []
    search_paths.extend(DEFAULT_DATA_CANDIDATES)

    for candidate in search_paths:
        if candidate and os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "No real dataset found. Expected one of: " + ", ".join(DEFAULT_DATA_CANDIDATES)
    )


def _infer_frequency_minutes(index: pd.DatetimeIndex) -> int:
    if len(index) < 2:
        raise ValueError("Dataset has too few timestamps to infer its frequency.")
    deltas = index.to_series().diff().dropna().dt.total_seconds() / 60
    if deltas.empty:
        raise ValueError("Could not infer timestamp frequency.")
    return int(round(float(deltas.mode().iloc[0])))


def _filter_complete_days(df: pd.DataFrame, samples_per_day: int) -> pd.DataFrame:
    counts = df.groupby(df.index.normalize()).size()
    full_days = counts[counts == samples_per_day].index
    filtered = df[df.index.normalize().isin(full_days)].copy()
    return filtered.sort_index()


def _normalise_dataset(raw_df: pd.DataFrame, source_path: str) -> tuple[pd.DataFrame, dict]:
    """Normalize the real combined dataset."""
    lower_cols = {str(col).strip().lower(): col for col in raw_df.columns}
    dataset_name = os.path.basename(source_path)

    time_col = lower_cols.get("timestamp") or lower_cols.get("datetime")
    load_col = lower_cols.get("load_mw") or lower_cols.get("load")
    if not time_col or not load_col:
        raise ValueError(f"Expected timestamp/datetime and load_mw/load columns in {source_path}")

    keep_cols = [time_col, load_col]
    for col in REAL_SIGNAL_COLUMNS:
        if col in lower_cols:
            keep_cols.append(lower_cols[col])

    data = raw_df[keep_cols].copy()
    rename_map = {time_col: "timestamp", load_col: "load"}
    for col in REAL_SIGNAL_COLUMNS:
        if col in lower_cols:
            rename_map[lower_cols[col]] = col

    data = data.rename(columns=rename_map)
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["load"] = pd.to_numeric(data["load"], errors="coerce")
    for col in REAL_SIGNAL_COLUMNS:
        if col in data.columns and col != "is_holiday":
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["timestamp", "load"]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    data = data.set_index("timestamp")

    frequency_minutes = _infer_frequency_minutes(data.index)
    if 1440 % frequency_minutes != 0:
        raise ValueError(f"Unsupported cadence {frequency_minutes} minutes from {source_path}")

    samples_per_day = 1440 // frequency_minutes
    data = _filter_complete_days(data, samples_per_day)

    if len(data) < samples_per_day * 30:
        raise ValueError(
            "Need at least 30 complete days of real data for training. "
            f"Found {len(data) / samples_per_day:.1f} complete days in {source_path}."
        )

    required_weather = [col for col in REAL_SIGNAL_COLUMNS if col in data.columns]
    data = data.dropna(subset=["load"] + [col for col in required_weather if col != "is_holiday"])

    info = {
        "dataset_name": dataset_name,
        "source_type": "real_karnataka_load_plus_weather",
        "frequency_minutes": frequency_minutes,
        "samples_per_day": samples_per_day,
        "available_signal_columns": [col for col in REAL_SIGNAL_COLUMNS if col in data.columns],
    }
    return data, info


def load_data(data_path: str) -> tuple[pd.DataFrame, dict]:
    """Load and validate the real dataset."""
    print(f"Loading data from {data_path}...")
    raw = pd.read_csv(data_path)
    data, info = _normalise_dataset(raw, data_path)

    print(f"  Loaded {len(data)} samples")
    print(f"  Date range: {data.index.min()} to {data.index.max()}")
    print(f"  Load range: {data['load'].min():.2f} - {data['load'].max():.2f} MW")
    print(f"  Frequency: {info['frequency_minutes']} minutes")
    print(f"  Samples per day: {info['samples_per_day']}")
    print(f"  Real signal columns used: {', '.join(info['available_signal_columns'])}")
    return data, info


def train_lightgbm(X_train, y_train, X_val, y_val, feature_cols, nn_type: str = "lstm"):
    """Train a regularized LightGBM model with per-hybrid hyperparameters."""
    print(f"\nTraining LightGBM model for {nn_type.upper()} base...")

    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    lgb_val = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=lgb_train)

    if nn_type == "lstm":
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": 0.01,
            "num_leaves": 63,
            "max_depth": -1,
            "min_data_in_leaf": 30,
            "feature_fraction": 0.80,
            "bagging_fraction": 0.80,
            "bagging_freq": 5,
            "lambda_l1": 0.1,
            "lambda_l2": 0.3,
            "min_gain_to_split": 0.01,
            "verbose": -1,
            "seed": 42,
        }
    else:  # gru
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": 0.008,
            "num_leaves": 95,
            "max_depth": -1,
            "min_data_in_leaf": 30,
            "feature_fraction": 0.80,
            "bagging_fraction": 0.80,
            "bagging_freq": 5,
            "lambda_l1": 0.1,
            "lambda_l2": 0.3,
            "min_gain_to_split": 0.01,
            "verbose": -1,
            "seed": 42,
        }

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=5000,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=150), lgb.log_evaluation(period=100)],
    )
    print(f"  Best iteration: {model.best_iteration}")
    return model


def build_lstm_model(nlags: int) -> Sequential:
    """Build a regularized LSTM model for residual prediction."""
    model = Sequential(
        [
            Input(shape=(nlags, 1)),
            LSTM(128, return_sequences=True, dropout=0.10, recurrent_dropout=0.05, kernel_regularizer=regularizers.l2(1e-4)),
            LSTM(64, return_sequences=True, dropout=0.10, recurrent_dropout=0.05, kernel_regularizer=regularizers.l2(1e-4)),
            LSTM(32, return_sequences=False, dropout=0.10, recurrent_dropout=0.0, kernel_regularizer=regularizers.l2(1e-4)),
            Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
            Dropout(0.10),
            Dense(1),
        ]
    )
    model.compile(loss="mse", optimizer=Adam(learning_rate=1e-3))
    return model


def build_gru_model(nlags: int) -> Sequential:
    """Build a regularized GRU model for residual prediction."""
    model = Sequential(
        [
            Input(shape=(nlags, 1)),
            GRU(128, return_sequences=True, dropout=0.10, recurrent_dropout=0.05, kernel_regularizer=regularizers.l2(1e-4)),
            GRU(64, return_sequences=True, dropout=0.10, recurrent_dropout=0.05, kernel_regularizer=regularizers.l2(1e-4)),
            GRU(32, return_sequences=False, dropout=0.10, recurrent_dropout=0.0, kernel_regularizer=regularizers.l2(1e-4)),
            Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
            Dropout(0.10),
            Dense(1),
        ]
    )
    model.compile(loss="mse", optimizer=Adam(learning_rate=1e-3))
    return model


def create_sequences(data: np.ndarray, nlags: int):
    """Create sequences for RNN training."""
    X, y = [], []
    for i in range(nlags, len(data)):
        X.append(data[i - nlags : i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def split_time_series(X, y, train_ratio=0.65, val_ratio=0.20, gap=24):
    """Split time series data with gaps between splits."""
    total = len(X)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    train_end = min(train_end, total - (2 * gap + 1))
    val_start = min(train_end + gap, total - 1)
    val_end = min(val_end, total - gap)
    test_start = min(val_end + gap, total)

    X_train = X[:train_end]
    y_train = y[:train_end]
    X_val = X[val_start:val_end]
    y_val = y[val_start:val_end]
    X_test = X[test_start:]
    y_test = y[test_start:]

    if min(len(X_train), len(X_val), len(X_test)) <= gap:
        raise ValueError("Dataset is too small for leakage-safe train/validation/test splitting.")

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_hybrid_model(data_path: str, model_dir: str, epochs: int = 150):
    """Train LightGBM + LSTM and LightGBM + GRU hybrids."""
    data, dataset_info = load_data(data_path)
    engineer = FeatureEngineer(samples_per_day=dataset_info["samples_per_day"])
    data_featured, feature_cols = engineer.create_features(data)
    data_featured = data_featured.dropna()

    X = data_featured[feature_cols].values
    y = data_featured["load"].values

    gap = max(engineer.lag_periods)
    X_train, y_train, X_val, y_val, X_test, y_test = split_time_series(
        X, y, train_ratio=0.65, val_ratio=0.20, gap=gap
    )

    print("\nData split:")
    print(f"  Train: {len(X_train)}")
    print(f"  Validation: {len(X_val)}")
    print(f"  Test: {len(X_test)}")
    print(f"  Gap between splits: {gap} samples")

    # Fix 1: Train two separate LightGBM boosters with per-hybrid hyperparameters
    lgb_lstm_model = train_lightgbm(X_train, y_train, X_val, y_val, feature_cols, nn_type="lstm")
    lgb_gru_model  = train_lightgbm(X_train, y_train, X_val, y_val, feature_cols, nn_type="gru")

    # Fix 2: Compute separate residuals for each hybrid
    lstm_pred_train = lgb_lstm_model.predict(X_train, num_iteration=lgb_lstm_model.best_iteration)
    lstm_pred_val   = lgb_lstm_model.predict(X_val,   num_iteration=lgb_lstm_model.best_iteration)
    lstm_pred_test  = lgb_lstm_model.predict(X_test,  num_iteration=lgb_lstm_model.best_iteration)

    gru_pred_train = lgb_gru_model.predict(X_train, num_iteration=lgb_gru_model.best_iteration)
    gru_pred_val   = lgb_gru_model.predict(X_val,   num_iteration=lgb_gru_model.best_iteration)
    gru_pred_test  = lgb_gru_model.predict(X_test,  num_iteration=lgb_gru_model.best_iteration)

    lstm_residuals_train = y_train - lstm_pred_train
    lstm_residuals_val   = y_val   - lstm_pred_val
    lstm_residuals_test  = y_test  - lstm_pred_test

    gru_residuals_train = y_train - gru_pred_train
    gru_residuals_val   = y_val   - gru_pred_val
    gru_residuals_test  = y_test  - gru_pred_test

    # Fix 2: Fit separate residual scalers per hybrid
    lstm_residual_scaler = MinMaxScaler(feature_range=(-1, 1))
    lstm_residuals_train_scaled = lstm_residual_scaler.fit_transform(lstm_residuals_train.reshape(-1, 1))
    lstm_residuals_val_scaled   = lstm_residual_scaler.transform(lstm_residuals_val.reshape(-1, 1))
    lstm_residuals_test_scaled  = lstm_residual_scaler.transform(lstm_residuals_test.reshape(-1, 1))

    gru_residual_scaler = MinMaxScaler(feature_range=(-1, 1))
    gru_residuals_train_scaled = gru_residual_scaler.fit_transform(gru_residuals_train.reshape(-1, 1))
    gru_residuals_val_scaled   = gru_residual_scaler.transform(gru_residuals_val.reshape(-1, 1))
    gru_residuals_test_scaled  = gru_residual_scaler.transform(gru_residuals_test.reshape(-1, 1))

    nlags = min(dataset_info["samples_per_day"] * 7, 168)
    nlags = max(24, nlags)

    # Fix 2: Build separate sequence arrays from each hybrid's residuals
    lstm_X_train, lstm_y_train = create_sequences(lstm_residuals_train_scaled, nlags)
    lstm_X_val,   lstm_y_val   = create_sequences(lstm_residuals_val_scaled,   nlags)
    lstm_X_test,  _            = create_sequences(lstm_residuals_test_scaled,  nlags)

    gru_X_train, gru_y_train = create_sequences(gru_residuals_train_scaled, nlags)
    gru_X_val,   gru_y_val   = create_sequences(gru_residuals_val_scaled,   nlags)
    gru_X_test,  _           = create_sequences(gru_residuals_test_scaled,  nlags)

    lstm_X_train = lstm_X_train.reshape(-1, nlags, 1)
    lstm_X_val   = lstm_X_val.reshape(-1, nlags, 1)
    lstm_X_test  = lstm_X_test.reshape(-1, nlags, 1)

    gru_X_train = gru_X_train.reshape(-1, nlags, 1)
    gru_X_val   = gru_X_val.reshape(-1, nlags, 1)
    gru_X_test  = gru_X_test.reshape(-1, nlags, 1)

    os.makedirs(model_dir, exist_ok=True)

    lstm_callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6, verbose=1),
        ModelCheckpoint(os.path.join(model_dir, "lstm_model.keras"), monitor="val_loss", save_best_only=True, verbose=1)
    ]
    
    gru_callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6, verbose=1),
        ModelCheckpoint(os.path.join(model_dir, "gru_model.keras"), monitor="val_loss", save_best_only=True, verbose=1)
    ]

    print("\nTraining LSTM residual model...")
    lstm_model = build_lstm_model(nlags)
    lstm_history = lstm_model.fit(
        lstm_X_train,
        lstm_y_train,
        epochs=epochs,
        batch_size=64,
        validation_data=(lstm_X_val, lstm_y_val),
        callbacks=lstm_callbacks,
        verbose=1,
        shuffle=False,
    )

    # Fix 3: GRU trains on its own residuals (gru_X_train/val), not LSTM's
    print("\nTraining GRU residual model...")
    gru_model = build_gru_model(nlags)
    gru_history = gru_model.fit(
        gru_X_train,
        gru_y_train,
        epochs=epochs,
        batch_size=64,
        validation_data=(gru_X_val, gru_y_val),
        callbacks=gru_callbacks,
        verbose=1,
        shuffle=False,
    )

    def calc_metrics(y_true, y_pred, name):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        print(f"  {name}: RMSE={rmse:.2f} MW, MAPE={mape:.2f}%")
        return {"rmse": float(rmse), "mape": float(mape)}

    print("\nTest Set Results:")
    lstm_nn_pred     = lstm_model.predict(lstm_X_test, verbose=0)
    lstm_pred_inv    = lstm_residual_scaler.inverse_transform(lstm_nn_pred).flatten()
    lstm_lgb_aligned = lstm_pred_test[nlags:]
    y_test_aligned   = y_test[nlags:]
    lstm_hybrid_pred = lstm_lgb_aligned + lstm_pred_inv

    gru_nn_pred     = gru_model.predict(gru_X_test, verbose=0)
    gru_pred_inv    = gru_residual_scaler.inverse_transform(gru_nn_pred).flatten()
    gru_lgb_aligned = gru_pred_test[nlags:]
    gru_hybrid_pred = gru_lgb_aligned + gru_pred_inv

    lgb_metrics  = calc_metrics(y_test_aligned, lstm_lgb_aligned, "LightGBM (LSTM base) Only")
    lstm_metrics = calc_metrics(y_test_aligned, lstm_hybrid_pred, "LightGBM + LSTM")
    gru_metrics  = calc_metrics(y_test_aligned, gru_hybrid_pred,  "LightGBM + GRU")

    # Fix 4: Build per-hybrid metadata with their own scalers and best iterations
    metadata_base = {
        "feature_cols": feature_cols,
        "lag_periods": engineer.lag_periods,
        "rolling_windows": engineer.rolling_windows,
        "exogenous_rolling_windows": engineer.exogenous_rolling_windows,
        "frequency_minutes": dataset_info["frequency_minutes"],
        "samples_per_day": dataset_info["samples_per_day"],
        "nlags": nlags,
        "train_date": datetime.now().isoformat(),
        "dataset_name": dataset_info["dataset_name"],
        "source_type": dataset_info["source_type"],
        "available_signal_columns": dataset_info["available_signal_columns"],
        "training_period_start": data.index.min().isoformat(),
        "training_period_end": data.index.max().isoformat(),
        "training_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "feature_count": int(len(feature_cols)),
        "split_gap_samples": int(gap),
        "metrics": {
            "lightgbm": lgb_metrics,
            "lstm_hybrid": lstm_metrics,
            "gru_hybrid": gru_metrics,
        },
        "training_history": {
            "lstm_best_val_loss": float(min(lstm_history.history["val_loss"])),
            "gru_best_val_loss": float(min(gru_history.history["val_loss"])),
            "lstm_epochs_ran": int(len(lstm_history.history["loss"])),
            "gru_epochs_ran": int(len(gru_history.history["loss"])),
        },
    }

    lstm_metadata = {
        **metadata_base,
        "residual_scaler": lstm_residual_scaler,
        "lightgbm_best_iteration": int(lgb_lstm_model.best_iteration),
    }
    gru_metadata = {
        **metadata_base,
        "residual_scaler": gru_residual_scaler,
        "lightgbm_best_iteration": int(lgb_gru_model.best_iteration),
    }

    # Fix 4: Save separate LightGBM models per hybrid
    lgb_lstm_model.save_model(os.path.join(model_dir, "lgb_lstm_model.txt"))
    lgb_gru_model.save_model(os.path.join(model_dir, "lgb_gru_model.txt"))
    lgb_lstm_model.save_model(os.path.join(model_dir, "lgb_model.txt"))  # fallback alias

    # Fix 4: Save per-hybrid metadata
    joblib.dump(lstm_metadata, os.path.join(model_dir, "lstm_metadata.joblib"))
    joblib.dump(gru_metadata,  os.path.join(model_dir, "gru_metadata.joblib"))
    joblib.dump(lstm_metadata, os.path.join(model_dir, "metadata.joblib"))  # shared fallback

    with open(os.path.join(model_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_base, f, indent=2, default=str)

    print("\nSaved models and metadata.")
    print(f"  lgb_lstm_model.txt  (LSTM-base LightGBM)")
    print(f"  lgb_gru_model.txt   (GRU-base LightGBM)")
    print(f"  lgb_model.txt       (fallback alias -> LSTM base)")
    print(f"  lstm_model.keras    (LSTM residual model)")
    print(f"  gru_model.keras     (GRU residual model)")
    print(f"  lstm_metadata.joblib / gru_metadata.joblib / metadata.joblib")
    return lgb_lstm_model, lgb_gru_model, lstm_model, gru_model, lstm_metadata, gru_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hybrid models on real Karnataka load data")
    parser.add_argument("--data", type=str, default=None, help="Path to the real training CSV")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--epochs", type=int, default=150, help="Maximum neural network epochs")
    args = parser.parse_args()

    try:
        resolved_data_path = resolve_data_path(args.data)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    train_hybrid_model(resolved_data_path, args.model_dir, args.epochs)
