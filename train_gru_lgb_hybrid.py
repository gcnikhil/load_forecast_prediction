"""
Reversed Hybrid Model Training Script: GRU → LightGBM (Multi-step version)
==========================================================================
Architecture:
  Stage 1 → GRU predicts `horizon` future steps simultaneously from `seq_len` past steps.
  Stage 2 → LightGBM corrects GRU residuals for all horizons using global cross-learning.
            It uses actual historical load (anchored at t) and future time features (at t+h).

Usage
-----
    python train_gru_lgb_hybrid.py
    python train_gru_lgb_hybrid.py --data data/karnataka_realdata.csv --epochs 80 --horizon 72
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import math
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.layers import Dense, GRU, Input, Dropout, Bidirectional, MultiHeadAttention, LayerNormalization, Add, Reshape, Flatten, Concatenate, Multiply, Lambda, Permute
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATA_CANDIDATES = [
    "data/karnataka_realdata.csv",
    "backend/data/karnataka_realdata.csv",
    "load_forecast_prediction/backend/data/karnataka_realdata.csv",
]

REAL_SIGNAL_COLUMNS = [
    "temperature_celsius",
    "humidity_percent",
    "precipitation_mm",
    "weather_code",
    "is_holiday",
]


# ---------------------------------------------------------------------------
# Data Loading & Normalisation  (mirrors train_hybrid_models.py)
# ---------------------------------------------------------------------------

def resolve_data_path(requested_path: str | None) -> str:
    """Return the first existing candidate path."""
    candidates = ([requested_path] if requested_path else []) + DEFAULT_DATA_CANDIDATES
    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(
        "No real dataset found. Expected one of: " + ", ".join(DEFAULT_DATA_CANDIDATES)
    )


def _infer_frequency_minutes(index: pd.DatetimeIndex) -> int:
    deltas = index.to_series().diff().dropna().dt.total_seconds() / 60
    return int(round(float(deltas.mode().iloc[0])))


def _filter_complete_days(df: pd.DataFrame, spd: int) -> pd.DataFrame:
    counts = df.groupby(df.index.normalize()).size()
    full_days = counts[counts == spd].index
    return df[df.index.normalize().isin(full_days)].sort_index()


def _normalise_dataset(raw: pd.DataFrame, source: str):
    lower = {str(c).strip().lower(): c for c in raw.columns}
    time_col = lower.get("timestamp") or lower.get("datetime")
    load_col = lower.get("load_mw") or lower.get("load")
    if not time_col or not load_col:
        raise ValueError(f"Need timestamp/datetime + load_mw/load columns in {source}")

    keep = [time_col, load_col]
    for col in REAL_SIGNAL_COLUMNS:
        if col in lower:
            keep.append(lower[col])

    data = raw[keep].copy()
    rmap = {time_col: "timestamp", load_col: "load"}
    for col in REAL_SIGNAL_COLUMNS:
        if col in lower:
            rmap[lower[col]] = col
    data = data.rename(columns=rmap)

    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["load"] = pd.to_numeric(data["load"], errors="coerce")
    for col in REAL_SIGNAL_COLUMNS:
        if col in data.columns and col != "is_holiday":
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data = (
        data.dropna(subset=["timestamp", "load"])
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .set_index("timestamp")
    )

    freq_min = _infer_frequency_minutes(data.index)
    if 1440 % freq_min != 0:
        raise ValueError(f"Unsupported cadence {freq_min} min")
    spd = 1440 // freq_min
    data = _filter_complete_days(data, spd)

    if len(data) < spd * 30:
        raise ValueError(
            f"Need ≥30 complete days; found {len(data)/spd:.1f} in {source}"
        )

    weather_required = [c for c in REAL_SIGNAL_COLUMNS if c in data.columns and c != "is_holiday"]
    data = data.dropna(subset=["load"] + weather_required)

    info = {
        "dataset_name": os.path.basename(source),
        "frequency_minutes": freq_min,
        "samples_per_day": spd,
        "available_signal_columns": [c for c in REAL_SIGNAL_COLUMNS if c in data.columns],
    }
    return data, info


def load_data(path: str):
    print(f"\nLoading data from {path} ...")
    raw = pd.read_csv(path)
    data, info = _normalise_dataset(raw, path)
    spd = info["samples_per_day"]
    print(f"  Rows        : {len(data)}")
    print(f"  Date range  : {data.index.min()} to {data.index.max()}")
    print(f"  Load range  : {data['load'].min():.1f} - {data['load'].max():.1f} MW")
    print(f"  Cadence     : {info['frequency_minutes']} min  ({spd} samples/day)")
    print(f"  Exogenous   : {info['available_signal_columns']}")
    return data, info


# ---------------------------------------------------------------------------
# Stage 1 – GRU model
# ---------------------------------------------------------------------------

def weighted_mse_loss(horizon: int):
    """Weighted MSE loss that penalizes errors in later horizons more heavily."""
    weights = np.linspace(1.0, 2.5, num=horizon).astype(np.float32)
    weights = tf.constant(weights)
    def loss(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred) * weights, axis=-1)
    return loss


def build_gru_model(seq_len: int, horizon: int) -> Model:
    """
    Encoder-Decoder + Multi-Head Attention architecture for multi-step
    load forecasting (TPU-friendly).
    """
    inputs = Input(shape=(seq_len, 1))
    
    # Feature Projection
    x = Dense(64, activation="relu")(inputs)
    
    # Encoder: Bi-directional GRU
    encoder_outputs = Bidirectional(GRU(128, return_sequences=True, dropout=0.2))(x)
    
    # Encoder Self-Attention
    self_attn = MultiHeadAttention(num_heads=4, key_dim=64)(query=encoder_outputs, value=encoder_outputs)
    x2 = LayerNormalization()(Add()([encoder_outputs, self_attn]))
    
    # Projection to decoder steps (horizon) via parameter-efficient permutation
    x_proj = Dense(64, activation="relu")(x2)  # Project features from 256 -> 64
    x_transposed = Permute((2, 1))(x_proj)  # Transpose to (batch, 64, seq_len)
    time_projected = Dense(horizon, activation="relu")(x_transposed)  # Project time seq_len -> horizon
    decoder_inputs = Permute((2, 1))(time_projected)  # Transpose back to (batch, horizon, 64)
    
    # Decoder GRU
    decoder_outputs = GRU(128, return_sequences=True, dropout=0.2)(decoder_inputs)
    
    # Decoder Cross-Attention to encoder outputs
    cross_attn = MultiHeadAttention(num_heads=4, key_dim=64)(query=decoder_outputs, value=x2, key=x2)
    x3 = LayerNormalization()(Add()([decoder_outputs, cross_attn]))
    
    # Output projection
    out = Dense(1)(x3)
    out = Flatten()(out)
    
    # Gated Skip Connection (Raw inputs -> final output)
    skip = Dense(horizon)(Flatten()(inputs))
    gate = Dense(horizon, activation="sigmoid")(Flatten()(inputs))
    
    one_minus_gate = Lambda(lambda g: 1.0 - g)(gate)
    gated_out = Multiply()([out, gate])
    gated_skip = Multiply()([skip, one_minus_gate])
    final_output = Add()([gated_out, gated_skip])
    
    model = Model(inputs=inputs, outputs=final_output)
    model.compile(
        loss=weighted_mse_loss(horizon),
        optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
    )
    return model


def make_sequences(series: np.ndarray, seq_len: int, horizon: int):
    """
    Convert a 1-D load array into (X, y) pairs.
    X[i] = series[i : i+seq_len]                 (shape: seq_len, 1)
    y[i] = series[i+seq_len : i+seq_len+horizon] (shape: horizon)
    """
    X, y = [], []
    for i in range(len(series) - seq_len - horizon + 1):
        X.append(series[i : i + seq_len])
        y.append(series[i + seq_len : i + seq_len + horizon])
    return np.array(X)[..., np.newaxis], np.array(y)


def train_gru(
    load_train: np.ndarray,
    load_val: np.ndarray,
    seq_len: int,
    horizon: int,
    load_scaler: MinMaxScaler,
    epochs: int,
    augment: bool = True,
) -> tuple:
    """
    Scale, build sequences, train multi-step GRU+Attention.
    """
    train_scaled = load_scaler.transform(load_train.reshape(-1, 1)).flatten()
    val_scaled   = load_scaler.transform(load_val.reshape(-1, 1)).flatten()

    X_tr, y_tr = make_sequences(train_scaled, seq_len, horizon)
    X_vl, y_vl = make_sequences(val_scaled,   seq_len, horizon)

    if augment:
        print("\n  Applying training data augmentation (0.5% noise injection)...")
        noise = np.random.normal(0, 0.005, size=X_tr.shape).astype(np.float32)
        X_tr_aug = X_tr + noise
        X_tr = np.concatenate([X_tr, X_tr_aug], axis=0)
        y_tr = np.concatenate([y_tr, y_tr], axis=0)

    print(f"\n  GRU sequences  train={X_tr.shape}  val={X_vl.shape}")

    # Auto-detect TPU Strategy if available
    strategy = None
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        print("\n  TPU detected! Building model inside TPU Strategy scope.")
    except Exception:
        strategy = None

    if strategy:
        with strategy.scope():
            model = build_gru_model(seq_len, horizon)
    else:
        model = build_gru_model(seq_len, horizon)
    model.summary(print_fn=lambda s: print("    " + s))

    # Cosine Annealing with Warm Restarts scheduler
    T_max = 30
    def cosine_scheduler(epoch, lr):
        lr_max = 1e-3
        lr_min = 1e-5
        epoch_mod = epoch % T_max
        cos_val = math.cos(math.pi * epoch_mod / T_max)
        return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + cos_val)

    cbs = [
        EarlyStopping(monitor="val_loss", patience=20,
                      restore_best_weights=True, verbose=1),
        LearningRateScheduler(cosine_scheduler, verbose=1),
    ]

    print("\n  Training GRU ...")
    history = model.fit(
        X_tr, y_tr,
        epochs=epochs,
        batch_size=256,
        validation_data=(X_vl, y_vl),
        callbacks=cbs,
        verbose=1,
        shuffle=True if augment else False,  # Shuffle augmented batches
    )

    # Predictions in MW scale (use non-augmented version for predictions feature extraction)
    X_tr_orig, _ = make_sequences(train_scaled, seq_len, horizon)
    pred_tr_scaled = model.predict(X_tr_orig, verbose=0)
    pred_vl_scaled = model.predict(X_vl, verbose=0)

    pred_tr = load_scaler.inverse_transform(pred_tr_scaled.reshape(-1, 1)).reshape(pred_tr_scaled.shape)
    pred_vl = load_scaler.inverse_transform(pred_vl_scaled.reshape(-1, 1)).reshape(pred_vl_scaled.shape)

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    best_val_loss = float(min(history.history["val_loss"]))
    epochs_ran = len(history.history["loss"])

    print(f"\n  GRU best epoch  : {best_epoch}/{epochs_ran}")
    print(f"  GRU best val loss: {best_val_loss:.6f}")

    return model, pred_tr, pred_vl, best_epoch, best_val_loss, epochs_ran


# ---------------------------------------------------------------------------
# Stage 2 – Feature engineering for LightGBM residual model
# ---------------------------------------------------------------------------

class ResidualFeatureEngineer:
    """
    Builds a flattened feature matrix for the multi-step LightGBM residual corrector.
    Each row is a single prediction step (h) for a single forecast origin (t).
    To avoid data leakage, historical features (lags, rolling stats) are strictly
    anchored at time `t`.
    """

    def __init__(self, samples_per_day: int, horizon: int):
        self.spd = samples_per_day
        self.horizon = horizon
        half = max(2, samples_per_day // 2)
        week = samples_per_day * 7
        self.lag_periods = sorted({1, 2, 3, half, samples_per_day, week})
        self.rolling_windows = sorted({max(2, samples_per_day // 4),
                                        half, samples_per_day})

    def build(self, df: pd.DataFrame, gru_pred: np.ndarray,
              seq_len: int, origin_start_idx: int = None) -> tuple[pd.DataFrame, list[str]]:
        d = df.copy()

        # 1. Historical load features (computed on the whole series, anchored at shift=0 to 't')
        d["anchor_load"] = d["load"]
        for lag in self.lag_periods:
            d[f"anchor_lag_{lag}"] = d["load"].shift(lag)

        for w in self.rolling_windows:
            d[f"anchor_roll_mean_{w}"] = d["load"].shift(1).rolling(w).mean()
            d[f"anchor_roll_std_{w}"]  = d["load"].shift(1).rolling(w).std()
            d[f"anchor_roll_min_{w}"]  = d["load"].shift(1).rolling(w).min()
            d[f"anchor_roll_max_{w}"]  = d["load"].shift(1).rolling(w).max()

        d["anchor_diff_1"] = d["load"].diff(1)
        d[f"anchor_diff_{self.spd}"] = d["load"].diff(self.spd)

        historical_cols = [c for c in d.columns if c.startswith("anchor_")]

        # 2. Time features corresponding to target time t+h
        d["hour"]        = d.index.hour
        d["minute"]      = d.index.minute
        d["dayofweek"]   = d.index.dayofweek
        d["dayofmonth"]  = d.index.day
        d["month"]       = d.index.month
        d["dayofyear"]   = d.index.dayofyear
        d["is_weekend"]  = (d.index.dayofweek >= 5).astype(int)

        minute_frac = (d.index.hour * 60 + d.index.minute) / 1440.0
        d["hour_sin"]       = np.sin(2 * np.pi * minute_frac)
        d["hour_cos"]       = np.cos(2 * np.pi * minute_frac)
        d["day_sin"]        = np.sin(2 * np.pi * d["dayofweek"] / 7)
        d["day_cos"]        = np.cos(2 * np.pi * d["dayofweek"] / 7)
        d["month_sin"]      = np.sin(2 * np.pi * d["month"] / 12)
        d["month_cos"]      = np.cos(2 * np.pi * d["month"] / 12)
        d["dayofyear_sin"]  = np.sin(2 * np.pi * d["dayofyear"] / 366)
        d["dayofyear_cos"]  = np.cos(2 * np.pi * d["dayofyear"] / 366)

        time_cols = [
            "hour", "minute", "dayofweek", "dayofmonth", "month", "dayofyear",
            "is_weekend",
            "hour_sin", "hour_cos", "day_sin", "day_cos",
            "month_sin", "month_cos", "dayofyear_sin", "dayofyear_cos",
        ]

        exo_cols = []
        for col in REAL_SIGNAL_COLUMNS:
            if col not in d.columns:
                continue
            if col == "is_holiday":
                d[col] = (
                    d[col].astype(str).str.lower()
                    .map({"true":1,"false":0,"1":1,"0":0,"yes":1,"no":0})
                    .fillna(pd.to_numeric(d[col], errors="coerce"))
                    .fillna(0).astype(int)
                )
                exo_cols.append(col)
            else:
                d[col] = pd.to_numeric(d[col], errors="coerce")
                exo_cols.append(col)

        target_time_cols = time_cols + exo_cols

        # 3. Flatten predictions into records
        num_samples = len(gru_pred)
        # For the i-th sample, origin_t is origin_start_idx + i
        if origin_start_idx is None:
            origin_start_idx = seq_len - 1
        origins = np.arange(origin_start_idx, origin_start_idx + num_samples)

        records = []
        d_hist = d[historical_cols].values
        d_target = d[target_time_cols].values
        d_load = d["load"].values

        for i, origin_t in enumerate(origins):
            if pd.isna(d_hist[origin_t]).any():
                continue

            for h in range(1, self.horizon + 1):
                target_t = origin_t + h
                if target_t >= len(d):
                    break

                actual = d_load[target_t]
                if pd.isna(actual) or pd.isna(d_target[target_t]).any():
                    continue

                g_pred = gru_pred[i, h - 1]

                rec = {
                    "origin_t": origin_t,
                    "target_t": target_t,
                    "step_ahead": h,
                    "gru_pred": g_pred,
                    "actual": actual,
                    "log_step_ahead": np.log(h),
                    "step_ahead_sq": h ** 2,
                    "step_ahead_frac": h / self.horizon,
                    "load_same_hour_recent": d_load[target_t - 24 * int(np.ceil(h / 24.0))],
                    "load_same_hour_last_week": d_load[target_t - 168],
                }
                # populate fast
                for j, c in enumerate(historical_cols):
                    rec[c] = d_hist[origin_t, j]
                for j, c in enumerate(target_time_cols):
                    rec[c] = d_target[target_t, j]
                records.append(rec)

        feat_df = pd.DataFrame(records)
        feature_cols = [
            "step_ahead",
            "gru_pred",
            "log_step_ahead",
            "step_ahead_sq",
            "step_ahead_frac",
            "load_same_hour_recent",
            "load_same_hour_last_week",
        ] + historical_cols + target_time_cols

        return feat_df, feature_cols


# ---------------------------------------------------------------------------
# Stage 2 – LightGBM residual corrector
# ---------------------------------------------------------------------------

def train_lightgbm_residual(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_cols: list[str],
) -> lgb.Booster:
    """Train a regularised LightGBM on GRU residuals."""
    print("\n  Training LightGBM residual corrector ...")

    lgb_tr = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    lgb_vl = lgb.Dataset(X_val,   label=y_val,   feature_name=feature_cols,
                          reference=lgb_tr)

    params = {
        "objective":        "regression",
        "metric":           "rmse",
        "boosting_type":    "gbdt",
        "learning_rate":    0.01,
        "num_leaves":       63,          # larger leaves since dataset is flattened
        "max_depth":        8,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "lambda_l1":        0.1,
        "lambda_l2":        0.5,
        "min_gain_to_split": 0.01,
        "verbose":          -1,
        "seed":             42,
    }

    model = lgb.train(
        params,
        lgb_tr,
        num_boost_round=3000,
        valid_sets=[lgb_tr, lgb_vl],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(period=200),
        ],
    )
    print(f"  Best iteration: {model.best_iteration}")
    return model


# ---------------------------------------------------------------------------
# Time-series split helper
# ---------------------------------------------------------------------------

def split_time_series(
    n: int,
    train_ratio: float = 0.65,
    val_ratio:   float = 0.20,
    gap:         int   = 400,
):
    """Return (train_end, val_start, val_end, test_start) indices."""
    train_end  = int(n * train_ratio)
    val_end    = int(n * (train_ratio + val_ratio))

    train_end  = min(train_end,  n - 2 * gap - 1)
    val_start  = min(train_end  + gap, n - 1)
    val_end    = min(val_end,    n - gap)
    test_start = min(val_end    + gap, n)

    if min(train_end, val_end - val_start, n - test_start) <= gap:
        raise ValueError("Dataset too small for leakage-safe split.")

    return train_end, val_start, val_end, test_start


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    rmse = float(np.sqrt(mean_squared_error(y_true_f, y_pred_f)))
    mape = float(mean_absolute_percentage_error(y_true_f, y_pred_f) * 100)
    mae  = float(np.mean(np.abs(y_true_f - y_pred_f)))
    bias = float(np.mean(y_pred_f - y_true_f))
    print(f"  {label:<35} RMSE={rmse:7.2f} MW  MAPE={mape:5.2f}%  "
          f"MAE={mae:7.2f} MW  Bias={bias:+7.2f} MW")
    return {"rmse": rmse, "mape": mape, "mae": mae, "bias": bias}


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_gru_lgb_hybrid(
    data_path: str,
    model_dir: str = "models_gru_lgb_multistep",
    epochs:    int = 150,
    seq_len:   int | None = None,
    horizon:   int = 168,
):
    """
    Full training pipeline for global multi-step forecasting.
    """

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    data, info = load_data(data_path)
    spd = info["samples_per_day"]
    load_values = data["load"].values.astype(np.float32)
    n = len(load_values)

    if seq_len is None:
        seq_len = spd * 14   # default 14 days of lookback
    print(f"\n  GRU sequence length : {seq_len} steps")
    print(f"  Forecast horizon    : {horizon} steps")

    gap = max(seq_len, spd) + horizon
    train_end, val_start, val_end, test_start = split_time_series(
        n, train_ratio=0.65, val_ratio=0.20, gap=gap
    )

    print(f"\nData split (gap={gap}):")
    print(f"  Train      : 0 - {train_end}  ({train_end} rows)")
    print(f"  Val        : {val_start} - {val_end}  ({val_end - val_start} rows)")
    print(f"  Test       : {test_start} - {n}  ({n - test_start} rows)")

    load_train = load_values[:train_end]
    
    # Seed validation sequences with history (seq_len steps)
    val_start_seeded = max(0, val_start - seq_len)
    load_val_with_seed = load_values[val_start_seeded : val_end]
    
    # Seed test sequences with history (seq_len steps)
    test_start_seeded = max(0, test_start - seq_len)
    load_test_with_seed = load_values[test_start_seeded:]

    # ------------------------------------------------------------------
    # 2. Scale load for GRU
    # ------------------------------------------------------------------
    load_scaler = MinMaxScaler(feature_range=(0, 1))
    load_scaler.fit(load_train.reshape(-1, 1))

    # ------------------------------------------------------------------
    # 3. Train GRU  (Stage 1)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 1 — GRU (multi-step temporal pattern capture)")
    print("=" * 60)

    (
        gru_model,
        gru_pred_train,
        gru_pred_val,
        gru_best_epoch,
        gru_best_val_loss,
        gru_epochs_ran,
    ) = train_gru(load_train, load_val_with_seed, seq_len, horizon, load_scaler, epochs)

    # ------------------------------------------------------------------
    # 4. Build feature matrices for LightGBM  (Stage 2)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 2 — LightGBM (multi-step global residual correction)")
    print("=" * 60)

    engineer = ResidualFeatureEngineer(spd, horizon)

    # Warmup context for feature engineering: max lag + max rolling window
    warmup = max(engineer.lag_periods) + max(engineer.rolling_windows)

    # --- TRAIN split ---------------------------------------------------
    train_df = data.iloc[:train_end].copy()
    feat_train, feature_cols = engineer.build(train_df, gru_pred_train, seq_len, origin_start_idx=seq_len - 1)

    # --- VAL split -----------------------------------------------------
    val_start_seeded = max(0, val_start - warmup)
    val_df = data.iloc[val_start_seeded : val_end].copy()
    val_offset = val_start - val_start_seeded
    feat_val, _ = engineer.build(val_df, gru_pred_val, seq_len, origin_start_idx=val_offset - 1)

    shared_cols = [c for c in feature_cols if c in feat_val.columns]
    feature_cols = shared_cols

    print(f"\n  Feature count       : {len(feature_cols)}")
    print(f"  LightGBM train rows : {len(feat_train)}")
    print(f"  LightGBM val rows   : {len(feat_val)}")

    X_lgb_tr = feat_train[feature_cols].values
    y_lgb_tr = feat_train["actual"].values  # Stacking: predict actual load directly
    X_lgb_vl = feat_val[feature_cols].values
    y_lgb_vl = feat_val["actual"].values  # Stacking: predict actual load directly

    print(f"\n  Actual Load stats (train):")
    print(f"    mean  = {y_lgb_tr.mean():.2f} MW")
    print(f"    std   = {y_lgb_tr.std():.2f} MW")

    lgb_model = train_lightgbm_residual(
        X_lgb_tr, y_lgb_tr, X_lgb_vl, y_lgb_vl, feature_cols
    )

    # ------------------------------------------------------------------
    # 5. Evaluate on TEST split
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    test_start_seeded = max(0, test_start - seq_len)
    test_with_seed = load_values[test_start_seeded:]
    test_scaled = load_scaler.transform(test_with_seed.reshape(-1, 1)).flatten()
    X_te_gru, y_te_gru_scaled = make_sequences(test_scaled, seq_len, horizon)

    gru_pred_test_scaled = gru_model.predict(X_te_gru, verbose=0)
    gru_pred_test = load_scaler.inverse_transform(gru_pred_test_scaled.reshape(-1, 1)).reshape(gru_pred_test_scaled.shape)

    test_feat_start_seeded = max(0, test_start - warmup)
    test_df = data.iloc[test_feat_start_seeded:].copy()
    test_offset = test_start - test_feat_start_seeded
    feat_test, _ = engineer.build(test_df, gru_pred_test, seq_len, origin_start_idx=test_offset - 1)
    
    for c in feature_cols:
        if c not in feat_test.columns:
            feat_test[c] = 0.0

    X_lgb_te = feat_test[feature_cols].values
    lgb_actual_test = lgb_model.predict(
        X_lgb_te, num_iteration=lgb_model.best_iteration
    )

    hybrid_pred_flattened = lgb_actual_test  # Stacking: LightGBM output is the final prediction
    y_te_flattened = feat_test["actual"].values
    gru_pred_flattened = feat_test["gru_pred"].values

    print()
    gru_te_metrics    = calc_metrics(y_te_flattened, gru_pred_flattened, "GRU only - test (overall)")
    hybrid_te_metrics = calc_metrics(y_te_flattened, hybrid_pred_flattened, "GRU + LightGBM stacking - test (overall)")

    print("\nPer-Horizon Performance Evaluation:")
    for h in [1, 6, 12, 24, 48, 72, 120, 168]:
        mask = feat_test["step_ahead"] == h
        if mask.sum() > 0:
            actual_vals = feat_test.loc[mask, "actual"].values
            gru_vals = feat_test.loc[mask, "gru_pred"].values
            hybrid_vals = lgb_actual_test[mask.values]  # Stacking: LGB output is final pred
            calc_metrics(actual_vals, gru_vals, f"GRU only - step {h}")
            calc_metrics(actual_vals, hybrid_vals, f"Hybrid   - step {h}")
            print("-" * 50)

    rmse_imp = (gru_te_metrics["rmse"] - hybrid_te_metrics["rmse"]) / gru_te_metrics["rmse"] * 100
    mape_imp = (gru_te_metrics["mape"] - hybrid_te_metrics["mape"]) / gru_te_metrics["mape"] * 100
    print(f"\n  Improvement from Residual Corrector:")
    print(f"    RMSE : {rmse_imp:+.2f}%")
    print(f"    MAPE : {mape_imp:+.2f}%")

    fi = pd.DataFrame({
        "feature":    feature_cols,
        "importance": lgb_model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    print("\n  Top-15 LightGBM features (gain):")
    print(fi.head(15).to_string(index=False))

    # ------------------------------------------------------------------
    # 6. Save artefacts
    # ------------------------------------------------------------------
    os.makedirs(model_dir, exist_ok=True)

    gru_model.save(os.path.join(model_dir, "gru_stage1_model.keras"))
    lgb_model.save_model(os.path.join(model_dir, "lgb_stage2_model.txt"))

    metadata = {
        "architecture":            "GRU+Attention→LightGBM stacking ensemble",
        "ensemble_mode":           "stacking",
        "dataset_name":            info["dataset_name"],
        "frequency_minutes":       info["frequency_minutes"],
        "samples_per_day":         spd,
        "training_samples":        int(train_end),
        "validation_samples":      int(val_end - val_start),
        "test_samples":            int(n - test_start),
        "seq_len":                 int(seq_len),
        "horizon":                 int(horizon),
        "load_scaler":             load_scaler,
        "feature_cols":            feature_cols,
        "lgb_best_iteration":      int(lgb_model.best_iteration),
        "train_date": datetime.now().isoformat(),
    }

    joblib.dump(metadata, os.path.join(model_dir, "gru_lgb_metadata.joblib"))

    summary = {k: v for k, v in metadata.items()
               if not isinstance(v, (MinMaxScaler, list, np.ndarray))}
    summary["feature_cols"] = feature_cols
    with open(os.path.join(model_dir, "gru_lgb_training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Automatically deploy to backend/models/ if it exists
    backend_models_dir = "load_forecast_prediction/backend/models"
    if os.path.exists(backend_models_dir):
        print(f"  Deploying models to backend folder: {backend_models_dir} ...")
        gru_model.save(os.path.join(backend_models_dir, "gru_stage1_model.keras"))
        lgb_model.save_model(os.path.join(backend_models_dir, "lgb_stage2_model.txt"))
        joblib.dump(metadata, os.path.join(backend_models_dir, "gru_lgb_metadata.joblib"))
        with open(os.path.join(backend_models_dir, "gru_lgb_training_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Improvement:  RMSE {rmse_imp:+.2f}%  MAPE {mape_imp:+.2f}%")
    print("=" * 60)

    return gru_model, lgb_model, metadata


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train reversed GRU→LightGBM multi-step hybrid on real Karnataka load data"
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to real training CSV (default: auto-detect karnataka_realdata.csv)",
    )
    parser.add_argument(
        "--model-dir", type=str, default="models_gru_lgb_multistep",
        help="Directory to save model artefacts",
    )
    parser.add_argument(
        "--epochs", type=int, default=150,
        help="Maximum GRU training epochs (default: 150)",
    )
    parser.add_argument(
        "--seq-len", type=int, default=336,
        help="GRU input window in timesteps (default: 336 = 14 days for hourly data)",
    )
    parser.add_argument(
        "--horizon", type=int, default=168,
        help="Forecast horizon in timesteps (default: 168 = 7 days for hourly data)",
    )
    args = parser.parse_args()

    try:
        resolved = resolve_data_path(args.data)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    train_gru_lgb_hybrid(
        data_path=resolved,
        model_dir=args.model_dir,
        epochs=args.epochs,
        seq_len=args.seq_len,
        horizon=args.horizon,
    )
