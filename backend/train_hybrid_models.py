"""
Hybrid Model Training Script
============================
Train LightGBM + LSTM and LightGBM + GRU hybrid models for load forecasting.

Usage:
    python train_hybrid_models.py [--data path/to/data.csv] [--epochs 100]
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class FeatureEngineer:
    """Feature engineering for load forecasting."""
    
    LAG_PERIODS = [1, 2, 3, 6, 12, 24, 288]
    ROLLING_WINDOWS = [12, 24, 288]
    
    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create all features for the model."""
        df = df.copy()
        
        # Time features
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['dayofweek'] = df.index.dayofweek
        df['dayofmonth'] = df.index.day
        df['month'] = df.index.month
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['time_slot'] = df['hour'] * 12 + df['minute'] // 5
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Lag features
        for lag in FeatureEngineer.LAG_PERIODS:
            df[f'lag_{lag}'] = df['load'].shift(lag)
        
        # Rolling features
        for window in FeatureEngineer.ROLLING_WINDOWS:
            df[f'rolling_mean_{window}'] = df['load'].shift(1).rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df['load'].shift(1).rolling(window=window).std()
            df[f'rolling_min_{window}'] = df['load'].shift(1).rolling(window=window).min()
            df[f'rolling_max_{window}'] = df['load'].shift(1).rolling(window=window).max()
        
        # Difference features
        df['diff_1'] = df['load'].diff(1)
        df['diff_288'] = df['load'].diff(288)
        
        return df
    
    @staticmethod
    def get_feature_columns() -> list:
        """Get list of feature column names."""
        time_features = ['hour', 'minute', 'dayofweek', 'dayofmonth', 'month', 
                        'is_weekend', 'time_slot',
                        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
                        'month_sin', 'month_cos']
        
        lag_features = [f'lag_{lag}' for lag in FeatureEngineer.LAG_PERIODS]
        
        rolling_features = []
        for window in FeatureEngineer.ROLLING_WINDOWS:
            rolling_features.extend([
                f'rolling_mean_{window}', f'rolling_std_{window}',
                f'rolling_min_{window}', f'rolling_max_{window}'
            ])
        rolling_features.extend(['diff_1', 'diff_288'])
        
        return time_features + lag_features + rolling_features


def load_data(data_path: str) -> pd.DataFrame:
    """Load and preprocess data."""
    print(f"Loading data from {data_path}...")
    
    try:
        # Try loading with header
        data = pd.read_csv(data_path, header=None, names=['datetime', 'load'])
        data['datetime'] = pd.to_datetime(data['datetime'], format='%d/%m/%Y %H:%M')
    except:
        # Try alternative format
        data = pd.read_csv(data_path)
        if 'datetime' in data.columns:
            data['datetime'] = pd.to_datetime(data['datetime'])
        else:
            raise ValueError("Could not parse data file. Expected 'datetime' and 'load' columns.")
    
    data = data.set_index('datetime').sort_index()
    
    # Ensure complete days
    total_samples = len(data)
    num_days = total_samples // 288
    samples_to_use = num_days * 288
    data = data.iloc[:samples_to_use]
    
    print(f"  Loaded {len(data)} samples ({num_days} complete days)")
    print(f"  Date range: {data.index.min()} to {data.index.max()}")
    print(f"  Load range: {data['load'].min():.2f} - {data['load'].max():.2f} MW")
    
    return data


def train_lightgbm(X_train, y_train, X_val, y_val, feature_cols):
    """Train LightGBM model."""
    print("\nTraining LightGBM model...")
    
    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    lgb_val = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=lgb_train)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }
    
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    print(f"  Best iteration: {model.best_iteration}")
    return model


def build_lstm_model(nlags: int) -> Sequential:
    """Build LSTM model for residual prediction."""
    model = Sequential([
        LSTM(64, input_shape=(nlags, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(loss='mse', optimizer='adam')
    return model


def build_gru_model(nlags: int) -> Sequential:
    """Build GRU model for residual prediction."""
    model = Sequential([
        GRU(64, input_shape=(nlags, 1), return_sequences=True),
        Dropout(0.2),
        GRU(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(loss='mse', optimizer='adam')
    return model


def create_sequences(data: np.ndarray, nlags: int):
    """Create sequences for RNN training."""
    X, y = [], []
    for i in range(nlags, len(data)):
        X.append(data[i-nlags:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def train_hybrid_model(data_path: str, model_dir: str, epochs: int = 100):
    """Train both hybrid models (LightGBM+LSTM and LightGBM+GRU)."""
    
    # Load and prepare data
    data = load_data(data_path)
    data_featured = FeatureEngineer.create_features(data)
    data_featured = data_featured.dropna()
    
    feature_cols = FeatureEngineer.get_feature_columns()
    X = data_featured[feature_cols].values
    y = data_featured['load'].values
    
    # Time-based split: 70% train, 15% validation, 15% test
    train_size = int(len(X) * 0.70)
    val_size = int(len(X) * 0.15)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    # Train LightGBM
    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val, feature_cols)
    
    # Get LightGBM predictions and residuals
    lgb_pred_train = lgb_model.predict(X_train, num_iteration=lgb_model.best_iteration)
    lgb_pred_val = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    lgb_pred_test = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    
    residuals_train = y_train - lgb_pred_train
    residuals_val = y_val - lgb_pred_val
    residuals_test = y_test - lgb_pred_test
    
    # Scale residuals
    residual_scaler = MinMaxScaler(feature_range=(-1, 1))
    residuals_train_scaled = residual_scaler.fit_transform(residuals_train.reshape(-1, 1))
    residuals_val_scaled = residual_scaler.transform(residuals_val.reshape(-1, 1))
    residuals_test_scaled = residual_scaler.transform(residuals_test.reshape(-1, 1))
    
    # Create sequences for RNN
    nlags = 288  # Full 24-hour pattern (critical for daily seasonality)
    
    lstm_X_train, lstm_y_train = create_sequences(residuals_train_scaled, nlags)
    lstm_X_val, lstm_y_val = create_sequences(residuals_val_scaled, nlags)
    lstm_X_test, lstm_y_test = create_sequences(residuals_test_scaled, nlags)
    
    lstm_X_train = lstm_X_train.reshape(-1, nlags, 1)
    lstm_X_val = lstm_X_val.reshape(-1, nlags, 1)
    lstm_X_test = lstm_X_test.reshape(-1, nlags, 1)
    
    print(f"\nRNN sequence shapes:")
    print(f"  Train: {lstm_X_train.shape}, Val: {lstm_X_val.shape}, Test: {lstm_X_test.shape}")
    
    # Early stopping callback (allow more patience)
    es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    
    # Train LSTM model
    print("\nTraining LSTM model on residuals...")
    lstm_model = build_lstm_model(nlags)
    lstm_model.fit(
        lstm_X_train, lstm_y_train,
        epochs=epochs,
        batch_size=64,
        validation_data=(lstm_X_val, lstm_y_val),
        callbacks=[es],
        verbose=1
    )
    
    # Train GRU model
    print("\nTraining GRU model on residuals...")
    gru_model = build_gru_model(nlags)
    gru_model.fit(
        lstm_X_train, lstm_y_train,  # Use same data
        epochs=epochs,
        batch_size=64,
        validation_data=(lstm_X_val, lstm_y_val),
        callbacks=[es],
        verbose=1
    )
    
    # Evaluate hybrid models
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # LSTM Hybrid predictions
    lstm_pred = lstm_model.predict(lstm_X_test, verbose=0)
    lstm_pred_inv = residual_scaler.inverse_transform(lstm_pred).flatten()
    lgb_pred_aligned = lgb_pred_test[nlags:]
    y_test_aligned = y_test[nlags:]
    lstm_hybrid_pred = lgb_pred_aligned + lstm_pred_inv
    
    # GRU Hybrid predictions
    gru_pred = gru_model.predict(lstm_X_test, verbose=0)
    gru_pred_inv = residual_scaler.inverse_transform(gru_pred).flatten()
    gru_hybrid_pred = lgb_pred_aligned + gru_pred_inv
    
    # Calculate metrics
    def calc_metrics(y_true, y_pred, name):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        print(f"  {name}: RMSE={rmse:.2f} MW, MAPE={mape:.2f}%")
        return rmse, mape
    
    print("\nTest Set Results:")
    lgb_metrics = calc_metrics(y_test_aligned, lgb_pred_aligned, "LightGBM Only")
    lstm_metrics = calc_metrics(y_test_aligned, lstm_hybrid_pred, "LightGBM + LSTM")
    gru_metrics = calc_metrics(y_test_aligned, gru_hybrid_pred, "LightGBM + GRU")
    
    # Calculate improvements
    lstm_improvement = ((lgb_metrics[0] - lstm_metrics[0]) / lgb_metrics[0]) * 100
    gru_improvement = ((lgb_metrics[0] - gru_metrics[0]) / lgb_metrics[0]) * 100
    
    print(f"\nImprovement over LightGBM:")
    print(f"  LSTM Hybrid: {lstm_improvement:.2f}%")
    print(f"  GRU Hybrid: {gru_improvement:.2f}%")
    
    # Save models
    print("\n" + "=" * 60)
    print("SAVING MODELS")
    print("=" * 60)
    
    os.makedirs(model_dir, exist_ok=True)
    
    # Save LightGBM
    lgb_model.save_model(os.path.join(model_dir, 'lgb_model.txt'))
    print(f"  ✓ LightGBM saved")
    
    # Save LSTM
    lstm_model.save(os.path.join(model_dir, 'lstm_model.keras'))
    print(f"  ✓ LSTM model saved")
    
    # Save GRU
    gru_model.save(os.path.join(model_dir, 'gru_model.keras'))
    print(f"  ✓ GRU model saved")
    
    # Save metadata
    metadata = {
        'feature_cols': feature_cols,
        'residual_scaler': residual_scaler,
        'nlags': nlags,
        'train_date': datetime.now().isoformat(),
        'metrics': {
            'lgb_rmse': lgb_metrics[0],
            'lstm_rmse': lstm_metrics[0],
            'gru_rmse': gru_metrics[0]
        }
    }
    
    joblib.dump(metadata, os.path.join(model_dir, 'lstm_metadata.joblib'))
    joblib.dump(metadata, os.path.join(model_dir, 'gru_metadata.joblib'))
    joblib.dump(metadata, os.path.join(model_dir, 'metadata.joblib'))
    print(f"  ✓ Metadata saved")
    
    # Verify models were saved correctly
    print("\n" + "=" * 60)
    print("VERIFYING SAVED MODELS")
    print("=" * 60)
    
    try:
        from tensorflow.keras.models import load_model
        test_lstm = load_model(os.path.join(model_dir, 'lstm_model.keras'))
        print("  ✓ LSTM model verified - loads correctly")
        del test_lstm
    except Exception as e:
        print(f"  ✗ LSTM model verification FAILED: {e}")
    
    try:
        test_gru = load_model(os.path.join(model_dir, 'gru_model.keras'))
        print("  ✓ GRU model verified - loads correctly")
        del test_gru
    except Exception as e:
        print(f"  ✗ GRU model verification FAILED: {e}")
    
    try:
        test_metadata = joblib.load(os.path.join(model_dir, 'metadata.joblib'))
        print(f"  ✓ Metadata verified - {len(test_metadata['feature_cols'])} features")
    except Exception as e:
        print(f"  ✗ Metadata verification FAILED: {e}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    return lgb_model, lstm_model, gru_model, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train hybrid load forecasting models')
    parser.add_argument('--data', type=str, default='data/monthdata1.csv',
                       help='Path to training data CSV')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs for neural networks')
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data):
        # Try alternative paths
        alt_paths = ['monthdata1.csv', '../monthdata1.csv', 'data/delhi.csv']
        for alt in alt_paths:
            if os.path.exists(alt):
                args.data = alt
                break
        else:
            print(f"Error: Data file not found: {args.data}")
            print("Please provide a valid data file path using --data argument")
            sys.exit(1)
    
    train_hybrid_model(args.data, args.model_dir, args.epochs)
