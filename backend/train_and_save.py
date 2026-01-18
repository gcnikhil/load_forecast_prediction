import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import json

# Configuration
DATA_PATH = 'backend/data/monthdata1.csv'
MODEL_DIR = 'backend/models'
os.makedirs(MODEL_DIR, exist_ok=True)

def create_features(df):
    """
    Create time-based and lag features for LightGBM
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['dayofweek'] = df.index.dayofweek
    df['dayofmonth'] = df.index.day
    df['month'] = df.index.month
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    # Time slot of the day (0-287)
    df['time_slot'] = df['hour'] * 12 + df['minute'] // 5
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    return df

def create_lag_features(df, lags=[1, 2, 3, 6, 12, 24, 288]):
    """
    Create lag features
    """
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df['load'].shift(lag)
    
    # Rolling statistics
    df['rolling_mean_12'] = df['load'].shift(1).rolling(window=12).mean() 
    df['rolling_std_12'] = df['load'].shift(1).rolling(window=12).std()
    df['rolling_mean_288'] = df['load'].shift(1).rolling(window=288).mean()
    
    # Difference features
    df['diff_1'] = df['load'].diff(1)
    df['diff_288'] = df['load'].diff(288)
    
    return df

def train_models():
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found at {DATA_PATH}. Using dummy data for demonstration.")
        # Build dummy data
        dates = pd.date_range(start='2025-01-17', periods=5000, freq='5min')
        # Create a synthetic load curve
        t = np.linspace(0, 100, 5000)
        load = 4000 + 1000 * np.sin(t) + 500 * np.random.normal(0, 0.5, 5000)
        data = pd.DataFrame({'load': load}, index=dates)
        data.index.name = 'datetime'
    else:
        print("Loading data...")
        try:
            data = pd.read_csv(DATA_PATH, header=None, names=['datetime', 'load'])
            data['datetime'] = pd.to_datetime(data['datetime'], format='%d/%m/%Y %H:%M')
            data = data.set_index('datetime')
        except Exception as e:
            print(f"Error reading csv: {e}. Using dummy data.")
            dates = pd.date_range(start='2025-01-17', periods=5000, freq='5min')
            data = pd.DataFrame({'load': np.random.rand(5000)*1000 + 4000}, index=dates)

    # Feature Engineering
    print("Feature engineering...")
    data_featured = create_features(data)
    data_featured = create_lag_features(data_featured)
    data_featured = data_featured.dropna()
    
    feature_cols = [col for col in data_featured.columns if col != 'load']
    X = data_featured[feature_cols].values
    y = data_featured['load'].values
    
    # Split
    train_size = int(len(X) * 0.8)
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:]
    y_val = y[train_size:]
    
    # Train LightGBM
    print("Training LightGBM...")
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1
    }
    
    lgb_model = lgb.train(
        params,
        lgb_train,
        num_boost_round=100,
        valid_sets=[lgb_train, lgb_val]
    )
    
    # Save LightGBM
    lgb_model.save_model(os.path.join(MODEL_DIR, 'lgb_model.txt'))
    print("LightGBM model saved.")
    
    # Save metadata (features list)
    metadata = {
        'feature_cols': feature_cols,
        'nlags': 288
    }
    joblib.dump(metadata, os.path.join(MODEL_DIR, 'metadata.joblib'))
    
    # Save dummy scaler just to keep API compatible if needed, or remove it.
    # We won't use LSTM so scaler not strictly needed for residuals.
    # But let's save an empty one or just skip it.
    
    print("All models and artifacts saved successfully.")

if __name__ == "__main__":
    train_models()

