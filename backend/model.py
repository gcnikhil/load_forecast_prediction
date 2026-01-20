"""
Hybrid Load Forecasting Model Service
=====================================
LightGBM + LSTM and LightGBM + GRU hybrid models for Delhi SLDC load forecasting.

Architecture:
- LightGBM: Captures calendar features, lags, and tabular patterns
- LSTM/GRU: Models residual temporal patterns that LightGBM misses
- Final prediction = LightGBM prediction + Neural Network residual correction
"""

import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports with error handling
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available.")


class FeatureEngineer:
    """Feature engineering for load forecasting."""
    
    LAG_PERIODS = [1, 2, 3, 6, 12, 24, 288]
    ROLLING_WINDOWS = [12, 24, 288]
    
    @staticmethod
    def get_feature_columns() -> List[str]:
        """Get list of feature columns - MATCHES train_hybrid_models.py (34 features)."""
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


class HybridModel:
    """Hybrid LightGBM + Neural Network model."""
    
    def __init__(self, model_name: str, nn_type: str):
        self.model_name = model_name
        self.nn_type = nn_type
        self.lgb_model = None
        self.nn_model = None
        self.residual_scaler = None
        self.feature_cols = None
        self.metadata = None
        self.nlags = 288  # Default to 288 (24 hours at 5-min intervals) to match training
        
    def load(self, model_dir: str) -> bool:
        """Load model artifacts."""
        try:
            # Load LightGBM model
            lgb_path = os.path.join(model_dir, f'lgb_{self.nn_type}_model.txt')
            if not os.path.exists(lgb_path):
                lgb_path = os.path.join(model_dir, 'lgb_model.txt')
            
            if os.path.exists(lgb_path):
                self.lgb_model = lgb.Booster(model_file=lgb_path)
            
            # Load Neural Network model
            if TF_AVAILABLE:
                for ext in ['.keras', '.h5']:
                    nn_path = os.path.join(model_dir, f'{self.nn_type}_model{ext}')
                    if os.path.exists(nn_path):
                        self.nn_model = load_model(nn_path)
                        break
            
            # Load metadata
            for meta_name in [f'{self.nn_type}_metadata.joblib', 'metadata.joblib']:
                meta_path = os.path.join(model_dir, meta_name)
                if os.path.exists(meta_path):
                    self.metadata = joblib.load(meta_path)
                    self.residual_scaler = self.metadata.get('residual_scaler')
                    self.feature_cols = self.metadata.get('feature_cols')
                    self.nlags = self.metadata.get('nlags', 288)  # Default to 288 (24 hours)
                    break
            
            print(f"  ✓ {self.model_name} loaded (nlags={self.nlags}, NN={'yes' if self.nn_model else 'no'})")
            return self.lgb_model is not None
            
        except Exception as e:
            print(f"  ✗ Error loading {self.model_name}: {e}")
            return False
    
    def predict_single(self, features: np.ndarray, residual_history: np.ndarray) -> float:
        """Make a single prediction."""
        lgb_pred = self.lgb_model.predict(features.reshape(1, -1))[0]
        
        nn_correction = 0.0
        if self.nn_model is not None and self.residual_scaler is not None:
            try:
                nn_input = residual_history[-self.nlags:].reshape(1, self.nlags, 1)
                nn_input_scaled = self.residual_scaler.transform(
                    nn_input.reshape(-1, 1)
                ).reshape(1, self.nlags, 1)
                nn_pred_scaled = self.nn_model.predict(nn_input_scaled, verbose=0)
                nn_correction = self.residual_scaler.inverse_transform(nn_pred_scaled)[0, 0]
            except:
                nn_correction = 0.0
        
        return max(0, lgb_pred + nn_correction)


class ModelService:
    """Main service for load forecasting predictions."""
    
    def __init__(self):
        self.lstm_hybrid = HybridModel("LightGBM + LSTM", "lstm")
        self.gru_hybrid = HybridModel("LightGBM + GRU", "gru")
        self.base_data = None
        self.model_dir = "models"
        self.data_dir = "data"
        self._is_loaded = False
        
    def load_models(self):
        """Load all models and historical data."""
        print("=" * 50)
        print("Loading Hybrid Forecasting Models...")
        print("=" * 50)
        
        lstm_loaded = self.lstm_hybrid.load(self.model_dir)
        gru_loaded = self.gru_hybrid.load(self.model_dir)
        self._load_historical_data()
        self._is_loaded = lstm_loaded or gru_loaded
        
        print("=" * 50)
        print(f"  LightGBM + LSTM: {'Ready' if lstm_loaded else 'Fallback mode'}")
        print(f"  LightGBM + GRU: {'Ready' if gru_loaded else 'Fallback mode'}")
        print("=" * 50)
        
    def _load_historical_data(self):
        """Load historical data for lag features."""
        data_files = [
            os.path.join(self.data_dir, 'monthdata1.csv'),
            os.path.join(self.data_dir, 'delhi.csv'),
            'monthdata1.csv',
        ]
        
        for data_path in data_files:
            if os.path.exists(data_path):
                try:
                    self.base_data = pd.read_csv(data_path, header=None, names=['datetime', 'load'])
                    self.base_data['datetime'] = pd.to_datetime(
                        self.base_data['datetime'], format='%d/%m/%Y %H:%M'
                    )
                    self.base_data = self.base_data.set_index('datetime').sort_index()
                    print(f"  ✓ Historical data: {len(self.base_data)} records")
                    return
                except:
                    continue
        
        self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Generate realistic synthetic load patterns."""
        print("  ⚠ Using synthetic data patterns")
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=30*288, freq='5min')
        hours = dates.hour + dates.minute / 60
        
        base = 4000
        daily = 500 * np.sin(np.pi * (hours - 6) / 12) + 300 * np.sin(np.pi * (hours - 18) / 6)
        weekend = np.where(dates.dayofweek >= 5, -200, 0)
        noise = np.random.normal(0, 100, len(dates))
        
        load = np.clip(base + daily + weekend + noise, 2000, 7000)
        self.base_data = pd.DataFrame({'load': load}, index=dates)
    
    def is_loaded(self, model_type: str = "all") -> bool:
        """Check if models are loaded."""
        if model_type == "lstm":
            return self.lstm_hybrid.lgb_model is not None
        elif model_type == "gru":
            return self.gru_hybrid.lgb_model is not None
        return self._is_loaded
    
    def _predict_with_nn(self, hybrid_model: HybridModel, features: np.ndarray, residual_history: np.ndarray) -> float:
        """Optimized NN prediction - avoids redundant LightGBM call."""
        lgb_pred = hybrid_model.lgb_model.predict(features.reshape(1, -1))[0]
        
        nn_correction = 0.0
        try:
            nlags = hybrid_model.nlags
            nn_input = residual_history[-nlags:].reshape(1, nlags, 1)
            nn_input_scaled = hybrid_model.residual_scaler.transform(
                nn_input.reshape(-1, 1)
            ).reshape(1, nlags, 1)
            # Use call() instead of predict() for faster inference
            nn_pred_scaled = hybrid_model.nn_model(nn_input_scaled, training=False).numpy()
            nn_correction = hybrid_model.residual_scaler.inverse_transform(nn_pred_scaled)[0, 0]
        except Exception as e:
            pass  # Fall back to LightGBM only
        
        return max(0, lgb_pred + nn_correction)
    
    def _get_nn_correction(self, hybrid_model: HybridModel, residual_history: np.ndarray) -> float:
        """Get just the NN correction value (for periodic application)."""
        try:
            nlags = hybrid_model.nlags
            nn_input = residual_history[-nlags:].reshape(1, nlags, 1)
            nn_input_scaled = hybrid_model.residual_scaler.transform(
                nn_input.reshape(-1, 1)
            ).reshape(1, nlags, 1)
            nn_pred_scaled = hybrid_model.nn_model(nn_input_scaled, training=False).numpy()
            return hybrid_model.residual_scaler.inverse_transform(nn_pred_scaled)[0, 0]
        except:
            return 0.0

    def _get_daily_pattern(self, target_date: datetime) -> Optional[pd.Series]:
        """Get historical pattern for same day of week."""
        if self.base_data is None:
            return None
        target_dow = target_date.weekday()
        similar = self.base_data[self.base_data.index.dayofweek == target_dow]
        if len(similar) >= 288:
            return similar.groupby(similar.index.time)['load'].mean()
        return self.base_data.groupby(self.base_data.index.time)['load'].mean()
    
    def _get_historical_reference(self, timestamps: list) -> np.ndarray:
        """Get historical load values for similar times as reference for lag features."""
        if self.base_data is None:
            return None
        
        # Build reference using historical average for each time slot
        time_avg = self.base_data.groupby(self.base_data.index.time)['load'].mean()
        
        reference = []
        for ts in timestamps:
            t = ts.time()
            if t in time_avg.index:
                reference.append(time_avg[t])
            else:
                reference.append(self.base_data['load'].mean())
        return np.array(reference)
    
    def predict(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate predictions for a date range - OPTIMIZED HYBRID MODEL VERSION."""
        import time
        start_time = time.time()
        
        # Generate timestamps
        timestamps = []
        current = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_ts = end_date.replace(hour=23, minute=55, second=0, microsecond=0)
        
        while current <= end_ts:
            timestamps.append(current)
            current += timedelta(minutes=5)
        
        if not timestamps:
            timestamps = [start_date]
        
        n_preds = len(timestamps)
        print(f"[PREDICT] Generating {n_preds} predictions...")
        
        # Quick checks
        if self.lstm_hybrid.lgb_model is None:
            print("[PREDICT] WARNING: Models not loaded, using fallback")
            return self._fast_fallback_prediction(timestamps)
        
        mean_load = self.base_data['load'].mean() if self.base_data is not None else 3700
        
        # Get historical reference pattern for realistic lag features
        hist_reference = self._get_historical_reference(timestamps)
        
        # Build extended reference including historical tail + forecast period reference
        # This provides realistic lag features without autoregressive drift
        historical_tail = list(self.base_data['load'].values[-576:]) if self.base_data is not None else [mean_load] * 576
        extended_reference = historical_tail + list(hist_reference)
        
        feature_cols = FeatureEngineer.get_feature_columns()
        
        # BATCH FEATURE CREATION - Much faster than loop
        all_features = []
        for i, ts in enumerate(timestamps):
            # Use position in extended_reference for lag features
            # This gives us historical context + realistic future pattern
            ref_position = 576 + i  # Position in extended_reference
            features = self._create_features_fast(ts, extended_reference, ref_position, mean_load)
            X = np.array([features.get(col, 0) for col in feature_cols])
            all_features.append(X)
        
        X_batch = np.array(all_features)
        print(f"[DEBUG] Feature batch shape: {X_batch.shape}")
        
        # BATCH LIGHTGBM PREDICTION - Very fast
        base_preds = self.lstm_hybrid.lgb_model.predict(X_batch)
        print(f"[DEBUG] LightGBM batch prediction done: min={base_preds.min():.0f}, max={base_preds.max():.0f}, std={base_preds.std():.0f}")
        
        # Apply NN corrections (sample-based for speed)
        lstm_correction = 0.0
        gru_correction = 0.0
        
        use_lstm_nn = self.lstm_hybrid.nn_model is not None and self.lstm_hybrid.residual_scaler is not None
        use_gru_nn = self.gru_hybrid.nn_model is not None and self.gru_hybrid.residual_scaler is not None
        
        if use_lstm_nn or use_gru_nn:
            # Get NN corrections from historical residual pattern (sample once)
            nlags = self.lstm_hybrid.nlags if self.lstm_hybrid.nlags else 288
            residual_history = np.array(historical_tail[-nlags:])
            
            if use_lstm_nn:
                lstm_correction = self._get_nn_correction(self.lstm_hybrid, residual_history)
            if use_gru_nn:
                gru_correction = self._get_nn_correction(self.gru_hybrid, residual_history)
            
            print(f"[DEBUG] NN corrections: LSTM={lstm_correction:.2f}, GRU={gru_correction:.2f}")
        
        # Apply corrections
        lstm_preds = np.clip(base_preds + lstm_correction, 1800, 7000)
        gru_preds = np.clip(base_preds + gru_correction, 1800, 7000)
        
        elapsed = time.time() - start_time
        print(f"[DEBUG] Final predictions - LSTM: min={lstm_preds.min():.0f}, max={lstm_preds.max():.0f}, std={lstm_preds.std():.0f}")
        print(f"[PREDICT] Completed in {elapsed:.2f} seconds")
        use_lstm_nn = self.lstm_hybrid.nn_model is not None and self.lstm_hybrid.residual_scaler is not None
        use_gru_nn = self.gru_hybrid.nn_model is not None and self.gru_hybrid.residual_scaler is not None
        
        return pd.DataFrame({
            'loads_lightgbm_lstm': lstm_preds,
            'loads_lightgbm_gru': gru_preds
        }, index=timestamps)
    
    def _create_features_fast(self, ts: datetime, extended_reference: list, 
                               ref_position: int, mean_load: float) -> Dict[str, float]:
        """Create feature dictionary using extended reference for lag features (no autoregressive drift)."""
        f = {}
        
        # Time features
        f['hour'] = ts.hour
        f['minute'] = ts.minute
        f['dayofweek'] = ts.weekday()
        f['dayofmonth'] = ts.day
        f['month'] = ts.month
        f['is_weekend'] = int(ts.weekday() >= 5)
        f['time_slot'] = ts.hour * 12 + ts.minute // 5
        
        # Cyclical features
        f['hour_sin'] = np.sin(2 * np.pi * ts.hour / 24)
        f['hour_cos'] = np.cos(2 * np.pi * ts.hour / 24)
        f['day_sin'] = np.sin(2 * np.pi * ts.weekday() / 7)
        f['day_cos'] = np.cos(2 * np.pi * ts.weekday() / 7)
        f['month_sin'] = np.sin(2 * np.pi * ts.month / 12)
        f['month_cos'] = np.cos(2 * np.pi * ts.month / 12)
        
        # Lag features from extended reference
        for lag in FeatureEngineer.LAG_PERIODS:
            idx = ref_position - lag
            if 0 <= idx < len(extended_reference):
                f[f'lag_{lag}'] = extended_reference[idx]
            else:
                f[f'lag_{lag}'] = mean_load
        
        # Rolling features
        for window in FeatureEngineer.ROLLING_WINDOWS:
            start_idx = max(0, ref_position - window)
            end_idx = ref_position
            if end_idx > start_idx and end_idx <= len(extended_reference):
                w = extended_reference[start_idx:end_idx]
                f[f'rolling_mean_{window}'] = np.mean(w)
                f[f'rolling_std_{window}'] = np.std(w) if len(w) > 1 else 0
                f[f'rolling_min_{window}'] = np.min(w)
                f[f'rolling_max_{window}'] = np.max(w)
            else:
                f[f'rolling_mean_{window}'] = mean_load
                f[f'rolling_std_{window}'] = 0
                f[f'rolling_min_{window}'] = mean_load * 0.9
                f[f'rolling_max_{window}'] = mean_load * 1.1
        
        # Difference features
        if ref_position >= 1 and ref_position < len(extended_reference):
            f['diff_1'] = extended_reference[ref_position] - extended_reference[ref_position - 1] if ref_position > 0 else 0
        else:
            f['diff_1'] = 0
            
        if ref_position >= 288:
            f['diff_288'] = extended_reference[ref_position] - extended_reference[ref_position - 288]
        else:
            f['diff_288'] = 0
        
        return f
    
    def _fast_fallback_prediction(self, timestamps: list) -> pd.DataFrame:
        """Fast fallback when models not loaded."""
        preds = []
        mean_load = 3700
        for ts in timestamps:
            hour = ts.hour + ts.minute / 60
            base = mean_load + 500 * np.sin(np.pi * (hour - 6) / 12)
            preds.append(np.clip(base + np.random.normal(0, 50), 1800, 6000))
        
        return pd.DataFrame({
            'loads_lightgbm_lstm': preds,
            'loads_lightgbm_gru': preds
        }, index=timestamps)
    
    def _create_features(self, ts: datetime, recent_loads: List[float], 
                         daily_pattern: Optional[pd.Series], mean_load: float) -> Dict[str, float]:
        """Create feature dictionary for a timestamp."""
        f = {}
        
        # Time features
        f['hour'] = ts.hour
        f['minute'] = ts.minute
        f['dayofweek'] = ts.weekday()
        f['dayofmonth'] = ts.day
        f['month'] = ts.month
        f['is_weekend'] = int(ts.weekday() >= 5)
        f['time_slot'] = ts.hour * 12 + ts.minute // 5
        
        # Cyclical features
        f['hour_sin'] = np.sin(2 * np.pi * ts.hour / 24)
        f['hour_cos'] = np.cos(2 * np.pi * ts.hour / 24)
        f['day_sin'] = np.sin(2 * np.pi * ts.weekday() / 7)
        f['day_cos'] = np.cos(2 * np.pi * ts.weekday() / 7)
        f['month_sin'] = np.sin(2 * np.pi * ts.month / 12)
        f['month_cos'] = np.cos(2 * np.pi * ts.month / 12)
        
        # Lag features
        for lag in FeatureEngineer.LAG_PERIODS:
            if len(recent_loads) >= lag:
                f[f'lag_{lag}'] = recent_loads[-lag]
            elif daily_pattern is not None:
                f[f'lag_{lag}'] = daily_pattern.get(ts.time(), mean_load)
            else:
                f[f'lag_{lag}'] = mean_load
        
        # Rolling features - ALL statistics for windows [12, 24, 288]
        for window in FeatureEngineer.ROLLING_WINDOWS:
            if len(recent_loads) >= window:
                w = recent_loads[-window:]
                f[f'rolling_mean_{window}'] = np.mean(w)
                f[f'rolling_std_{window}'] = np.std(w) if len(w) > 1 else 0
                f[f'rolling_min_{window}'] = np.min(w)
                f[f'rolling_max_{window}'] = np.max(w)
            else:
                f[f'rolling_mean_{window}'] = mean_load
                f[f'rolling_std_{window}'] = 0
                f[f'rolling_min_{window}'] = mean_load * 0.9
                f[f'rolling_max_{window}'] = mean_load * 1.1
        
        f['diff_1'] = recent_loads[-1] - recent_loads[-2] if len(recent_loads) >= 2 else 0
        f['diff_288'] = recent_loads[-1] - recent_loads[-288] if len(recent_loads) >= 288 else 0
        
        return f
    
    def _fallback_prediction(self, ts: datetime, daily_pattern: Optional[pd.Series], mean_load: float) -> float:
        """Fallback prediction using historical patterns."""
        if daily_pattern is not None:
            base = daily_pattern.get(ts.time(), mean_load)
        else:
            hour = ts.hour + ts.minute / 60
            base = mean_load + 500 * np.sin(np.pi * (hour - 6) / 12)
        return max(0, base + np.random.normal(0, 50))

