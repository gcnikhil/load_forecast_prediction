"""
Hybrid Load Forecasting Model Service
=====================================
LightGBM + LSTM and LightGBM + GRU hybrid models for Bengaluru/BESCOM load forecasting.
"""

import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BENGALURU_BASE_DEMAND_MW = 3800
BENGALURU_PEAK_DEMAND_MW = 6200
BENGALURU_MIN_DEMAND_MW = 2400
BENGALURU_MAX_DEMAND_MW = 7800

WEATHER_COLUMNS = [
    "temperature_celsius",
    "humidity_percent",
    "precipitation_mm",
    "weather_code",
]

try:
    from tensorflow.keras.models import load_model

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available.")

try:
    from data_scraper import BengaluruDataScraper

    SCRAPER_AVAILABLE = True
except ImportError:
    SCRAPER_AVAILABLE = False

try:
    import holidays

    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False


class FeatureEngineer:
    """Feature engineering metadata mirror used by inference."""

    LAG_PERIODS = [1, 2, 3, 6, 12, 24, 288]
    ROLLING_WINDOWS = [12, 24, 288]


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
        self.nlags = 24

    def load(self, model_dir: str) -> bool:
        """Load model artifacts."""
        try:
            lgb_path = os.path.join(model_dir, f"lgb_{self.nn_type}_model.txt")
            if not os.path.exists(lgb_path):
                lgb_path = os.path.join(model_dir, "lgb_model.txt")

            if os.path.exists(lgb_path):
                self.lgb_model = lgb.Booster(model_file=lgb_path)

            if TF_AVAILABLE:
                for ext in [".keras", ".h5"]:
                    nn_path = os.path.join(model_dir, f"{self.nn_type}_model{ext}")
                    if os.path.exists(nn_path):
                        self.nn_model = load_model(nn_path)
                        break

            for meta_name in [f"{self.nn_type}_metadata.joblib", "metadata.joblib"]:
                meta_path = os.path.join(model_dir, meta_name)
                if os.path.exists(meta_path):
                    self.metadata = joblib.load(meta_path)
                    self.residual_scaler = self.metadata.get("residual_scaler")
                    self.feature_cols = self.metadata.get("feature_cols")
                    self.nlags = self.metadata.get("nlags", 24)
                    break

            print(f"  Loaded {self.model_name}")
            return self.lgb_model is not None
        except Exception as e:
            print(f"  Error loading {self.model_name}: {e}")
            return False


class ModelService:
    """Main service for load forecasting predictions."""

    def __init__(self):
        self.lstm_hybrid = HybridModel("LightGBM + LSTM", "lstm")
        self.gru_hybrid = HybridModel("LightGBM + GRU", "gru")
        self.base_data = None
        self.model_dir = "models"
        self.data_dir = "data"
        self._is_loaded = False
        self.weather_profile = None
        self.forecast_weather = None
        self.load_min_mw = BENGALURU_MIN_DEMAND_MW
        self.load_max_mw = BENGALURU_MAX_DEMAND_MW

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
        """Load historical data for lag features and optional weather features."""
        data_files = [
            os.path.join(self.data_dir, "karnataka_realdata.csv"),
            "karnataka_realdata.csv",
        ]

        for data_path in data_files:
            if not os.path.exists(data_path):
                continue

            try:
                raw = pd.read_csv(data_path)
            except Exception:
                try:
                    raw = pd.read_csv(data_path, header=None)
                except Exception:
                    continue

            try:
                self.base_data = self._normalize_historical_data(raw)
                self.weather_profile = self._build_weather_profile(self.base_data)
                load_series = self.base_data["load"].dropna()
                if not load_series.empty:
                    lower_bound = float(load_series.quantile(0.01))
                    upper_bound = float(load_series.quantile(0.99))
                    margin = max(100.0, 0.05 * (upper_bound - lower_bound))
                    self.load_min_mw = max(0.0, lower_bound - margin)
                    self.load_max_mw = upper_bound + margin
                print(f"  Loaded historical data: {len(self.base_data)} records")
                print(f"  Forecast bounds: {self.load_min_mw:.0f} - {self.load_max_mw:.0f} MW")
                return
            except Exception:
                continue

        raise RuntimeError(
            "No real historical dataset was found. Expected karnataka_realdata.csv or another real data file."
        )

    def _normalize_historical_data(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Normalize supported Bengaluru historical data formats."""
        lower_cols = {str(col).strip().lower(): col for col in raw.columns}

        if {"timestamp", "load_mw"}.issubset(lower_cols):
            rename_map = {
                lower_cols["timestamp"]: "datetime",
                lower_cols["load_mw"]: "load",
            }
            for col in WEATHER_COLUMNS + ["is_holiday"]:
                if col in lower_cols:
                    rename_map[lower_cols[col]] = col
            data = raw.rename(columns=rename_map)[list(rename_map.values())]
            data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
        elif {"datetime", "load"}.issubset(lower_cols):
            rename_map = {
                lower_cols["datetime"]: "datetime",
                lower_cols["load"]: "load",
            }
            for col in WEATHER_COLUMNS + ["is_holiday"]:
                if col in lower_cols:
                    rename_map[lower_cols[col]] = col
            data = raw.rename(columns=rename_map)[list(rename_map.values())]
            data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
        else:
            if raw.shape[1] < 2:
                raise ValueError("Unsupported historical data format")
            data = raw.iloc[:, :2].copy()
            data.columns = ["datetime", "load"]
            data["datetime"] = pd.to_datetime(
                data["datetime"],
                format="%d/%m/%Y %H:%M",
                errors="coerce",
            )
            if data["datetime"].isna().all():
                data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")

        data["load"] = pd.to_numeric(data["load"], errors="coerce")
        for col in WEATHER_COLUMNS:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")
        if "is_holiday" in data.columns:
            data["is_holiday"] = (
                data["is_holiday"].astype(str).str.strip().str.lower().map(
                    {"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0}
                )
            ).fillna(pd.to_numeric(data["is_holiday"], errors="coerce")).fillna(0).astype(int)

        data = data.dropna(subset=["datetime", "load"]).drop_duplicates(subset=["datetime"])
        data = data.set_index("datetime").sort_index()
        data = data[data["load"] > 0]

        if len(data.index) > 1:
            step_minutes = int(round(data.index.to_series().diff().dropna().dt.total_seconds().median() / 60))
            full_index = pd.date_range(data.index.min(), data.index.max(), freq=f"{step_minutes}min")
            data = data.reindex(full_index)
            data = data.dropna(subset=["load"])

        return data

    def _resample_frame(self, frame: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Resample mixed numeric/flag data safely."""
        result = pd.DataFrame(index=pd.date_range(frame.index.min(), frame.index.max(), freq=freq))
        for col in frame.columns:
            if col == "is_holiday":
                result[col] = frame[col].resample(freq).ffill()
            else:
                result[col] = pd.to_numeric(frame[col], errors="coerce").resample(freq).interpolate(method="time")
        return result.ffill().bfill()

    def _build_weather_profile(self, data: pd.DataFrame):
        """Build weather backfill profiles from historical real data."""
        available = [col for col in WEATHER_COLUMNS if col in data.columns]
        if not available:
            return None

        prof = data[available].copy()
        prof["month"] = prof.index.month
        prof["hour"] = prof.index.hour
        return prof.groupby(["month", "hour"]).mean()

    def _fetch_forecast_weather(self, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch forward weather forecast when weather features are part of the model."""
        feature_cols = self.lstm_hybrid.feature_cols or self.gru_hybrid.feature_cols or []
        if not any(col in feature_cols for col in WEATHER_COLUMNS):
            return None
        if not SCRAPER_AVAILABLE:
            return None

        days_ahead = min(max((end_date.date() - datetime.now().date()).days + 2, 2), 16)
        try:
            scraper = BengaluruDataScraper()
            forecast = scraper.fetch_forecast_weather(days_ahead=days_ahead)
            if forecast is None or forecast.empty:
                return None
            forecast["timestamp"] = pd.to_datetime(forecast["timestamp"], errors="coerce")
            forecast = forecast.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
            forecast = self._resample_frame(forecast[WEATHER_COLUMNS], "5min")
            return forecast
        except Exception:
            return None

    def is_loaded(self, model_type: str = "all") -> bool:
        """Check if models are loaded."""
        if model_type == "lstm":
            return self.lstm_hybrid.lgb_model is not None
        if model_type == "gru":
            return self.gru_hybrid.lgb_model is not None
        return self._is_loaded

    def _get_daily_pattern(self, target_date: datetime) -> Optional[pd.Series]:
        """Get historical pattern for same day of week."""
        if self.base_data is None:
            return None
        target_dow = target_date.weekday()
        similar = self.base_data[self.base_data.index.dayofweek == target_dow]
        if len(similar) >= 288:
            return similar.groupby(similar.index.time)["load"].mean()
        return self.base_data.groupby(self.base_data.index.time)["load"].mean()

    def _get_holiday_flag(self, ts: datetime) -> int:
        """Return Karnataka holiday flag when library is available."""
        if not HOLIDAYS_AVAILABLE:
            return int(ts.weekday() >= 5)
        try:
            india_holidays = holidays.India(state="KA", years=[ts.year])
            return int(ts.date() in india_holidays)
        except Exception:
            return int(ts.weekday() >= 5)

    def _get_weather_context(self, ts: datetime) -> Dict[str, float]:
        """Get weather features from forecast first, then historical profile."""
        context = {}

        if self.forecast_weather is not None and ts in self.forecast_weather.index:
            row = self.forecast_weather.loc[ts]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            for col in WEATHER_COLUMNS:
                if col in row.index and pd.notna(row[col]):
                    context[col] = float(row[col])

        if self.weather_profile is not None:
            key = (ts.month, ts.hour)
            if key in self.weather_profile.index:
                profile_row = self.weather_profile.loc[key]
                for col in WEATHER_COLUMNS:
                    if col not in context and col in profile_row.index and pd.notna(profile_row[col]):
                        context[col] = float(profile_row[col])

        return context

    def predict(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate predictions for a date range with optional weather-aware features."""
        timestamps = []
        current = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_ts = end_date.replace(hour=23, minute=55, second=0, microsecond=0)

        while current <= end_ts:
            timestamps.append(current)
            current += timedelta(minutes=5)

        if not timestamps:
            timestamps = [start_date]

        self.forecast_weather = self._fetch_forecast_weather(start_date, end_date)

        mean_load = self.base_data["load"].mean() if self.base_data is not None else 3700
        daily_pattern = self._get_daily_pattern(start_date)
        recent_loads = list(self.base_data["load"].values[-576:]) if self.base_data is not None else [mean_load] * 576

        feature_cols = self.lstm_hybrid.feature_cols or self.gru_hybrid.feature_cols or []

        all_features = []
        for ts in timestamps:
            features = self._create_features(ts, recent_loads, daily_pattern, mean_load, feature_cols)
            X = np.array([features.get(col, 0.0) for col in feature_cols])
            all_features.append(X)
            estimated = mean_load + 500 * np.sin(np.pi * (ts.hour - 6) / 12)
            recent_loads.append(estimated)

        X_batch = np.array(all_features)

        if self.lstm_hybrid.lgb_model:
            base_preds = self.lstm_hybrid.lgb_model.predict(X_batch)
        else:
            base_preds = np.array([self._fallback_prediction(ts, daily_pattern, mean_load) for ts in timestamps])

        # Use actual LSTM model for residual prediction
        lstm_preds = base_preds.copy().astype(float)
        if self.lstm_hybrid.nn_model is not None:
            lstm_preds = self._apply_nn_residuals(
                base_preds,
                self.lstm_hybrid.nn_model,
                self.lstm_hybrid.residual_scaler,
                nlags=self.lstm_hybrid.nlags
            )
        
        # Use actual GRU model for residual prediction
        gru_preds = base_preds.copy().astype(float)
        if self.gru_hybrid.nn_model is not None:
            gru_preds = self._apply_nn_residuals(
                base_preds,
                self.gru_hybrid.nn_model,
                self.gru_hybrid.residual_scaler,
                nlags=self.gru_hybrid.nlags
            )

        lstm_preds = np.clip(lstm_preds, self.load_min_mw, self.load_max_mw)
        gru_preds = np.clip(gru_preds, self.load_min_mw, self.load_max_mw)

        return pd.DataFrame(
            {
                "loads_lightgbm_lstm": lstm_preds.tolist(),
                "loads_lightgbm_gru": gru_preds.tolist(),
            },
            index=timestamps,
        )

    def _create_features(
        self,
        ts: datetime,
        recent_loads: List[float],
        daily_pattern: Optional[pd.Series],
        mean_load: float,
        feature_cols: List[str],
    ) -> Dict[str, float]:
        """Create feature dictionary for a timestamp."""
        f = {}

        f["hour"] = ts.hour
        f["minute"] = ts.minute
        f["dayofweek"] = ts.weekday()
        f["dayofmonth"] = ts.day
        f["month"] = ts.month
        f["dayofyear"] = ts.timetuple().tm_yday
        f["is_weekend"] = int(ts.weekday() >= 5)
        f["time_slot"] = ts.hour * 12 + ts.minute // 5

        f["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
        f["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)
        f["day_sin"] = np.sin(2 * np.pi * ts.weekday() / 7)
        f["day_cos"] = np.cos(2 * np.pi * ts.weekday() / 7)
        f["month_sin"] = np.sin(2 * np.pi * ts.month / 12)
        f["month_cos"] = np.cos(2 * np.pi * ts.month / 12)
        f["dayofyear_sin"] = np.sin(2 * np.pi * f["dayofyear"] / 366)
        f["dayofyear_cos"] = np.cos(2 * np.pi * f["dayofyear"] / 366)

        for lag in FeatureEngineer.LAG_PERIODS:
            if len(recent_loads) >= lag:
                f[f"lag_{lag}"] = recent_loads[-lag]
            elif daily_pattern is not None:
                f[f"lag_{lag}"] = daily_pattern.get(ts.time(), mean_load)
            else:
                f[f"lag_{lag}"] = mean_load

        for window in FeatureEngineer.ROLLING_WINDOWS:
            if len(recent_loads) >= window:
                w = recent_loads[-window:]
                f[f"rolling_mean_{window}"] = np.mean(w)
                f[f"rolling_std_{window}"] = np.std(w) if len(w) > 1 else 0
                f[f"rolling_min_{window}"] = np.min(w)
                f[f"rolling_max_{window}"] = np.max(w)
            else:
                f[f"rolling_mean_{window}"] = mean_load
                f[f"rolling_std_{window}"] = 0
                f[f"rolling_min_{window}"] = mean_load * 0.9
                f[f"rolling_max_{window}"] = mean_load * 1.1

        f["diff_1"] = recent_loads[-1] - recent_loads[-2] if len(recent_loads) >= 2 else 0
        f["diff_288"] = recent_loads[-1] - recent_loads[-288] if len(recent_loads) >= 288 else 0

        weather_context = self._get_weather_context(ts)
        if "is_holiday" in feature_cols:
            f["is_holiday"] = self._get_holiday_flag(ts)

        for col in WEATHER_COLUMNS:
            if col in feature_cols:
                f[col] = weather_context.get(col, 0.0)
            if f"{col}_lag_1" in feature_cols:
                f[f"{col}_lag_1"] = weather_context.get(col, f.get(col, 0.0))
            if f"{col}_rolling_mean_12" in feature_cols:
                f[f"{col}_rolling_mean_12"] = weather_context.get(col, f.get(col, 0.0))
            if f"{col}_rolling_mean_24" in feature_cols:
                f[f"{col}_rolling_mean_24"] = weather_context.get(col, f.get(col, 0.0))

        return f

    def _apply_nn_residuals(self, base_preds: np.ndarray, nn_model, residual_scaler, nlags: int = 24) -> np.ndarray:
        """Apply neural network residual corrections to base predictions.
        
        The LSTM/GRU models are trained to predict residuals (actual - base_pred).
        During inference, we use historical actual values to initialize, then
        bootstrap forward using predicted residuals.
        """
        preds = base_preds.copy().astype(float)
        
        if not TF_AVAILABLE:
            return preds
        
        try:
            # Get historical actual values to bootstrap the residual sequence
            if self.base_data is not None and len(self.base_data) > 0:
                hist_actuals = self.base_data["load"].values[-(nlags*2):]
            else:
                # Fallback: use base_preds as approximation
                hist_actuals = base_preds[:min(nlags*2, len(base_preds))]
            
            # Initialize residual sequence with historical data
            # We approximate historical residuals from historical load data
            residual_history = hist_actuals.copy() if isinstance(hist_actuals, np.ndarray) else np.array(hist_actuals)
            
            # Predict residuals for each timestamp
            predicted_residuals = []
            current_residuals = residual_history[-nlags:].tolist()
            
            for i in range(len(base_preds)):
                # Build sequence from current residual history
                seq = np.array(current_residuals[-nlags:], dtype='float32')
                X_seq = seq.reshape(1, nlags, 1)
                
                # Predict next residual (scaled)
                residual_scaled = nn_model.predict(X_seq, verbose=0)[0, 0]
                
                # Unscale residual
                if residual_scaler is not None:
                    residual_unscaled = residual_scaler.inverse_transform(
                        np.array([[residual_scaled]])
                    )[0, 0]
                else:
                    residual_unscaled = residual_scaled
                
                predicted_residuals.append(residual_unscaled)
                
                # Add predicted residual to base prediction to get new "actual"
                # This bootstrapped value becomes part of residual history
                new_actual = base_preds[i] + residual_unscaled
                current_residuals.append(new_actual)
            
            # Add residuals to base predictions
            preds = preds + np.array(predicted_residuals)
            
        except Exception as e:
            print(f"Warning: Error applying NN residuals: {e}")
            # Return base predictions if NN fails
            return base_preds.copy()
        
        return preds

    def _fallback_prediction(self, ts: datetime, daily_pattern: Optional[pd.Series], mean_load: float) -> float:
        """Fallback prediction using historical patterns."""
        if daily_pattern is not None:
            base = daily_pattern.get(ts.time(), mean_load)
        else:
            hour = ts.hour + ts.minute / 60
            base = BENGALURU_BASE_DEMAND_MW + 0.45 * (BENGALURU_PEAK_DEMAND_MW - BENGALURU_BASE_DEMAND_MW) * np.sin(np.pi * (hour - 6) / 12)
        pred = base + np.random.normal(0, 50)
        return float(np.clip(pred, self.load_min_mw, self.load_max_mw))
