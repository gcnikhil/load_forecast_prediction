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
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(base_dir, "models")
        self.data_dir = os.path.join(base_dir, "data")
        self._is_loaded = False
        self.weather_profile = None
        self.forecast_weather = None
        self._forecast_weather_cache = None
        self._forecast_weather_cache_until = None
        self._forecast_weather_cache_days = 0
        self.load_min_mw = None
        self.load_max_mw = None
        self.last_prediction_info = {
            "used_dummy": False,
            "dummy_reason": "",
            "dummy_model": "",
        }

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
            os.path.join(self.data_dir, "bengaluru_realdata.csv"),
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

        today = datetime.now().date()

        # If the full requested range is in the past, weather forecast API is unnecessary.
        if end_date.date() < today:
            return None

        # Open-Meteo expects forecast_days from today onward. Use only the necessary
        # span for the requested prediction window, clamped to API limits.
        window_start = max(start_date.date(), today)
        window_end = max(end_date.date(), window_start)
        days_ahead = min(max((window_end - today).days + 1, 1), 16)

        # Cache short-lived forecast responses to avoid repeated network latency.
        now = datetime.now()
        if (
            self._forecast_weather_cache is not None
            and self._forecast_weather_cache_until is not None
            and now <= self._forecast_weather_cache_until
            and self._forecast_weather_cache_days >= days_ahead
        ):
            return self._forecast_weather_cache

        try:
            scraper = BengaluruDataScraper()
            forecast = scraper.fetch_forecast_weather(days_ahead=days_ahead)
            if forecast is None or forecast.empty:
                return None
            forecast["timestamp"] = pd.to_datetime(forecast["timestamp"], errors="coerce")
            forecast = forecast.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
            forecast = self._resample_frame(forecast[WEATHER_COLUMNS], "5min")
            self._forecast_weather_cache = forecast
            self._forecast_weather_cache_days = days_ahead
            self._forecast_weather_cache_until = now + timedelta(minutes=30)
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

    def get_last_prediction_info(self) -> Dict[str, str]:
        """Get metadata from the most recent prediction call."""
        return self.last_prediction_info.copy()

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
            india_holidays = holidays.IN(subdiv="KA", years=[ts.year])
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

    def _fetch_actuals_for_range(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch actuals for the requested range, using base_data and scraper if needed."""
        actuals = pd.DataFrame()
        if self.base_data is not None:
            mask = (self.base_data.index >= start_date) & (self.base_data.index <= end_date)
            actuals = self.base_data.loc[mask, ['load']].copy()
            
        max_actual = actuals.index.max() if not actuals.empty else (self.base_data.index.max() if self.base_data is not None else start_date - timedelta(days=1))
        
        if SCRAPER_AVAILABLE and max_actual < min(end_date, datetime.now()):
            try:
                from data_scraper import BengaluruDataScraper
                scraper = BengaluruDataScraper()
                scrape_start = (max_actual + timedelta(days=1)).strftime("%Y-%m-%d")
                scrape_end = min(end_date, datetime.now()).strftime("%Y-%m-%d")
                
                scraped_df = scraper.fetch_historical_kptcl_load(scrape_start, scrape_end)
                if not scraped_df.empty:
                    scraped_df = scraped_df.set_index("timestamp")[["load_mw"]].rename(columns={"load_mw": "load"})
                    actuals = pd.concat([actuals, scraped_df])
                    if self.base_data is not None:
                        self.base_data = pd.concat([self.base_data, scraped_df])
                        self.base_data = self.base_data[~self.base_data.index.duplicated(keep='last')].sort_index()
            except Exception as e:
                print(f"Warning: Failed to scrape recent actuals: {e}")
                
        if not actuals.empty:
            actuals = actuals[~actuals.index.duplicated(keep='last')].sort_index()
            actuals_5min = actuals.resample("5min").interpolate(method="time")
            return actuals_5min
        return pd.DataFrame()

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
        fallback_used = False
        fallback_reason = ""

        actuals_df = self._fetch_actuals_for_range(start_date, end_date)
        base_preds = []
        is_actual = []
        
        for ts in timestamps:
            actual_load = None
            if not actuals_df.empty and ts in actuals_df.index and pd.notna(actuals_df.loc[ts, 'load']):
                actual_load = float(actuals_df.loc[ts, 'load'])
                
            if actual_load is not None:
                base_preds.append(actual_load)
                recent_loads.append(actual_load)
                is_actual.append(True)
                continue
                
            features = self._create_features(ts, recent_loads, daily_pattern, mean_load, feature_cols)
            if self.lstm_hybrid.lgb_model and feature_cols:
                X = np.array([features.get(col, 0.0) for col in feature_cols], dtype=float).reshape(1, -1)
                try:
                    base_pred = float(self.lstm_hybrid.lgb_model.predict(X)[0])
                except Exception as exc:
                    fallback_used = True
                    if not fallback_reason:
                        fallback_reason = f"LightGBM prediction failed: {exc}"
                    # Do not synthesize fallback values; mark as unavailable
                    base_pred = float("nan")
            else:
                fallback_used = True
                if not fallback_reason:
                    if self.lstm_hybrid.lgb_model is None:
                        fallback_reason = "LightGBM model artifact is not loaded"
                    else:
                        fallback_reason = "Model feature metadata is missing"
                # Do not synthesize fallback values; mark as unavailable
                base_pred = float("nan")

            base_preds.append(base_pred)
            # Use previous model prediction as the next-step lag signal.
            recent_loads.append(base_pred)
            # Mark this timestep as a model-generated value (not an actual observation)
            is_actual.append(False)

        base_preds = np.array(base_preds, dtype=float)

        # Use actual LSTM model for residual prediction
        lstm_preds = base_preds.copy().astype(float)
        if self.lstm_hybrid.nn_model is not None:
            lstm_preds = self._apply_nn_residuals(
                base_preds,
                self.lstm_hybrid.nn_model,
                self.lstm_hybrid.residual_scaler,
                nlags=self.lstm_hybrid.nlags,
                is_actual=is_actual,
            )
        
        # Use actual GRU model for residual prediction
        gru_preds = base_preds.copy().astype(float)
        if self.gru_hybrid.nn_model is not None:
            gru_preds = self._apply_nn_residuals(
                base_preds,
                self.gru_hybrid.nn_model,
                self.gru_hybrid.residual_scaler,
                nlags=self.gru_hybrid.nlags,
                is_actual=is_actual
            )

        lower_bound = self.load_min_mw if self.load_min_mw is not None else float(np.min(recent_loads))
        upper_bound = self.load_max_mw if self.load_max_mw is not None else float(np.max(recent_loads))
        if upper_bound <= lower_bound:
            upper_bound = lower_bound + max(100.0, abs(lower_bound) * 0.05)

        lstm_preds = np.clip(lstm_preds, lower_bound, upper_bound)
        gru_preds = np.clip(gru_preds, lower_bound, upper_bound)

        self.last_prediction_info = {
            "used_dummy": fallback_used,
            "dummy_reason": fallback_reason,
            "dummy_model": "fallback_prediction" if fallback_used else "",
        }

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

    def _apply_nn_residuals(self, base_preds: np.ndarray, nn_model, residual_scaler, nlags: int = 24, is_actual: List[bool] = None) -> np.ndarray:
        """Apply neural network residual corrections to base predictions.
        
        The LSTM/GRU models are trained to predict residuals (actual - base_pred).
        During inference, we bootstrap residuals from a neutral zero state and
        roll forward only with predicted residuals.
        """
        preds = base_preds.copy().astype(float)
        
        if not TF_AVAILABLE:
            return preds
            
        import tensorflow as tf
        
        try:
            # If base predictions contain NaNs (unavailable), skip NN residuals
            if np.isnan(base_preds).any():
                print("Info: Skipping NN residuals because some base predictions are NA")
                return preds

            steps = len(base_preds)
            if steps == 0:
                return preds

            # Cache a tf.function per model object and nlags to avoid repetitive retracing
            cache_key = (id(nn_model), int(nlags))
            if not hasattr(ModelService, "_tf_predict_fn_cache"):
                ModelService._tf_predict_fn_cache = {}

            cached_fn = ModelService._tf_predict_fn_cache.get(cache_key)
            if cached_fn is None:
                # Define a TF function with explicit input signature so tracing is stable
                @tf.function(
                    input_signature=[
                        tf.TensorSpec(shape=[1, int(nlags), 1], dtype=tf.float32),
                        tf.TensorSpec(shape=(), dtype=tf.int32),
                        tf.TensorSpec(shape=[None], dtype=tf.bool),
                    ],
                    reduce_retracing=True,
                )
                def _fast_residual_predict(initial_seq, num_steps, is_actual_mask):
                    current_seq = initial_seq
                    residuals = tf.TensorArray(tf.float32, size=num_steps)

                    i = tf.constant(0)

                    def cond(i, seq, res):
                        return i < num_steps

                    def body(i, seq, res):
                        r = tf.cond(
                            is_actual_mask[i],
                            lambda: tf.constant(0.0, dtype=tf.float32),
                            lambda: nn_model(seq, training=False)[0, 0],
                        )
                        res = res.write(i, r)
                        r_tensor = tf.reshape(r, [1, 1, 1])
                        seq = tf.concat([seq[:, 1:, :], r_tensor], axis=1)
                        return i + 1, seq, res

                    _, _, final_residuals = tf.while_loop(cond, body, [i, current_seq, residuals])
                    return final_residuals.stack()

                cached_fn = _fast_residual_predict
                ModelService._tf_predict_fn_cache[cache_key] = cached_fn

            # Start with zero residual history when true residual history is unavailable.
            initial_seq = np.zeros((1, int(nlags), 1), dtype='float32')

            if is_actual is None:
                is_actual = [False] * steps
            is_actual_tensor = tf.constant(is_actual, dtype=tf.bool)

            # Predict using cached TF function
            residuals_scaled_tensor = cached_fn(
                tf.convert_to_tensor(initial_seq, dtype=tf.float32),
                tf.constant(steps, dtype=tf.int32),
                is_actual_tensor,
            )
            residuals_scaled = residuals_scaled_tensor.numpy().reshape(-1, 1)
            
            # Unscale residual
            if residual_scaler is not None:
                residuals_unscaled = residual_scaler.inverse_transform(residuals_scaled).flatten()
            else:
                residuals_unscaled = residuals_scaled.flatten()
            
            # Add residuals to base predictions
            preds = preds + residuals_unscaled
            
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
            base = mean_load

        lower_bound = self.load_min_mw if self.load_min_mw is not None else 0.0
        upper_bound = self.load_max_mw if self.load_max_mw is not None else max(lower_bound + 100.0, mean_load * 2)
        return float(np.clip(base, lower_bound, upper_bound))
