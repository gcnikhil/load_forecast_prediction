"""
Hybrid Load Forecasting Model Service
=====================================
<<<<<<< HEAD
Multi-step LightGBM + GRU hybrid model for Bengaluru/BESCOM load forecasting.
=======
LightGBM + LSTM and LightGBM + GRU hybrid models for Bengaluru/BESCOM load forecasting.
>>>>>>> origin/what-if-enhancements
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
<<<<<<< HEAD
REAL_SIGNAL_COLUMNS = WEATHER_COLUMNS + ["is_holiday"]

try:
    from tensorflow.keras.models import load_model
=======

try:
    from tensorflow.keras.models import load_model

>>>>>>> origin/what-if-enhancements
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available.")

try:
    from data_scraper import BengaluruDataScraper
<<<<<<< HEAD
=======

>>>>>>> origin/what-if-enhancements
    SCRAPER_AVAILABLE = True
except ImportError:
    SCRAPER_AVAILABLE = False

try:
    import holidays
<<<<<<< HEAD
=======

>>>>>>> origin/what-if-enhancements
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False


<<<<<<< HEAD
class ModelService:
    """Main service for multi-step load forecasting predictions."""

    def __init__(self):
        self.gru_model = None
        self.lgb_model = None
        self.metadata = None
        self.load_scaler = None
        self.feature_cols = []
        self.ensemble_mode = "stacking"  # "stacking" (LGB predicts actual) or "residual" (LGB predicts residual)
        
        self.seq_len = 168
        self.horizon = 72
        self.spd = 24
        self.freq_min = 60

=======
class FeatureEngineer:
    """Feature engineering metadata mirror used by inference."""

    # Fallback values — only used when metadata is missing
    LAG_PERIODS = [1, 2, 3, 6, 12, 24, 48, 168]
    ROLLING_WINDOWS = [6, 12, 24]
    EXOGENOUS_ROLLING_WINDOWS = [6, 12]


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
            # Fix 5: Prefer the per-hybrid LightGBM file; fall back to shared with warning
            lgb_preferred = os.path.join(model_dir, f"lgb_{self.nn_type}_model.txt")
            lgb_fallback  = os.path.join(model_dir, "lgb_model.txt")

            if os.path.exists(lgb_preferred):
                self.lgb_model = lgb.Booster(model_file=lgb_preferred)
                print(f"  [{self.model_name}] Loaded LightGBM from lgb_{self.nn_type}_model.txt")
            elif os.path.exists(lgb_fallback):
                self.lgb_model = lgb.Booster(model_file=lgb_fallback)
                print(f"  [{self.model_name}] WARNING: lgb_{self.nn_type}_model.txt not found "
                      f"— using shared lgb_model.txt. Re-train to fix.")

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
>>>>>>> origin/what-if-enhancements
        self.base_data = None
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(base_dir, "models")
        self.data_dir = os.path.join(base_dir, "data")
<<<<<<< HEAD
        
=======
>>>>>>> origin/what-if-enhancements
        self._is_loaded = False
        self.weather_profile = None
        self.forecast_weather = None
        self._forecast_weather_cache = None
        self._forecast_weather_cache_until = None
        self._forecast_weather_cache_days = 0
        self.load_min_mw = None
        self.load_max_mw = None
<<<<<<< HEAD
        
=======
>>>>>>> origin/what-if-enhancements
        self.last_prediction_info = {
            "used_dummy": False,
            "dummy_reason": "",
            "dummy_model": "",
        }

    def load_models(self):
<<<<<<< HEAD
        """Load the multi-step models and historical data."""
        print("=" * 50)
        print("Loading GRU+LightGBM Multi-Step Hybrid Model...")
        print("=" * 50)

        gru_path = os.path.join(self.model_dir, "gru_stage1_model.keras")
        lgb_path = os.path.join(self.model_dir, "lgb_stage2_model.txt")
        meta_path = os.path.join(self.model_dir, "gru_lgb_metadata.joblib")

        try:
            if os.path.exists(meta_path):
                self.metadata = joblib.load(meta_path)
                self.load_scaler = self.metadata.get("load_scaler")
                self.feature_cols = self.metadata.get("feature_cols", [])
                self.seq_len = self.metadata.get("seq_len", 168)
                self.horizon = self.metadata.get("horizon", 72)
                self.spd = self.metadata.get("samples_per_day", 24)
                self.freq_min = self.metadata.get("frequency_minutes", 60)
                self.ensemble_mode = self.metadata.get("ensemble_mode", "stacking")
                print(f"  Ensemble mode: {self.ensemble_mode}")

            if os.path.exists(lgb_path):
                self.lgb_model = lgb.Booster(model_file=lgb_path)

            if TF_AVAILABLE and os.path.exists(gru_path):
                self.gru_model = load_model(gru_path)

            self._is_loaded = (self.gru_model is not None) and (self.lgb_model is not None)
            
            if self._is_loaded:
                print("  Successfully loaded multi-step GRU and LightGBM models.")
            else:
                print("  Warning: One or more model artifacts are missing.")

        except Exception as e:
            print(f"  Error loading models: {e}")
            self._is_loaded = False

        self._load_historical_data()
=======
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
>>>>>>> origin/what-if-enhancements

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
<<<<<<< HEAD
            try:
                raw = pd.read_csv(data_path)
            except Exception:
                continue
=======

            try:
                raw = pd.read_csv(data_path)
            except Exception:
                try:
                    raw = pd.read_csv(data_path, header=None)
                except Exception:
                    continue
>>>>>>> origin/what-if-enhancements

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

<<<<<<< HEAD
        print("Warning: No valid historical dataset was found.")

    def _normalize_historical_data(self, raw: pd.DataFrame) -> pd.DataFrame:
        lower_cols = {str(col).strip().lower(): col for col in raw.columns}

        time_col = lower_cols.get("timestamp") or lower_cols.get("datetime")
        load_col = lower_cols.get("load_mw") or lower_cols.get("load")
        
        if not time_col or not load_col:
            raise ValueError("Unsupported historical data format")

        rename_map = {time_col: "datetime", load_col: "load"}
        for col in WEATHER_COLUMNS + ["is_holiday"]:
            if col in lower_cols:
                rename_map[lower_cols[col]] = col
                
        data = raw.rename(columns=rename_map)[list(rename_map.values())]
        data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
        data["load"] = pd.to_numeric(data["load"], errors="coerce")
        
        for col in WEATHER_COLUMNS:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")
                
        data = data.dropna(subset=["datetime", "load"]).drop_duplicates(subset=["datetime"])
        data = data.set_index("datetime").sort_index()
        return data

    def _build_weather_profile(self, data: pd.DataFrame):
        available = [col for col in WEATHER_COLUMNS if col in data.columns]
        if not available:
            return None
=======
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

>>>>>>> origin/what-if-enhancements
        prof = data[available].copy()
        prof["month"] = prof.index.month
        prof["hour"] = prof.index.hour
        return prof.groupby(["month", "hour"]).mean()

    def _fetch_forecast_weather(self, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
<<<<<<< HEAD
        if not any(col in self.feature_cols for col in WEATHER_COLUMNS):
=======
        """Fetch forward weather forecast when weather features are part of the model."""
        feature_cols = self.lstm_hybrid.feature_cols or self.gru_hybrid.feature_cols or []
        if not any(col in feature_cols for col in WEATHER_COLUMNS):
>>>>>>> origin/what-if-enhancements
            return None
        if not SCRAPER_AVAILABLE:
            return None

        today = datetime.now().date()
<<<<<<< HEAD
        if end_date.date() < today:
            return None

=======

        # If the full requested range is in the past, weather forecast API is unnecessary.
        if end_date.date() < today:
            return None

        # Open-Meteo expects forecast_days from today onward. Use only the necessary
        # span for the requested prediction window, clamped to API limits.
>>>>>>> origin/what-if-enhancements
        window_start = max(start_date.date(), today)
        window_end = max(end_date.date(), window_start)
        days_ahead = min(max((window_end - today).days + 1, 1), 16)

<<<<<<< HEAD
=======
        # Cache short-lived forecast responses to avoid repeated network latency.
>>>>>>> origin/what-if-enhancements
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
<<<<<<< HEAD
            # Resample to forecast model frequency
            forecast = forecast[WEATHER_COLUMNS].resample(f"{self.freq_min}min").interpolate(method="time").ffill().bfill()
=======
            forecast = self._resample_frame(forecast[WEATHER_COLUMNS], "5min")
>>>>>>> origin/what-if-enhancements
            self._forecast_weather_cache = forecast
            self._forecast_weather_cache_days = days_ahead
            self._forecast_weather_cache_until = now + timedelta(minutes=30)
            return forecast
        except Exception:
            return None

<<<<<<< HEAD
    def is_loaded(self) -> bool:
        return self._is_loaded

    def get_last_prediction_info(self) -> Dict[str, str]:
        return self.last_prediction_info.copy()

    def _get_holiday_flag(self, ts: datetime) -> int:
        if not HOLIDAYS_AVAILABLE:
            return int(ts.weekday() >= 5)
        try:
            india_holidays = holidays.IN(subdiv="KA", years=[ts.year])
=======
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
            india_holidays = holidays.India(state="KA", years=[ts.year])
>>>>>>> origin/what-if-enhancements
            return int(ts.date() in india_holidays)
        except Exception:
            return int(ts.weekday() >= 5)

    def _get_weather_context(self, ts: datetime) -> Dict[str, float]:
<<<<<<< HEAD
        context = {}
=======
        """Get weather features from forecast first, then historical profile."""
        context = {}

>>>>>>> origin/what-if-enhancements
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
<<<<<<< HEAD
=======
        """Fetch actuals for the requested range, using base_data and scraper if needed."""
>>>>>>> origin/what-if-enhancements
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
<<<<<<< HEAD
=======
                    if self.base_data is not None:
                        self.base_data = pd.concat([self.base_data, scraped_df])
                        self.base_data = self.base_data[~self.base_data.index.duplicated(keep='last')].sort_index()
>>>>>>> origin/what-if-enhancements
            except Exception as e:
                print(f"Warning: Failed to scrape recent actuals: {e}")
                
        if not actuals.empty:
            actuals = actuals[~actuals.index.duplicated(keep='last')].sort_index()
<<<<<<< HEAD
            # Do NOT interpolate to 5-min here. We interpolate to model frequency.
            actuals = actuals.resample(f"{self.freq_min}min").interpolate(method="time")
            return actuals
        return pd.DataFrame()

    def _get_past_actuals(self, end_ts: datetime, count: int) -> list:
        start_ts = end_ts - timedelta(minutes=self.freq_min * (count - 1))
        actuals_df = self._fetch_actuals_for_range(start_ts, end_ts)
        
        idx = pd.date_range(start_ts, end_ts, freq=f"{self.freq_min}min")
        
        if actuals_df.empty:
             mean_val = self.base_data["load"].mean() if self.base_data is not None else 3700.0
             return [mean_val] * count
             
        aligned = actuals_df.reindex(idx).interpolate(method="time").ffill().bfill()
        mean_val = self.base_data["load"].mean() if self.base_data is not None else 3700.0
        aligned["load"] = aligned["load"].fillna(mean_val)
        
        return aligned["load"].tolist()

    def _build_single_step_features(self, target_ts: datetime, step_ahead: int, gru_pred: float, past_actuals: list, feature_overrides: dict = None):
        f = {
            "step_ahead": step_ahead,
            "gru_pred": gru_pred,
        }
        
        f["anchor_load"] = past_actuals[-1]
        
        half = max(2, self.spd // 2)
        week = self.spd * 7
        lag_periods = sorted({1, 2, 3, half, self.spd, week})
        
        for lag in lag_periods:
            if len(past_actuals) >= lag + 1:
                f[f"anchor_lag_{lag}"] = past_actuals[-(lag + 1)]
            else:
                f[f"anchor_lag_{lag}"] = past_actuals[0]

        rolling_windows = sorted({max(2, self.spd // 4), half, self.spd})
        for w in rolling_windows:
            window_data = past_actuals[-w:] if len(past_actuals) >= w else past_actuals
            f[f"anchor_roll_mean_{w}"] = float(np.mean(window_data))
            f[f"anchor_roll_std_{w}"] = float(np.std(window_data))
            f[f"anchor_roll_min_{w}"] = float(np.min(window_data))
            f[f"anchor_roll_max_{w}"] = float(np.max(window_data))
            
        f["anchor_diff_1"] = past_actuals[-1] - past_actuals[-2] if len(past_actuals) >= 2 else 0
        f[f"anchor_diff_{self.spd}"] = past_actuals[-1] - past_actuals[-(self.spd + 1)] if len(past_actuals) >= self.spd + 1 else 0

        f["hour"] = target_ts.hour
        f["minute"] = target_ts.minute
        f["dayofweek"] = target_ts.weekday()
        f["dayofmonth"] = target_ts.day
        f["month"] = target_ts.month
        f["dayofyear"] = target_ts.timetuple().tm_yday
        f["is_weekend"] = int(target_ts.weekday() >= 5)

        minute_frac = (target_ts.hour * 60 + target_ts.minute) / 1440.0
        f["hour_sin"] = np.sin(2 * np.pi * minute_frac)
        f["hour_cos"] = np.cos(2 * np.pi * minute_frac)
        f["day_sin"] = np.sin(2 * np.pi * f["dayofweek"] / 7)
        f["day_cos"] = np.cos(2 * np.pi * f["dayofweek"] / 7)
        f["month_sin"] = np.sin(2 * np.pi * f["month"] / 12)
        f["month_cos"] = np.cos(2 * np.pi * f["month"] / 12)
        f["dayofyear_sin"] = np.sin(2 * np.pi * f["dayofyear"] / 366)
        f["dayofyear_cos"] = np.cos(2 * np.pi * f["dayofyear"] / 366)
        
        weather_context = self._get_weather_context(target_ts)
        for col in REAL_SIGNAL_COLUMNS:
             if col == "is_holiday":
                 f["is_holiday"] = self._get_holiday_flag(target_ts)
             else:
                 f[col] = weather_context.get(col, 0.0)
                 
        if feature_overrides:
            for k, v in feature_overrides.items():
                if k in f or k in REAL_SIGNAL_COLUMNS:
                    f[k] = int(v) if k == "is_holiday" else float(v)
                 
        return f

    def _fallback_dataframe(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        timestamps = []
        current = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_ts = end_date.replace(hour=23, minute=55, second=0, microsecond=0)
        while current <= end_ts:
            timestamps.append(current)
            current += timedelta(minutes=5)
            
        mean_val = self.base_data["load"].mean() if self.base_data is not None else 3700.0
        return pd.DataFrame({"loads_lightgbm_gru": [mean_val] * len(timestamps)}, index=timestamps)

    def predict(self, start_date: datetime, end_date: datetime, feature_overrides: dict = None, ignore_actuals: bool = False) -> pd.DataFrame:
        if not self._is_loaded or self.gru_model is None or self.lgb_model is None:
            self.last_prediction_info = {
                "used_dummy": True,
                "dummy_reason": "Models not loaded",
                "dummy_model": "fallback_prediction",
            }
            return self._fallback_dataframe(start_date, end_date)
            
        # Target timestamps exactly aligned to the model's training frequency (e.g., hourly)
        target_ts_list = []
        current = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_ts = end_date.replace(hour=23, minute=59, second=59, microsecond=0)
        while current <= end_ts:
            target_ts_list.append(current)
            current += timedelta(minutes=self.freq_min)
            
        # Origin time is the last step before the prediction window
        origin_time = start_date - timedelta(minutes=self.freq_min)
        past_actuals = self._get_past_actuals(origin_time, self.seq_len)
        
        # Scale and GRU predict
        past_scaled = self.load_scaler.transform(np.array(past_actuals).reshape(-1, 1)).reshape(1, self.seq_len, 1)
        gru_scaled = self.gru_model.predict(past_scaled, verbose=0)
        gru_preds = self.load_scaler.inverse_transform(gru_scaled)[0]
        
        # Build LightGBM features
        self.forecast_weather = self._fetch_forecast_weather(start_date, end_date)
        records = []
        
        for h, ts in enumerate(target_ts_list, start=1):
            if h > self.horizon:
                # If requested range exceeds trained horizon, just cap at the last predicted step
                break
                
            gru_val = gru_preds[h-1]
            rec = self._build_single_step_features(ts, h, gru_val, past_actuals, feature_overrides)
            records.append(rec)
            
        if not records:
             return self._fallback_dataframe(start_date, end_date)
             
        feat_df = pd.DataFrame(records)
        for col in self.feature_cols:
             if col not in feat_df.columns:
                 feat_df[col] = 0.0
                 
        X_lgb = feat_df[self.feature_cols].values
        lgb_output = self.lgb_model.predict(X_lgb)
        
        if self.ensemble_mode == "stacking":
            # Stacking: LightGBM directly predicts the actual load
            final_preds = lgb_output
        else:
            # Legacy residual mode: add LGB correction to GRU base
            final_preds = feat_df["gru_pred"].values + lgb_output
        
        if self.load_min_mw is not None and self.load_max_mw is not None:
            final_preds = np.clip(final_preds, self.load_min_mw, self.load_max_mw)
        
        # Assemble DataFrame using the actual model frequency
        out_df = pd.DataFrame({
             "loads_lightgbm_gru": final_preds
        }, index=target_ts_list[:len(final_preds)])
        
        # The frontend expects 5-minute resolution timestamps.
        # We upsample and interpolate our hourly (or other freq) predictions to 5-min.
        out_timestamps = []
        curr = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        while curr <= end_ts:
            out_timestamps.append(curr)
            curr += timedelta(minutes=5)
            
        out_df = out_df.reindex(out_timestamps).interpolate(method="time").ffill().bfill()
        
        self.last_prediction_info = {
            "used_dummy": False,
            "dummy_reason": "",
            "dummy_model": "",
        }
        return out_df
=======
            actuals_5min = actuals.resample("5min").interpolate(method="time")
            return actuals_5min
        return pd.DataFrame()

    def predict(
        self,
        start_date: datetime,
        end_date: datetime,
        feature_overrides: dict | None = None,  # NEW — dict of feature_name → value
        ignore_actuals: bool = False,           # NEW — forces pure model prediction
    ) -> pd.DataFrame:
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
        # Fix 6: Separate base prediction lists per hybrid
        lstm_base_preds = []
        gru_base_preds  = []
        is_actual = []

        for ts in timestamps:
            actual_load = None
            if not ignore_actuals and not actuals_df.empty and ts in actuals_df.index and pd.notna(actuals_df.loc[ts, 'load']):
                actual_load = float(actuals_df.loc[ts, 'load'])

            # Fix 6: Handle actual-load branch — append to BOTH lists
            if actual_load is not None:
                lstm_base_preds.append(actual_load)
                gru_base_preds.append(actual_load)
                recent_loads.append(actual_load)
                is_actual.append(True)
                continue

            features = self._create_features(ts, recent_loads, daily_pattern, mean_load, feature_cols)

            # Apply What-If overrides if provided (do not mutate original dict)
            if feature_overrides:
                for k, v in feature_overrides.items():
                    features[k] = v

            # Recalculate interaction features dynamically after any overrides are applied
            self._apply_interaction_features(features, feature_overrides)

            X = np.array([features.get(col, 0.0) for col in feature_cols], dtype=float).reshape(1, -1) if feature_cols else None

            # Fix 6: LSTM base — uses lstm_hybrid's own lgb_model
            if self.lstm_hybrid.lgb_model and feature_cols:
                try:
                    lstm_base_pred = float(self.lstm_hybrid.lgb_model.predict(X)[0])
                except Exception as exc:
                    fallback_used = True
                    if not fallback_reason:
                        fallback_reason = f"LightGBM (LSTM) prediction failed: {exc}"
                    lstm_base_pred = float("nan")
            else:
                fallback_used = True
                if not fallback_reason:
                    fallback_reason = (
                        "LightGBM (LSTM) model artifact is not loaded"
                        if self.lstm_hybrid.lgb_model is None
                        else "Model feature metadata is missing"
                    )
                lstm_base_pred = float("nan")

            # Fix 6: GRU base — uses gru_hybrid's own lgb_model (separate model)
            if self.gru_hybrid.lgb_model and feature_cols:
                try:
                    gru_base_pred = float(self.gru_hybrid.lgb_model.predict(X)[0])
                except Exception as exc:
                    fallback_used = True
                    if not fallback_reason:
                        fallback_reason = f"LightGBM (GRU) prediction failed: {exc}"
                    gru_base_pred = float("nan")
            else:
                fallback_used = True
                if not fallback_reason:
                    fallback_reason = (
                        "LightGBM (GRU) model artifact is not loaded"
                        if self.gru_hybrid.lgb_model is None
                        else "Model feature metadata is missing"
                    )
                gru_base_pred = float("nan")

            lstm_base_preds.append(lstm_base_pred)
            gru_base_preds.append(gru_base_pred)
            # Use LSTM base pred as next-step lag signal (fall back to mean if nan)
            recent_loads.append(lstm_base_pred if not np.isnan(lstm_base_pred) else mean_load)
            is_actual.append(False)

        # Fix 6: Apply NN residuals separately per hybrid using their own base preds and scalers
        lstm_preds = self._apply_nn_residuals(
            np.array(lstm_base_preds, dtype=float),
            self.lstm_hybrid.nn_model,
            self.lstm_hybrid.residual_scaler,
            nlags=self.lstm_hybrid.nlags,
            is_actual=is_actual,
        )
        gru_preds = self._apply_nn_residuals(
            np.array(gru_base_preds, dtype=float),
            self.gru_hybrid.nn_model,
            self.gru_hybrid.residual_scaler,
            nlags=self.gru_hybrid.nlags,
            is_actual=is_actual,
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

        metadata = self.lstm_hybrid.metadata or self.gru_hybrid.metadata or {}
        for col in WEATHER_COLUMNS:
            val = weather_context.get(col, 0.0)
            if col in feature_cols:
                f[col] = val
            lag1_key = f"{col}_lag_1"
            if lag1_key in feature_cols:
                f[lag1_key] = val
            
            exog_windows = metadata.get("exogenous_rolling_windows", FeatureEngineer.EXOGENOUS_ROLLING_WINDOWS)
            for w in exog_windows:
                rk = f"{col}_rolling_mean_{w}"
                if rk in feature_cols:
                    f[rk] = val

        return f

    def _apply_interaction_features(self, features: dict, overrides: dict = None):
        """Recompute derived interaction features, capturing any modified base features."""
        # Propagate overridden weather values to their lag and rolling mean features
        if overrides:
            for col in WEATHER_COLUMNS:
                if col in overrides:
                    try:
                        val = float(overrides[col])
                        lag1_key = f"{col}_lag_1"
                        if lag1_key in features:
                            features[lag1_key] = val
                        for w in [6, 12]:  # Standard exogenous windows
                            rk = f"{col}_rolling_mean_{w}"
                            if rk in features:
                                features[rk] = val
                    except ValueError:
                        pass

        # Holiday aliasing
        if "is_holiday" in features:
            features["is_karnataka_holiday"] = features["is_holiday"]

        # Temperature-load interaction
        if "temperature_celsius" in features:
            temp = float(features["temperature_celsius"])
            features["temp_sq"] = temp ** 2
            features["heat_stress"] = max(0.0, temp - 28.0)
            features["cold_stress"] = max(0.0, 18.0 - temp)
            features["temp_hour_interaction"] = temp * features.get("hour_sin", 0.0)
            
        # Humidity discomfort index
        if "humidity_percent" in features and "temperature_celsius" in features:
            temp = float(features["temperature_celsius"])
            hum = float(features["humidity_percent"])
            features["discomfort_index"] = temp - 0.55 * (1.0 - hum / 100.0) * (temp - 14.5)
            features["temp_humidity_interaction"] = temp * hum / 100.0
            
        # Seasonal dummies
        month = features.get("month", 1)
        features["is_summer"] = 1 if month in [3, 4, 5] else 0
        features["is_monsoon"] = 1 if month in [6, 7, 8, 9] else 0
        features["is_winter"] = 1 if month in [10, 11, 12, 1, 2] else 0
        features["season_hour_sin"] = features["is_summer"] * features.get("hour_sin", 0.0) - features["is_monsoon"] * features.get("hour_cos", 0.0)
        
        # Peak period flags
        hour = features.get("hour", 0)
        features["is_morning_peak"] = 1 if 9 <= hour <= 12 else 0
        features["is_evening_peak"] = 1 if 18 <= hour <= 22 else 0
        features["is_off_peak"] = 1 if 0 <= hour <= 5 else 0

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
>>>>>>> origin/what-if-enhancements
