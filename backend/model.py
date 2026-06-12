"""
Hybrid Load Forecasting Model Service
=====================================
Multi-step LightGBM + GRU hybrid model for Bengaluru/BESCOM load forecasting.
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
REAL_SIGNAL_COLUMNS = WEATHER_COLUMNS + ["is_holiday"]

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
        prof = data[available].copy()
        prof["month"] = prof.index.month
        prof["hour"] = prof.index.hour
        return prof.groupby(["month", "hour"]).mean()

    def _fetch_forecast_weather(self, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        if not any(col in self.feature_cols for col in WEATHER_COLUMNS):
            return None
        if not SCRAPER_AVAILABLE:
            return None

        today = datetime.now().date()
        if end_date.date() < today:
            return None

        window_start = max(start_date.date(), today)
        window_end = max(end_date.date(), window_start)
        days_ahead = min(max((window_end - today).days + 1, 1), 16)

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
            # Resample to forecast model frequency
            forecast = forecast[WEATHER_COLUMNS].resample(f"{self.freq_min}min").interpolate(method="time").ffill().bfill()
            self._forecast_weather_cache = forecast
            self._forecast_weather_cache_days = days_ahead
            self._forecast_weather_cache_until = now + timedelta(minutes=30)
            return forecast
        except Exception:
            return None

    def is_loaded(self) -> bool:
        return self._is_loaded

    def get_last_prediction_info(self) -> Dict[str, str]:
        return self.last_prediction_info.copy()

    def _get_holiday_flag(self, ts: datetime) -> int:
        if not HOLIDAYS_AVAILABLE:
            return int(ts.weekday() >= 5)
        try:
            india_holidays = holidays.IN(subdiv="KA", years=[ts.year])
            return int(ts.date() in india_holidays)
        except Exception:
            return int(ts.weekday() >= 5)

    def _get_weather_context(self, ts: datetime) -> Dict[str, float]:
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
            except Exception as e:
                print(f"Warning: Failed to scrape recent actuals: {e}")
                
        if not actuals.empty:
            actuals = actuals[~actuals.index.duplicated(keep='last')].sort_index()
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

    def _build_single_step_features(self, target_ts: datetime, step_ahead: int, gru_pred: float, past_actuals: list):
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

    def predict(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
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
            rec = self._build_single_step_features(ts, h, gru_val, past_actuals)
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
