import os
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from data_scraper import BengaluruDataScraper
from model import ModelService
from schemas import ForecastRequest, ForecastResponse

app = FastAPI(title="Bengaluru BESCOM Load Forecasting API - Hybrid Models")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_service = ModelService()
realtime_cache = {"expires_at": None, "payload": None, "error": None}
REALTIME_CACHE_TTL_SECONDS = 300
REALTIME_ERROR_CACHE_TTL_SECONDS = 120


def _load_latest_available_curve(scraper: BengaluruDataScraper, reference_day: datetime, lookback_days: int = 7):
    """Find the most recent available KPTCL workbook up to lookback window."""
    for i in range(lookback_days + 1):
        day = reference_day - timedelta(days=i)
        workbook_bytes = scraper.download_daily_loadcurve(day, log_missing=False)
        if workbook_bytes is None:
            continue
        try:
            curve_df = scraper.parse_daily_loadcurve(workbook_bytes, day).sort_values("timestamp")
            if not curve_df.empty:
                return day, curve_df
        except Exception:
            continue
    return None, None


def load_training_metadata():
    """Load saved training metadata when available."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_paths = [
        os.path.join(base_dir, "models", "metadata.joblib"),
        os.path.join(base_dir, "metadata.joblib"),
    ]
    for path in metadata_paths:
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception:
                continue
    return None


@app.on_event("startup")
async def startup_event():
    try:
        model_service.load_models()
    except Exception as e:
        print(f"Error loading models: {e}")


@app.post("/predict", response_model=ForecastResponse)
async def predict_load(request: ForecastRequest):
    try:
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d")

        if end_date < start_date:
            raise HTTPException(status_code=400, detail="End date must be after start date")

        if (end_date - start_date).days > 7:
            raise HTTPException(status_code=400, detail="Maximum forecast range is 7 days")

        forecast_df = model_service.predict(start_date, end_date)

        timestamps = forecast_df.index.strftime("%Y-%m-%d %H:%M").tolist()
        loads_lstm = forecast_df["loads_lightgbm_lstm"].tolist()
        loads_gru = forecast_df["loads_lightgbm_gru"].tolist()
        import math

        # Convert NaN to None for JSON-safe "NA" output
        def _nan_to_none(x):
            try:
                return None if (isinstance(x, float) and math.isnan(x)) else x
            except Exception:
                return x

        loads_lstm = [_nan_to_none(x) for x in loads_lstm]
        loads_gru = [_nan_to_none(x) for x in loads_gru]
        all_loads = [x for x in (loads_lstm + loads_gru) if x is not None]
        prediction_info = model_service.get_last_prediction_info()

        return ForecastResponse(
            timestamps=timestamps,
            loads_lightgbm_lstm=loads_lstm,
            loads_lightgbm_gru=loads_gru,
            min_load=min(all_loads) if all_loads else None,
            max_load=max(all_loads) if all_loads else None,
            mean_load=(sum(all_loads) / len(all_loads)) if all_loads else None,
            used_dummy=bool(prediction_info.get("used_dummy", False)),
            dummy_reason=str(prediction_info.get("dummy_reason", "")),
            dummy_model=str(prediction_info.get("dummy_model", "")),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": model_service.is_loaded()}


@app.get("/model-metrics")
def get_model_metrics():
    """Get model performance metrics from saved metadata when available."""
    metadata = load_training_metadata()
    if metadata:
        metrics = metadata.get("metrics", {})
        lstm_metrics = metrics.get("lstm_hybrid", {})
        gru_metrics = metrics.get("gru_hybrid", {})
        training_start = metadata.get("training_period_start", "")
        training_end = metadata.get("training_period_end", "")
        training_period = "Unknown"
        if training_start and training_end:
            training_period = f"{training_start[:10]} to {training_end[:10]}"

        return {
            "lstm_hybrid": {
                "name": "LightGBM + LSTM",
                "rmse_mw": round(float(lstm_metrics.get("rmse", 0)), 2),
                "mape_percent": round(float(lstm_metrics.get("mape", 0)), 2),
                "training_samples": int(metadata.get("training_samples", 0)),
                "features": int(metadata.get("feature_count", 0)),
                "architecture": metadata.get("architecture", {}).get(
                    "lstm_hybrid",
                    "LightGBM + LSTM hybrid",
                ),
                "characteristics": "Regularized residual model for smoother temporal correction",
            },
            "gru_hybrid": {
                "name": "LightGBM + GRU",
                "rmse_mw": round(float(gru_metrics.get("rmse", 0)), 2),
                "mape_percent": round(float(gru_metrics.get("mape", 0)), 2),
                "training_samples": int(metadata.get("training_samples", 0)),
                "features": int(metadata.get("feature_count", 0)),
                "architecture": metadata.get("architecture", {}).get(
                    "gru_hybrid",
                    "LightGBM + GRU hybrid",
                ),
                "characteristics": "Regularized residual model that reacts faster to recent changes",
            },
            "training_period": training_period,
            "data_source": (
                f"{metadata.get('dataset_name', 'Bengaluru dataset')} "
                f"({metadata.get('frequency_minutes', 'unknown')} minute cadence)"
            ),
            "anti_overfit": metadata.get("anti_overfit", {}),
        }

    raise HTTPException(status_code=404, detail="Training metadata not available. Please train models on real data.")


@app.get("/realtime-status")
def get_realtime_status():
    """Get real-time status from Karnataka SLDC load-curve source only."""
    now = datetime.now()

    if (
        realtime_cache.get("payload") is not None
        and realtime_cache["expires_at"] is not None
        and now <= realtime_cache["expires_at"]
    ):
        return realtime_cache["payload"]

    if (
        realtime_cache.get("error") is not None
        and realtime_cache["expires_at"] is not None
        and now <= realtime_cache["expires_at"]
    ):
        raise HTTPException(status_code=503, detail=realtime_cache["error"])

    try:
        scraper = BengaluruDataScraper()
        ref_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

        today, today_df = _load_latest_available_curve(scraper, ref_day, lookback_days=7)
        if today_df is None:
            raise RuntimeError("No KPTCL load-curve workbook available in last 7 days")

        upto_now = today_df[today_df["timestamp"] <= now]
        current_row = upto_now.iloc[-1] if not upto_now.empty else today_df.iloc[-1]
        current_ts = pd.to_datetime(current_row["timestamp"])

        current_load = float(current_row["load_mw"])
        drawal = current_load
        frequency = float(current_row["frequency_hz"]) if pd.notna(current_row["frequency_hz"]) else None

        yesterday, yesterday_df = _load_latest_available_curve(scraper, today - timedelta(days=1), lookback_days=7)
        if yesterday_df is None:
            raise RuntimeError("No prior KPTCL workbook available for schedule comparison")

        same_hour = yesterday_df[yesterday_df["timestamp"].dt.hour == current_ts.hour]
        schedule = float(same_hour.iloc[-1]["load_mw"]) if not same_hour.empty else None
        od_ud = (drawal - schedule) if schedule is not None else None

        today_max_row = today_df.loc[today_df["load_mw"].idxmax()]
        today_min_row = today_df.loc[today_df["load_mw"].idxmin()]
        yday_max_row = yesterday_df.loc[yesterday_df["load_mw"].idxmax()]
        yday_min_row = yesterday_df.loc[yesterday_df["load_mw"].idxmin()]

        payload = {
            "timestamp": current_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "frequency_hz": round(frequency, 2) if frequency is not None else None,
            "current_load_mw": int(round(current_load)),
            "schedule_mw": int(round(schedule)) if schedule is not None else None,
            "drawal_mw": int(round(drawal)),
            "od_ud_mw": int(round(od_ud)) if od_ud is not None else None,
            # Generation is not present in KPTCL load-curve workbook; keep null instead of synthesizing.
            "generation_mw": None,
            "today_max": {
                "value": int(round(float(today_max_row["load_mw"]))),
                "time": pd.to_datetime(today_max_row["timestamp"]).strftime("%H:%M:%S"),
            },
            "today_min": {
                "value": int(round(float(today_min_row["load_mw"]))),
                "time": pd.to_datetime(today_min_row["timestamp"]).strftime("%H:%M:%S"),
            },
            "yesterday_max": {
                "value": int(round(float(yday_max_row["load_mw"]))),
                "time": pd.to_datetime(yday_max_row["timestamp"]).strftime("%H:%M:%S"),
            },
            "yesterday_min": {
                "value": int(round(float(yday_min_row["load_mw"]))),
                "time": pd.to_datetime(yday_min_row["timestamp"]).strftime("%H:%M:%S"),
            },
            "source": "kptcl_sldc_loadcurve",
            "as_of_date": today.strftime("%Y-%m-%d"),
            "schedule_reference_date": yesterday.strftime("%Y-%m-%d"),
            "is_stale": today.date() < ref_day.date(),
        }

        realtime_cache["payload"] = payload
        realtime_cache["error"] = None
        realtime_cache["expires_at"] = now + timedelta(seconds=REALTIME_CACHE_TTL_SECONDS)
        return payload
    except Exception as exc:
        detail = f"Real-time source unavailable: {exc}"
        realtime_cache["payload"] = None
        realtime_cache["error"] = detail
        realtime_cache["expires_at"] = now + timedelta(seconds=REALTIME_ERROR_CACHE_TTL_SECONDS)
        raise HTTPException(status_code=503, detail=detail)


@app.get("/historical-accuracy")
def get_historical_accuracy():
    """Get historical model accuracy computed from actual historical data."""
    if not model_service.is_loaded() or model_service.base_data is None:
        raise HTTPException(status_code=503, detail="Model or historical data not loaded")
        
    days = []
    base_data = model_service.base_data
    
    # Check if we have enough data
    if len(base_data) < 288 * 7:
        raise HTTPException(status_code=400, detail="Not enough historical data to compute accuracy")

    # Get the last 7 days from the dataset
    end_date = base_data.index.max().replace(hour=23, minute=55, second=0, microsecond=0)
    start_date = (end_date - timedelta(days=6)).replace(hour=0, minute=0, second=0, microsecond=0)
    
    try:
        # Run predictions for the last 7 days
        predictions_df = model_service.predict(start_date, end_date)
        
        # Merge predictions with actuals
        merged = predictions_df.join(base_data['load'], how='inner')
        if merged.empty:
            raise HTTPException(status_code=400, detail="Could not align predictions with actual data")
            
        # Calculate daily metrics
        for i in range(7):
            day = start_date + timedelta(days=i)
            next_day = day + timedelta(days=1)
            daily_data = merged[(merged.index >= day) & (merged.index < next_day)]
            
            if len(daily_data) > 0:
                actual = daily_data['load'].values
                lstm_pred = daily_data['loads_lightgbm_lstm'].values
                gru_pred = daily_data['loads_lightgbm_gru'].values
                
                lstm_mape = np.mean(np.abs((actual - lstm_pred) / actual)) * 100
                gru_mape = np.mean(np.abs((actual - gru_pred) / actual)) * 100
                lstm_rmse = np.sqrt(np.mean((actual - lstm_pred)**2))
                gru_rmse = np.sqrt(np.mean((actual - gru_pred)**2))
                
                days.append({
                    "date": day.strftime("%Y-%m-%d"),
                    "lstm_mape": round(float(lstm_mape), 2),
                    "gru_mape": round(float(gru_mape), 2),
                    "lstm_rmse": round(float(lstm_rmse), 1),
                    "gru_rmse": round(float(gru_rmse), 1),
                })
        
        if not days:
            raise HTTPException(status_code=400, detail="No data available for the period")

        return {
            "period": "Last 7 days of actual data",
            "daily_accuracy": days,
            "average": {
                "lstm_mape": round(np.mean([d["lstm_mape"] for d in days]), 2),
                "gru_mape": round(np.mean([d["gru_mape"] for d in days]), 2),
                "lstm_rmse": round(np.mean([d["lstm_rmse"] for d in days]), 1),
                "gru_rmse": round(np.mean([d["gru_rmse"] for d in days]), 1),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
