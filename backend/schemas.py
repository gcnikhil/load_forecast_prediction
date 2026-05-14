from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ForecastRequest(BaseModel):
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD

class ForecastResponse(BaseModel):
    timestamps: List[str]
    loads_lightgbm_lstm: List[Optional[float]]  # LightGBM + LSTM hybrid predictions (None = NA)
    loads_lightgbm_gru: List[Optional[float]]   # LightGBM + GRU hybrid predictions (None = NA)
    min_load: Optional[float]
    max_load: Optional[float]
    mean_load: Optional[float]
    used_dummy: bool = False
    dummy_reason: str = ""
    dummy_model: str = ""


class WhatIfRequest(BaseModel):
    """
    Request body for the /whatif/predict endpoint.
    start_date and end_date follow the same YYYY-MM-DD convention as /predict.
    feature_overrides: flat dict of feature name → new scalar value.
      Keys must match names used in ModelService._create_features().
      Examples:
        {"is_holiday": 1}
        {"temperature_celsius": 38.5, "is_holiday": 1}
        {"humidity_percent": 90, "precipitation_mm": 15}
    """
    start_date: str
    end_date: str
    feature_overrides: Dict[str, Any]


class WhatIfResponse(BaseModel):
    """
    Response from /whatif/predict.
    Returns both baseline and modified forecasts so the frontend
    can render a side-by-side comparison chart.
    """
    timestamps: List[str]
    baseline_lstm: List[Optional[float]]
    baseline_gru: List[Optional[float]]
    modified_lstm: List[Optional[float]]
    modified_gru: List[Optional[float]]
    delta_lstm: List[Optional[float]]   # modified - baseline, per timestamp
    delta_gru: List[Optional[float]]
    summary: Dict[str, Any]             # mean/min/max for both scenarios
    applied_overrides: Dict[str, Any]   # echo back what was actually applied
