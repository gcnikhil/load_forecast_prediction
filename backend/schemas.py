from pydantic import BaseModel
from typing import List

class ForecastRequest(BaseModel):
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD

class ForecastResponse(BaseModel):
    timestamps: List[str]
    loads_lightgbm_lstm: List[float]  # LightGBM + LSTM hybrid predictions
    loads_lightgbm_gru: List[float]   # LightGBM + GRU hybrid predictions
    min_load: float
    max_load: float
    mean_load: float
