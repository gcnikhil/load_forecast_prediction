from pydantic import BaseModel
from typing import List, Optional

class ForecastRequest(BaseModel):
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD

class ForecastResponse(BaseModel):
    timestamps: List[str]
    loads_lightgbm_gru: List[Optional[float]]   # LightGBM + GRU hybrid predictions (None = NA)
    min_load: Optional[float]
    max_load: Optional[float]
    mean_load: Optional[float]
    used_dummy: bool = False
    dummy_reason: str = ""
    dummy_model: str = ""
