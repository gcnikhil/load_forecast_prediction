import pandas as pd
import numpy as np
from prophet import Prophet
import holidays
import joblib
import os
from datetime import datetime, timedelta

# Constants
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'delhi_model.joblib')

def generate_dummy_data(start_date='2023-01-01', periods=24*365*2, freq='H'):
    """Generates synthetic hourly load data for Delhi."""
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # Base load + seasonality
    t = np.linspace(0, 4*np.pi, periods)
    # Yearly seasonality (higher in summer)
    yearly = 1000 * np.sin(np.linspace(0, 4*np.pi, periods))  
    # Daily seasonality (higher during day)
    daily = 500 * np.sin(np.linspace(0, periods*2*np.pi, periods))
    # Noise
    noise = np.random.normal(0, 100, periods)
    
    # Trend (slightly increasing)
    trend = np.linspace(0, 500, periods)
    
    load = 3000 + yearly + daily + trend + noise
    
    # Ensure no negative values
    load = np.maximum(load, 1000)
    
    df = pd.DataFrame({'ds': dates, 'y': load})
    return df

def train_model():
    """Trains a Prophet model on dummy data."""
    print("Generating dummy Delhi load data...")
    df = generate_dummy_data()
    
    print("Initializing Prophet model with Indian holidays...")
    # Using a custom list of holidays if 'holidays' library issues, but we installed it.
    # Prophet can use built-in country holidays.
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True
    )
    
    # Add Indian holidays
    model.add_country_holidays(country_name='IN')
    
    print("Fitting model...")
    model.fit(df)
    
    # Create model directory if not exists
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    print("Delhi model training completed successfully.")

if __name__ == "__main__":
    train_model()
