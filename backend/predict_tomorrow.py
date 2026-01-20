import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from model import ModelService

# Initialize model service
model_service = ModelService()
model_service.load_models()  # Load your trained models

# Function to predict tomorrow's energy consumption
def predict_tomorrow():
    # Set the start and end date for tomorrow
    now = datetime.now()
    start_date = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = start_date + timedelta(days=1)

    # Generate predictions
    forecast_df = model_service.predict(start_date, end_date)

    # Extract predictions for LightGBM + LSTM and LightGBM + GRU
    timestamps = forecast_df.index.strftime("%Y-%m-%d %H:%M").tolist()
    loads_lstm = forecast_df['loads_lightgbm_lstm'].values
    loads_gru = forecast_df['loads_lightgbm_gru'].values

    return timestamps, loads_lstm, loads_gru

# Function to visualize the predictions
def visualize_predictions(timestamps, loads_lstm, loads_gru):
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, loads_lstm, label='LightGBM + LSTM', color='blue', linewidth=2)
    plt.plot(timestamps, loads_gru, label='LightGBM + GRU', color='orange', linewidth=2)
    plt.title("Predicted Energy Consumption for Tomorrow")
    plt.xlabel("Time (5-minute intervals)")
    plt.ylabel("Energy Consumption (MW)")
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    timestamps, loads_lstm, loads_gru = predict_tomorrow()
    visualize_predictions(timestamps, loads_lstm, loads_gru)