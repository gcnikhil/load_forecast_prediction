"""Quick test script for prediction logic."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info messages

from model import ModelService
from datetime import datetime, timedelta

model_service = ModelService()
model_service.load_models()

# Test prediction for tomorrow
start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
end = start

print('\n' + '='*60)
print('Running prediction test for:', start.strftime('%Y-%m-%d'))
print('='*60)
forecast = model_service.predict(start, end)
print('\n' + '='*60)
print('Forecast sample (first 10):')
print('='*60)
print(forecast.head(10))
print('\n' + '='*60)
print('Statistics:')
print('='*60)
print(f'LSTM - min: {forecast["loads_lightgbm_lstm"].min():.2f}, max: {forecast["loads_lightgbm_lstm"].max():.2f}, std: {forecast["loads_lightgbm_lstm"].std():.2f}')
print(f'GRU  - min: {forecast["loads_lightgbm_gru"].min():.2f}, max: {forecast["loads_lightgbm_gru"].max():.2f}, std: {forecast["loads_lightgbm_gru"].std():.2f}')
