# Karnataka Load Forecast Prediction System

A real-time energy load forecasting application using official Karnataka SLDC load data and real Bengaluru weather data.

## Features

- Hybrid ML models: LightGBM + LSTM and LightGBM + GRU
- Real-time dashboard for grid status and forecast visualization
- 5-minute interval forecasting up to 7 days ahead
- Official Karnataka SLDC archived load-curve scraping
- Real Bengaluru weather enrichment from Open-Meteo
- Colab-friendly training script that defaults to the real dataset

## Architecture

- Frontend: React + Vite + TailwindCSS
- Backend: FastAPI + Python
- Models: LightGBM + TensorFlow/Keras
- Data: Pandas + NumPy + scikit-learn

## Installation

### Backend

```bash
cd backend
pip install -r requirements.txt
python main.py
```

Backend runs on [http://localhost:8002](http://localhost:8002).

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on [http://localhost:5173](http://localhost:5173).

## Training In Google Colab

Build the real dataset first, then train from the `backend` folder:

```bash
cd backend
python data_scraper.py
python train_hybrid_models.py
```

To force a specific real dataset:

```bash
python train_hybrid_models.py --data data/karnataka_realdata.csv --epochs 60
```

Real source used:

- Karnataka SLDC archived daily load-curve workbooks published by KPTCL
- Bengaluru weather history from Open-Meteo
- Karnataka holiday calendar

Important note:

- The new default training path is real Karnataka state load, not synthetic data
- The old synthetic sample-data path should not be used for training

## API Endpoints

- `POST /predict`: generate load forecast for a date range
- `GET /model-metrics`: get saved training metrics
- `GET /realtime-status`: get current simulated Bengaluru grid status
- `GET /historical-accuracy`: get recent accuracy summary
- `GET /health`: health check

### Example Request

```bash
curl -X POST "http://localhost:8002/predict" \
  -H "Content-Type: application/json" \
  -d '{"start_date": "2026-01-17", "end_date": "2026-01-18"}'
```

## Project Structure

```text
load_forecast_prediction/
|-- backend/
|   |-- main.py
|   |-- model.py
|   |-- schemas.py
|   |-- train_hybrid_models.py
|   |-- train_and_save.py
|   |-- requirements.txt
|   |-- data/
|   |   `-- karnataka_realdata.csv
|   `-- models/
|       |-- lstm_model.keras
|       |-- gru_model.keras
|       `-- lgb_model.txt
`-- frontend/
    |-- src/
    `-- package.json
```

## Notes

- The backend now reads saved model metadata so the UI can show the actual training metrics after you retrain.
- The real-data scraper depends on the official KPTCL workbook archive and Open-Meteo weather API.

## References

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [TensorFlow/Keras](https://www.tensorflow.org/)
