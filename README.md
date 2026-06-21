# Karnataka Load Forecast Prediction System

A real-time energy load forecasting and analytics application using live Karnataka SLDC grid telemetry and real-time Bengaluru weather data.

## Features

- **Hybrid ML Model**: Multi-step LightGBM + GRU hybrid architecture specifically trained for Bengaluru/BESCOM load forecasting.
- **Real-Time Telemetry**: Live grid status fetched directly from the official Karnataka SLDC homepage (`https://kptclsldc.in/`) showing BESCOM load, State Demand, Grid Frequency, Deviation (UI), and Generation/ESCOM breakdowns.
- **Interactive Analytics**: Deep forecast statistics, Load Duration Curves (LDC), load frequency distributions, and error metrics (MAPE/RMSE).
- **What-If Simulation Engine**: Multi-scenario generator powered by OpenRouter LLMs (defaults to stable `Llama 3.3 70B (Free)`) allowing you to simulate heatwaves, monsoon rain, or industrial demand surges.
- **PDF Report Generation**: Elegant, high-contrast, print-formatted PDF report downloads directly from the Analytics page.

## Architecture

- **Frontend**: React + Vite + TailwindCSS + Recharts
- **Backend**: FastAPI + Python (Uvicorn)
- **ML Engine**: TensorFlow/Keras (GRU) + LightGBM + Joblib + Pandas/NumPy
- **Data Sc scraper**: BeautifulSoup + Requests (KPTCL SLDC Live Scraper & Open-Meteo Weather API)

---

## Installation & Running

### 1. Run the Backend API

```bash
cd backend
# Create and activate python virtual environment, install requirements
pip install -r requirements.txt
python main.py
```
* The backend API server starts at [http://localhost:8002](http://localhost:8002).

### 2. Run the Frontend Dashboard

```bash
cd frontend
# Install node packages
npm install
# Run development server
npm run dev
```
* The frontend web application opens at [http://localhost:5173](http://localhost:5173).

---

## Training the Models

The training script resides in the root directory and runs on a real historical dataset:

```bash
# From the root directory:
python train_gru_lgb_hybrid.py --epochs 60
```

By default, the training pipeline:
1. Normalizes the historical dataset (`backend/data/karnataka_realdata.csv`).
2. Trains a Stage 1 GRU model to capture sequential historical demand dependencies.
3. Fits a Stage 2 LightGBM model to predict the residual error based on weather and calendar features.
4. Outputs the model binaries and metadata directly into `backend/models/`.

---

## API Endpoints Reference

- `POST /predict`: Generate load forecast for a date range
- `GET /model-metrics`: Retrieve current model metrics and parameters
- `GET /realtime-status`: Fetch live real-time BESCOM and Karnataka grid status
- `GET /historical-accuracy`: Retrieve recent historical forecast accuracy (MAPE/RMSE)
- `GET /health`: API health check

---

## Project Structure

```text
load_forecast_prediction/
|-- backend/
|   |-- main.py                    # FastAPI server & routes
|   |-- model.py                   # Prediction service & model loaders
|   |-- data_scraper.py            # Live SLDC webpage and Open-Meteo scraper
|   |-- requirements.txt           # Backend python dependencies
|   |-- schemas.py                 # Pydantic schemas
|   |-- test_backend.py            # Unit tests
|   |-- data/
|   |   `-- karnataka_realdata.csv # Real historical load dataset
|   `-- models/                    # Model binary storage (loaded at runtime)
|       |-- gru_stage1_model.keras
|       |-- lgb_stage2_model.txt
|       |-- gru_lgb_metadata.joblib
|       `-- gru_lgb_training_summary.json
|-- frontend/
|   |-- src/                       # React source files
|   |-- package.json               # Node dependencies
|   `-- tailwind.config.js
|-- train_gru_lgb_hybrid.py        # Model training script (root directory)
`-- README.md
```

## References

- [Karnataka SLDC Page](https://kptclsldc.in/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [TensorFlow Core](https://www.tensorflow.org/)
