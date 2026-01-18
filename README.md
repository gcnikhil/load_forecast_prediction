# ⚡ Delhi Load Forecast Prediction System

A real-time energy load forecasting application for **Delhi State Load Despatch Centre (SLDC)** using hybrid **LightGBM + LSTM/GRU** deep learning models.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Model Performance](#-model-performance)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Endpoints](#-api-endpoints)
- [Project Structure](#-project-structure)
- [Team](#-team)

---

## ✨ Features

- **Hybrid ML Models**: LightGBM + LSTM and LightGBM + GRU ensemble for accurate predictions
- **Real-time Dashboard**: Live grid status with frequency, load, schedule, and generation data
- **Interactive Analytics**: Hourly patterns, model comparison, load distribution charts
- **5-Minute Interval Forecasting**: High-resolution predictions up to 7 days ahead
- **Delhi SLDC Integration**: Based on actual grid parameters and historical data

---

## 🏗 Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   React + Vite  │────▶│  FastAPI Backend │────▶│  Hybrid Models  │
│   (Frontend)    │◀────│    (Port 8002)   │◀────│  LSTM/GRU+LGB   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                        │
        ▼                        ▼
   Recharts UI            Delhi SLDC Data
   TailwindCSS            (34,560 samples)
```

### Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 19, Vite, TailwindCSS, Recharts, Lucide Icons |
| Backend | FastAPI, Uvicorn, Python 3.10+ |
| ML Models | LightGBM, TensorFlow/Keras (LSTM, GRU) |
| Data | Pandas, NumPy, Scikit-learn |

---

## 📊 Model Performance

| Model | RMSE (MW) | MAPE (%) | Training Samples |
|-------|-----------|----------|------------------|
| **LightGBM + LSTM** | 9.54 | 0.19% | 34,560 |
| **LightGBM + GRU** | 9.97 | 0.21% | 34,560 |

- **Training Period**: October 2025 - January 2026
- **Features**: 34 engineered features (temporal, lag, rolling statistics)
- **Data Source**: Delhi SLDC (5-minute intervals)

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- Node.js 18+
- Git

### Backend Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Start the server
python main.py
```

Server runs at: `http://localhost:8002`

### Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend runs at: `http://localhost:5173`

---

## 💻 Usage

1. **Start Backend**: Run `python main.py` in the `backend` folder
2. **Start Frontend**: Run `npm run dev` in the `frontend` folder
3. **Open Browser**: Navigate to `http://localhost:5173`
4. **Generate Forecast**: Select date range and click "Generate Forecast"

### Dashboard Features

- **Live Grid Status**: Real-time frequency, load, schedule, and OD/UD values
- **Forecast Chart**: Interactive visualization of LSTM and GRU predictions
- **Model Stats**: Peak, min, and average values for each model

### Analytics Page

- Hourly load patterns
- Model comparison charts
- Load distribution histogram
- Prediction variance analysis

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Generate load forecast for date range |
| `/model-metrics` | GET | Get model performance metrics |
| `/realtime-status` | GET | Get current grid status |
| `/historical-accuracy` | GET | Get historical accuracy data |
| `/health` | GET | Health check endpoint |

### Example Request

```bash
curl -X POST "http://localhost:8002/predict" \
  -H "Content-Type: application/json" \
  -d '{"start_date": "2026-01-17", "end_date": "2026-01-18"}'
```

---

## 📁 Project Structure

```
load_forecast_prediction/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── model.py             # Model service & prediction logic
│   ├── schemas.py           # Pydantic schemas
│   ├── train_hybrid_models.py  # Model training script
│   ├── requirements.txt     # Python dependencies
│   ├── data/
│   │   └── delhi.csv        # Historical load data
│   └── models/
│       ├── lstm_model.keras # Trained LSTM model
│       ├── gru_model.keras  # Trained GRU model
│       └── lgb_model.txt    # LightGBM model
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx
│   │   │   ├── Analytics.jsx
│   │   │   └── AboutModels.jsx
│   │   ├── components/
│   │   │   ├── ForecastChart.jsx
│   │   │   ├── ControlPanel.jsx
│   │   │   └── Navigation.jsx
│   │   └── services/
│   │       └── api.js
│   └── package.json
└── README.md
```

---

## 👥 Team

| Name | Role |
|------|------|
| Nikhil GC | Developer |

---

## 📝 License

This project is for educational and research purposes.

---

## 🔗 References

- [Delhi SLDC](https://www.delhisldc.org/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [TensorFlow/Keras](https://www.tensorflow.org/)
