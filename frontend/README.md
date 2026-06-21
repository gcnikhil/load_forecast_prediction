# Energy Forecasting & Telemetry Dashboard (Frontend)

This is the React frontend for the Karnataka Load Forecast Prediction System. It provides an interactive dashboard, historical load analytics, model accuracy reports, and a What-If simulation engine.

## Features

* **Live Status Dashboard**: Visualizes current frequency, BESCOM load, Karnataka state demand, deviation, regional ESCOM draws, and live generation breakdown (Solar, Wind, Thermal, Hydro).
* **Deep Analytics**: Comprehensive forecast graphs showing predicted loads, Load Duration Curves (LDC), load frequency distribution, and historical model errors (MAPE/RMSE).
* **PDF Report Exports**: Seamless, formatted print styles for generating clean, portrait-aligned, black-and-white printouts/PDFs directly from the browser.
* **What-If Simulation Engine**: Test grid stability under monsoon surges, heatwaves, or industrial demand fluctuations with simulated weather parameters using LLMs.

## Core Stack

* **framework**: React + Vite
* **styling**: Vanilla CSS + TailwindCSS (using cyber/neon themes)
* **icons**: Lucide-React
* **charts**: Recharts (Responsive SVG plots)

---

## Getting Started

### 1. Install Dependencies

Navigate to the frontend folder and install the Node packages:

```bash
npm install
```

### 2. Run the Development Server

Start the development server:

```bash
npm run dev
```

* The application will run locally at [http://localhost:5173/](http://localhost:5173/).

### 3. OpenRouter API Key Setup

To use the **What-If Sim** page:
1. Obtain an API Key from [OpenRouter](https://openrouter.ai/).
2. Paste the API key into the **OpenRouter API Key** input box at the top of the What-If simulation page.
3. Select a model (defaults to the stable and fast `Llama 3.3 70B (Free)`) and trigger simulation scenarios.
