"""
Real Karnataka Load + Weather Data Scraper
==========================================

This module builds a training dataset from official Karnataka SLDC load-curve
files and real Bengaluru weather data.

Official load source:
- KPTCL / Karnataka SLDC archived daily load-curve workbooks
- Example landing page: https://loadcurve.kptcl.net/LoadCurveUpload/lcdownloadview.asp
- File pattern observed from the official page:
  https://loadcurve.kptcl.net/LoadCurveUpload/data/D21APR2026.xls

Weather source:
- Open-Meteo historical archive API

The resulting dataset contains only real sourced values for:
- `load_mw` from KPTCL SLDC archived workbooks
- `temperature_celsius`, `humidity_percent`, `precipitation_mm`, `weather_code`
  from Open-Meteo
- `is_holiday` for Karnataka public holidays
"""

import logging
import os
from datetime import datetime, timedelta
from io import BytesIO
from typing import Optional

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPEN_METEO_API = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
KPTCL_LOADCURVE_INDEX = "https://loadcurve.kptcl.net/LoadCurveUpload/lcdownloadview.asp"
KPTCL_LOADCURVE_FILE = "https://loadcurve.kptcl.net/LoadCurveUpload/data/D{day}{month}{year}.xls"

BENGALURU_LAT = 12.9716
BENGALURU_LON = 77.5946

MONTH_ABBR = {
    1: "JAN",
    2: "FEB",
    3: "MAR",
    4: "APR",
    5: "MAY",
    6: "JUN",
    7: "JUL",
    8: "AUG",
    9: "SEP",
    10: "OCT",
    11: "NOV",
    12: "DEC",
}


class BengaluruDataScraper:
    """Scrapes Karnataka SLDC load data and Bengaluru weather data."""

    def __init__(self, request_timeout: int = 30):
        self.request_timeout = request_timeout

    def fetch_historical_weather(
        self,
        start_date: str = "2024-01-01",
        end_date: str = "2025-12-31",
    ) -> pd.DataFrame:
        """Fetch real historical weather data from Open-Meteo."""
        logger.info(f"Fetching Bengaluru weather from {start_date} to {end_date}...")

        params = {
            "latitude": BENGALURU_LAT,
            "longitude": BENGALURU_LON,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,weather_code",
            "timezone": "Asia/Kolkata",
        }

        response = requests.get(OPEN_METEO_API, params=params, timeout=self.request_timeout)
        response.raise_for_status()
        data = response.json()
        hourly = data["hourly"]

        weather_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(hourly["time"]),
                "temperature_celsius": hourly["temperature_2m"],
                "humidity_percent": hourly["relative_humidity_2m"],
                "precipitation_mm": hourly["precipitation"],
                "weather_code": hourly["weather_code"],
            }
        )
        weather_df["timestamp"] = weather_df["timestamp"].dt.floor("h")
        logger.info(f"Fetched {len(weather_df)} hourly weather rows")
        return weather_df

    def fetch_forecast_weather(self, days_ahead: int = 7) -> Optional[pd.DataFrame]:
        """Fetch forecast weather for inference use."""
        logger.info(f"Fetching {days_ahead}-day Bengaluru weather forecast...")
        params = {
            "latitude": BENGALURU_LAT,
            "longitude": BENGALURU_LON,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,weather_code",
            "forecast_days": days_ahead,
            "timezone": "Asia/Kolkata",
        }

        response = requests.get(OPEN_METEO_FORECAST, params=params, timeout=self.request_timeout)
        response.raise_for_status()
        data = response.json()
        hourly = data["hourly"]

        forecast_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(hourly["time"]),
                "temperature_celsius": hourly["temperature_2m"],
                "humidity_percent": hourly["relative_humidity_2m"],
                "precipitation_mm": hourly["precipitation"],
                "weather_code": hourly["weather_code"],
            }
        )
        forecast_df["timestamp"] = forecast_df["timestamp"].dt.floor("h")
        logger.info(f"Fetched {len(forecast_df)} hourly forecast rows")
        return forecast_df

    def fetch_indian_holidays(self, year: int, state: str = "KA") -> dict:
        """Fetch Karnataka holidays."""
        import holidays

        india_holidays = holidays.India(state=state, years=year)
        logger.info(f"Loaded {len(india_holidays)} holidays for {state} in {year}")
        return india_holidays

    def build_loadcurve_url(self, day: datetime) -> str:
        """Build the official KPTCL daily workbook URL."""
        return KPTCL_LOADCURVE_FILE.format(
            day=f"{day.day:02d}",
            month=MONTH_ABBR[day.month],
            year=day.year,
        )

    def download_daily_loadcurve(self, day: datetime) -> Optional[bytes]:
        """Download one official KPTCL daily load-curve workbook."""
        url = self.build_loadcurve_url(day)
        response = requests.get(url, timeout=self.request_timeout)
        if response.status_code == 404:
            logger.warning(f"No KPTCL load-curve workbook found for {day.date()} at {url}")
            return None
        response.raise_for_status()
        logger.info(f"Downloaded KPTCL load curve for {day.date()}")
        return response.content

    def parse_daily_loadcurve(self, workbook_bytes: bytes, day: datetime) -> pd.DataFrame:
        """
        Parse the official KPTCL workbook.

        The verified sample workbook contains a `TIME | LOAD | FREQUENCY`
        block in sheet `LOAD CURVE`.
        """
        sheet = pd.read_excel(BytesIO(workbook_bytes), sheet_name="LOAD CURVE", header=None, engine="xlrd")

        header_row = None
        header_col = None
        for row_idx in range(sheet.shape[0]):
            for col_idx in range(sheet.shape[1] - 2):
                values = [str(sheet.iat[row_idx, col_idx + offset]).strip().upper() for offset in range(3)]
                if values == ["TIME", "LOAD", "FREQUENCY"]:
                    header_row = row_idx
                    header_col = col_idx
                    break
            if header_row is not None:
                break

        if header_row is None or header_col is None:
            raise ValueError(f"Could not locate TIME/LOAD/FREQUENCY block for {day.date()}")

        block = sheet.iloc[header_row + 1 :, header_col : header_col + 3].copy()
        block.columns = ["hour", "load_mw", "frequency_hz"]

        block["hour"] = pd.to_numeric(block["hour"], errors="coerce")
        block["load_mw"] = pd.to_numeric(block["load_mw"], errors="coerce")
        block["frequency_hz"] = pd.to_numeric(block["frequency_hz"], errors="coerce")
        block = block.dropna(subset=["hour", "load_mw"])
        block = block[(block["hour"] >= 0) & (block["hour"] <= 24)]

        # Keep one real reading per hour and drop the terminal 24th hour to avoid
        # duplicating the next day's midnight.
        block = block[block["hour"] < 24].copy()
        block["hour"] = block["hour"].astype(int)

        block["timestamp"] = pd.to_datetime(day.date()) + pd.to_timedelta(block["hour"], unit="h")
        block["load_data_source"] = "kptcl_sldc_loadcurve"
        block["load_curve_url"] = self.build_loadcurve_url(day)
        return block[["timestamp", "load_mw", "frequency_hz", "load_data_source", "load_curve_url"]]

    def fetch_historical_kptcl_load(
        self,
        start_date: str = "2024-01-01",
        end_date: str = "2025-12-31",
    ) -> pd.DataFrame:
        """Fetch real hourly Karnataka state load from official KPTCL workbooks."""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        rows = []
        current = start.normalize()
        final = end.normalize()

        while current <= final:
            workbook_bytes = self.download_daily_loadcurve(current)
            if workbook_bytes is not None:
                try:
                    rows.append(self.parse_daily_loadcurve(workbook_bytes, current))
                except Exception as exc:
                    logger.warning(f"Failed to parse KPTCL workbook for {current.date()}: {exc}")
            current += timedelta(days=1)

        if not rows:
            raise RuntimeError("No official KPTCL load-curve files could be downloaded for the requested range.")

        load_df = pd.concat(rows, ignore_index=True)
        load_df = load_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        logger.info(f"Built real load dataset with {len(load_df)} hourly rows")
        return load_df

    def create_combined_dataset(
        self,
        start_date: str = "2024-01-01",
        end_date: str = "2025-12-31",
        output_path: str = "data/karnataka_realdata.csv",
    ) -> pd.DataFrame:
        """Create a real-only training dataset from official load + weather + holidays."""
        logger.info("=" * 60)
        logger.info("KARNATAKA REAL DATA SCRAPER")
        logger.info("=" * 60)
        logger.info(f"Official load index: {KPTCL_LOADCURVE_INDEX}")

        load_df = self.fetch_historical_kptcl_load(start_date, end_date)
        weather_df = self.fetch_historical_weather(start_date, end_date)

        weather_df["date"] = weather_df["timestamp"].dt.date
        start_year = int(start_date.split("-")[0])
        end_year = int(end_date.split("-")[0])

        holidays_data = {}
        for year in range(start_year, end_year + 1):
            holidays_data.update(self.fetch_indian_holidays(year))

        merged = load_df.merge(weather_df, on="timestamp", how="inner")
        merged["date"] = merged["timestamp"].dt.date
        merged["is_holiday"] = merged["date"].isin(holidays_data.keys())
        merged["holiday_name"] = merged["date"].map({date: name for date, name in holidays_data.items()})
        merged["day_of_week"] = merged["timestamp"].dt.day_name()
        merged["hour_of_day"] = merged["timestamp"].dt.hour
        merged["day_of_year"] = merged["timestamp"].dt.dayofyear
        merged["weather_data_source"] = "open-meteo"
        merged["has_real_load"] = True
        merged["grid_region"] = "karnataka_state"

        merged = merged.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        merged.to_csv(output_path, index=False)

        logger.info(f"Saved combined real dataset to {output_path}")
        logger.info(f"  Rows: {len(merged)}")
        logger.info(f"  Date range: {merged['timestamp'].min()} to {merged['timestamp'].max()}")
        logger.info(f"  Load range: {merged['load_mw'].min():.2f} - {merged['load_mw'].max():.2f} MW")
        logger.info(f"  Weather rows matched: {merged['temperature_celsius'].notna().sum()}")

        return merged


def main():
    """Main execution."""
    scraper = BengaluruDataScraper()
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    df = scraper.create_combined_dataset(
        start_date=start_date,
        end_date=end_date,
        output_path="data/karnataka_realdata.csv",
    )

    print("\n" + "=" * 60)
    print("Data Summary:")
    print("=" * 60)
    print(df.describe(include="all"))
    print("\n" + "=" * 60)
    print("Sample Data:")
    print("=" * 60)
    print(df.head(10))


if __name__ == "__main__":
    main()
