"""
Continuous Learning Data Pipeline
==================================
Fetches real-time data from Karnataka SLDC (KPTCL) and manages incremental model updates.

Features:
- Scrapes load data from the official KPTCL daily load-curve workbooks every 5 minutes
- Stores data with proper timestamps
- Triggers retraining when enough new data is collected
- Maintains model versioning
"""

import os
import time
import csv
import logging
import schedule
from datetime import datetime, timedelta
import pandas as pd
from threading import Thread

# Fix 12: Import Bengaluru scraper instead of the Delhi one
from data_scraper import BengaluruDataScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DelhiSLDCScraper:
    """Scraper for Delhi SLDC electricity load data."""
    
    BASE_URL = "http://www.delhisldc.org/Loaddata.aspx"
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.session = self._create_session()
        os.makedirs(data_dir, exist_ok=True)
        
    def _create_session(self):
        """Create requests session with retry logic."""
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504]
        )
        session.mount('http://', HTTPAdapter(max_retries=retries))
        session.mount('https://', HTTPAdapter(max_retries=retries))
        return session
    
    def scrape_day(self, date_str: str) -> list:
        """
        Scrape load data for a specific date.
        
        Args:
            date_str: Date in format 'DD/MM/YYYY'
            
        Returns:
            List of (datetime_str, load_value) tuples
        """
        url = f"{self.BASE_URL}?mode={date_str}"
        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'id': 'ContentPlaceHolder3_DGGridAv'})
            
            if table is None:
                logger.warning(f"No data table found for {date_str}")
                return []
            
            rows = table.find_all('tr')[1:]  # Skip header
            data = []
            
            for row in rows:
                cells = row.find_all('font')
                if len(cells) >= 2:
                    time_str = cells[0].get_text(strip=True)
                    load_str = cells[1].get_text(strip=True)
                    
                    try:
                        load_value = float(load_str.replace(',', ''))
                        datetime_str = f"{date_str} {time_str}"
                        data.append((datetime_str, load_value))
                    except ValueError:
                        continue
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {date_str}: {e}")
            return []
    
    def scrape_latest(self) -> tuple:
        """
        Scrape the latest available data point.
        
        Returns:
            Tuple of (datetime, load) or None if failed
        """
        today = datetime.now().strftime('%d/%m/%Y')
        data = self.scrape_day(today)
        
        if data:
            return data[-1]  # Return most recent reading
        return None
    
    def fetch_historical(self, days_back: int = 31, save_path: str = None) -> pd.DataFrame:
        """
        Fetch historical data for the past N days.
        
        Args:
            days_back: Number of days to fetch
            save_path: Path to save CSV (optional)
            
        Returns:
            DataFrame with datetime and load columns
        """
        logger.info(f"Fetching {days_back} days of historical data...")
        
        all_data = []
        successful = 0
        failed = 0
        
        for i in range(days_back, 0, -1):
            date = datetime.now() - timedelta(days=i)
            date_str = date.strftime('%d/%m/%Y')
            
            day_data = self.scrape_day(date_str)
            
            if day_data:
                all_data.extend(day_data)
                successful += 1
                logger.info(f"  {date_str}: {len(day_data)} readings")
            else:
                failed += 1
                logger.warning(f"  {date_str}: Failed")
            
            time.sleep(0.5)  # Rate limiting
        
        logger.info(f"Completed: {successful} days success, {failed} days failed")
        logger.info(f"Total readings: {len(all_data)}")
        
        if not all_data:
            logger.error("No data could be fetched!")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data, columns=['datetime', 'load'])
        df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M', dayfirst=True)
        df = df.sort_values('datetime').drop_duplicates(subset='datetime', keep='first')
        
        if save_path:
            # Save in format expected by model
            df_save = df.copy()
            df_save['datetime'] = df_save['datetime'].dt.strftime('%d/%m/%Y %H:%M')
            df_save.to_csv(save_path, index=False, header=False)
            logger.info(f"Data saved to {save_path}")
        
        return df


class DataPipeline:
    """Manages continuous data collection and model updates using Bengaluru KPTCL source."""

    def __init__(self, data_dir: str = "data", model_dir: str = "models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        # Fix 12: Use BengaluruDataScraper instead of DelhiSLDCScraper
        self.scraper = BengaluruDataScraper()
        self.live_data_file = os.path.join(data_dir, "live_data.csv")
        self.last_retrain = None
        self.retrain_threshold = 288 * 7  # Retrain after 7 days of new data
        self.new_samples_count = 0

        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
    
    def append_data_point(self, datetime_str: str, load: float):
        """Append a new data point to live data file."""
        file_exists = os.path.exists(self.live_data_file)
        
        with open(self.live_data_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime_str, load])
        
        self.new_samples_count += 1
        logger.debug(f"Recorded: {datetime_str} = {load:.2f} MW")
    
    def fetch_and_store(self):
        """Fetch latest KPTCL data point and store it."""
        try:
            # Fix 12: Use BengaluruDataScraper — fetch today's load curve and pick the latest row
            from datetime import datetime as _dt
            import pandas as _pd

            today = _dt.now().replace(hour=0, minute=0, second=0, microsecond=0)
            workbook_bytes = self.scraper.download_daily_loadcurve(today, log_missing=False)
            if workbook_bytes is None:
                logger.warning("Could not fetch latest KPTCL load-curve workbook")
                return

            curve_df = self.scraper.parse_daily_loadcurve(workbook_bytes, today)
            if curve_df.empty:
                logger.warning("Parsed load-curve is empty")
                return

            # Pick the most recent reading up to now
            now = _dt.now()
            upto_now = curve_df[_pd.to_datetime(curve_df["timestamp"]) <= now]
            latest_row = upto_now.iloc[-1] if not upto_now.empty else curve_df.iloc[-1]

            # Fix 12: BengaluruDataScraper returns (timestamp, load_mw) columns
            timestamp_str = _pd.to_datetime(latest_row["timestamp"]).strftime("%Y-%m-%d %H:%M")
            load = float(latest_row["load_mw"])

            self.append_data_point(timestamp_str, load)
            logger.info(f"Latest reading: {timestamp_str} = {load:.2f} MW")

            if self.new_samples_count >= self.retrain_threshold:
                self.trigger_retrain()

        except Exception as e:
            logger.error(f"Error in fetch_and_store: {e}")
    
    def trigger_retrain(self):
        """Trigger model retraining with new data."""
        logger.info("=" * 50)
        logger.info("TRIGGERING MODEL RETRAIN")
        logger.info("=" * 50)
        
        try:
            # Merge live data with historical data
            self.merge_data_files()
            
            # Run training script
            from train_hybrid_models import train_hybrid_model
            
            merged_path = os.path.join(self.data_dir, "merged_data.csv")
            train_hybrid_model(merged_path, self.model_dir, epochs=50)
            
            # Reset counter
            self.new_samples_count = 0
            self.last_retrain = datetime.now()
            
            # Clear live data file
            if os.path.exists(self.live_data_file):
                os.rename(
                    self.live_data_file,
                    os.path.join(self.data_dir, f"archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                )
            
            logger.info("Retrain completed successfully!")
            
        except Exception as e:
            logger.error(f"Retrain failed: {e}")
    
    def merge_data_files(self):
        """Merge historical and live data files."""
        dfs = []

        # Load existing Karnataka historical data
        historical_files = [
            os.path.join(self.data_dir, 'karnataka_realdata.csv'),
            os.path.join(self.data_dir, 'bengaluru_realdata.csv'),
        ]

        for f in historical_files:
            if os.path.exists(f):
                try:
                    # Fix 12: Karnataka data uses timestamp/load_mw column names
                    df = pd.read_csv(f)
                    lower = {str(c).strip().lower(): c for c in df.columns}
                    ts_col = lower.get("timestamp") or lower.get("datetime")
                    load_col = lower.get("load_mw") or lower.get("load")
                    if ts_col and load_col:
                        df = df[[ts_col, load_col]].rename(columns={ts_col: "timestamp", load_col: "load_mw"})
                        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                        df = df.dropna(subset=["timestamp", "load_mw"])
                        dfs.append(df)
                except Exception as exc:
                    logger.warning(f"Failed to load {f}: {exc}")
                    continue

        # Load live data
        if os.path.exists(self.live_data_file):
            try:
                df = pd.read_csv(self.live_data_file, header=None, names=['timestamp', 'load_mw'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp', 'load_mw'])
                dfs.append(df)
            except Exception as exc:
                logger.warning(f"Failed to load live data: {exc}")

        if dfs:
            merged = pd.concat(dfs, ignore_index=True)
            merged = merged.sort_values('timestamp').drop_duplicates(subset='timestamp', keep='last')
            # Save in the format expected by train_hybrid_models.py
            merged.to_csv(os.path.join(self.data_dir, "merged_data.csv"), index=False)
            logger.info(f"Merged {len(merged)} records")
    
    def run_scheduled(self):
        """Run the pipeline on a schedule (every 5 minutes)."""
        logger.info("Starting continuous data pipeline...")
        logger.info("Fetching data every 5 minutes")
        
        # Fetch immediately
        self.fetch_and_store()
        
        # Schedule regular fetches
        schedule.every(5).minutes.do(self.fetch_and_store)
        
        while True:
            schedule.run_pending()
            time.sleep(60)


def run_initial_fetch():
    """Fetch initial historical data from Karnataka KPTCL."""
    # Fix 12: Use BengaluruDataScraper instead of DelhiSLDCScraper
    scraper = BengaluruDataScraper()
    from datetime import datetime, timedelta
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=31)).strftime("%Y-%m-%d")

    df = scraper.fetch_historical_kptcl_load(
        start_date=start_date,
        end_date=end_date,
    )

    if not df.empty:
        print(f"\nData fetched successfully!")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Total records: {len(df)}")
        print(f"  Load range: {df['load_mw'].min():.2f} - {df['load_mw'].max():.2f} MW")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Karnataka SLDC (KPTCL) Data Pipeline')
    parser.add_argument('--mode', type=str, default='fetch',
                       choices=['fetch', 'continuous', 'latest'],
                       help='Mode: fetch (historical), continuous (scheduled), latest (single)')
    parser.add_argument('--days', type=int, default=31,
                       help='Days of historical data to fetch')

    args = parser.parse_args()

    if args.mode == 'fetch':
        run_initial_fetch()
    elif args.mode == 'continuous':
        pipeline = DataPipeline()
        pipeline.run_scheduled()
    elif args.mode == 'latest':
        # Fix 12: Use BengaluruDataScraper for single latest fetch
        scraper = BengaluruDataScraper()
        from datetime import datetime as _dt
        import pandas as _pd
        today = _dt.now().replace(hour=0, minute=0, second=0, microsecond=0)
        wb = scraper.download_daily_loadcurve(today, log_missing=True)
        if wb:
            curve = scraper.parse_daily_loadcurve(wb, today)
            if not curve.empty:
                latest = curve.iloc[-1]
                print(f"Latest reading: {latest['timestamp']} = {float(latest['load_mw']):.2f} MW")
            else:
                print("Could not parse latest data")
        else:
            print("Could not fetch latest KPTCL workbook")
