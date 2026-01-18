"""
Continuous Learning Data Pipeline
==================================
Fetches real-time data from Delhi SLDC and manages incremental model updates.

Features:
- Scrapes data from http://www.delhisldc.org/Loaddata.aspx every 5 minutes
- Stores data with proper timestamps
- Triggers retraining when enough new data is collected
- Maintains model versioning
"""

import os
import time
import csv
import logging
import requests
import schedule
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from threading import Thread

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
    """Manages continuous data collection and model updates."""
    
    def __init__(self, data_dir: str = "data", model_dir: str = "models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.scraper = DelhiSLDCScraper(data_dir)
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
        """Fetch latest data point and store it."""
        try:
            result = self.scraper.scrape_latest()
            
            if result:
                datetime_str, load = result
                self.append_data_point(datetime_str, load)
                logger.info(f"Latest reading: {datetime_str} = {load:.2f} MW")
                
                # Check if retraining is needed
                if self.new_samples_count >= self.retrain_threshold:
                    self.trigger_retrain()
            else:
                logger.warning("Could not fetch latest data")
                
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
        
        # Load existing historical data
        historical_files = [
            os.path.join(self.data_dir, 'monthdata1.csv'),
            os.path.join(self.data_dir, 'delhi.csv'),
        ]
        
        for f in historical_files:
            if os.path.exists(f):
                try:
                    df = pd.read_csv(f, header=None, names=['datetime', 'load'])
                    df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M')
                    dfs.append(df)
                except:
                    continue
        
        # Load live data
        if os.path.exists(self.live_data_file):
            try:
                df = pd.read_csv(self.live_data_file, header=None, names=['datetime', 'load'])
                df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M')
                dfs.append(df)
            except:
                pass
        
        if dfs:
            merged = pd.concat(dfs, ignore_index=True)
            merged = merged.sort_values('datetime').drop_duplicates(subset='datetime', keep='last')
            merged['datetime'] = merged['datetime'].dt.strftime('%d/%m/%Y %H:%M')
            
            output_path = os.path.join(self.data_dir, "merged_data.csv")
            merged.to_csv(output_path, index=False, header=False)
            logger.info(f"Merged {len(merged)} records to {output_path}")
    
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
    """Fetch initial historical data."""
    scraper = DelhiSLDCScraper("data")
    
    # Fetch 31 days of data
    df = scraper.fetch_historical(
        days_back=31,
        save_path="data/delhi_latest.csv"
    )
    
    if not df.empty:
        print(f"\nData fetched successfully!")
        print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"  Total records: {len(df)}")
        print(f"  Load range: {df['load'].min():.2f} - {df['load'].max():.2f} MW")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Delhi SLDC Data Pipeline')
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
        scraper = DelhiSLDCScraper()
        result = scraper.scrape_latest()
        if result:
            print(f"Latest reading: {result[0]} = {result[1]:.2f} MW")
        else:
            print("Could not fetch latest data")
