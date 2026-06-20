"""
Generate Sample Training Data
==============================
Creates realistic synthetic load data for model training when 
actual historical data is not available.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_realistic_load_data(
    start_date: str = "01/10/2025",
    days: int = 120,
    output_path: str = "data/bengaluru.csv"
):
    """
    Generate realistic electricity load data for Bengaluru/BESCOM.
    
    Parameters:
    - start_date: Start date in DD/MM/YYYY format
    - days: Number of days to generate
    - output_path: Path to save CSV
    """
    print(f"Generating {days} days of synthetic load data...")
    
    # Parse start date
    start = datetime.strptime(start_date, "%d/%m/%Y")
    
    # Generate 5-minute intervals
    intervals_per_day = 288  # 24 * 12
    total_intervals = days * intervals_per_day
    
    timestamps = [start + timedelta(minutes=5*i) for i in range(total_intervals)]
    
    # Base load parameters (MW) - realistic for Bengaluru/BESCOM
    base_load = 4200  # Average base load
    
    loads = []
    
    for ts in timestamps:
        hour = ts.hour + ts.minute / 60
        dow = ts.weekday()
        month = ts.month
        
        # Bengaluru/BESCOM profile:
        # - Night low (2-5 AM): ~2600-3000 MW
        # - Morning rise (6-10 AM): commercial ramp
        # - Afternoon moderate demand
        # - Evening peak (6-10 PM): strongest demand
        # - Night decline after 10 PM
        
        base_load = 3950  # Center point
        
        # Daily pattern matching Bengaluru behavior
        if 2 <= hour < 5:
            # Deep night low
            daily_factor = -1200 + np.random.normal(0, 45)
        elif 5 <= hour < 7:
            # Early morning rise
            daily_factor = -950 + (hour - 5) * 380
        elif 7 <= hour < 9:
            # Morning acceleration
            daily_factor = -150 + (hour - 7) * 520
        elif 9 <= hour < 12:
            # Morning commercial plateau
            daily_factor = 700 + 250 * np.sin(np.pi * (hour - 9) / 3)
        elif 12 <= hour < 15:
            # Afternoon slight dip
            daily_factor = 650 - (hour - 12) * 60
        elif 15 <= hour < 18:
            # Late afternoon rise
            daily_factor = 520 + (hour - 15) * 170
        elif 18 <= hour < 21:
            # Evening peak
            daily_factor = 1500 + 360 * np.sin(np.pi * (hour - 18) / 3)
        elif 21 <= hour < 24:
            # Night decline
            daily_factor = 1200 - (hour - 21) * 500
        else:
            # Late night (0-2 AM)
            daily_factor = -350 - hour * 220
        
        # Weekly pattern: weekends slightly lower
        if dow >= 5:  # Weekend
            weekly_factor = -200
        else:  # Weekday
            weekly_factor = 50
        
        # Seasonal pattern (Bengaluru has moderated seasonality)
        if month in [3, 4, 5]:
            seasonal_factor = 650 + 220 * np.sin(np.pi * (month - 3) / 2)
        elif month in [6, 7, 8, 9]:
            seasonal_factor = -120
        elif month in [10, 11]:
            seasonal_factor = 1000
        elif month in [12, 1, 2]:
            seasonal_factor = 150
        else:  # Spring/Fall
            seasonal_factor = 300
        
        # Add realistic noise
        noise = np.random.normal(0, 80)
        
        # Combine all factors
        load = base_load + daily_factor + weekly_factor + seasonal_factor + noise
        
        # Ensure realistic bounds matching Bengaluru demand
        load = max(2400, min(7800, load))
        
        loads.append(load)
    
    # Create dataframe
    df = pd.DataFrame({
        'datetime': [ts.strftime('%d/%m/%Y %H:%M') for ts in timestamps],
        'load': np.round(loads, 2)
    })
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV (with header, matching expected format)
    df.to_csv(output_path, index=False, header=True)
    
    print(f"SUCCESS: Generated {len(df)} data points")
    print(f"SUCCESS: Date range: {timestamps[0].strftime('%Y-%m-%d')} to {timestamps[-1].strftime('%Y-%m-%d')}")
    print(f"SUCCESS: Load range: {min(loads):.2f} - {max(loads):.2f} MW")
    print(f"SUCCESS: Mean load: {np.mean(loads):.2f} MW")
    print(f"SUCCESS: Saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    # Generate 90 days of training data
    df = generate_realistic_load_data(
        start_date="01/10/2025",
        days=90,
        output_path="data/bengaluru.csv"
    )

    # Also create copies for compatibility with existing training paths
    df.to_csv("data/monthdata1.csv", index=False, header=True)
    df.to_csv("data/delhi.csv", index=False, header=True)
    print("SUCCESS: Also saved to: data/monthdata1.csv")
    print("SUCCESS: Also saved to: data/delhi.csv")
