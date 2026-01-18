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
    output_path: str = "data/monthdata1.csv"
):
    """
    Generate realistic electricity load data for Delhi.
    
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
    
    # Base load parameters (MW) - realistic for Delhi
    base_load = 4500  # Average base load
    
    loads = []
    
    for ts in timestamps:
        hour = ts.hour + ts.minute / 60
        dow = ts.weekday()
        month = ts.month
        
        # REAL Delhi load pattern based on actual SLDC data:
        # - Night low (2-5 AM): ~2000-2200 MW
        # - Morning rise (5-9 AM): 2200-4500 MW  
        # - Morning peak (9-12 PM): ~5200-5500 MW
        # - Afternoon (12-5 PM): ~4500-5000 MW
        # - Evening peak (5-9 PM): ~5000-5400 MW
        # - Night decline (9 PM-2 AM): 5000-2500 MW
        
        # Base load for winter months (Jan-Feb, Nov-Dec)
        base_load = 3700  # Center point
        
        # Daily pattern matching real Delhi data
        if 2 <= hour < 5:
            # Deep night low: ~2000 MW
            daily_factor = -1700 + np.random.normal(0, 50)
        elif 5 <= hour < 7:
            # Early morning rise
            daily_factor = -1500 + (hour - 5) * 500
        elif 7 <= hour < 9:
            # Morning acceleration
            daily_factor = -500 + (hour - 7) * 700
        elif 9 <= hour < 12:
            # Morning peak: ~5200-5500 MW
            daily_factor = 900 + 400 * np.sin(np.pi * (hour - 9) / 3)
        elif 12 <= hour < 15:
            # Afternoon slight dip
            daily_factor = 800 - (hour - 12) * 100
        elif 15 <= hour < 18:
            # Late afternoon rise
            daily_factor = 500 + (hour - 15) * 200
        elif 18 <= hour < 21:
            # Evening peak: ~5200-5400 MW
            daily_factor = 1100 + 300 * np.sin(np.pi * (hour - 18) / 3)
        elif 21 <= hour < 24:
            # Night decline
            daily_factor = 1100 - (hour - 21) * 600
        else:
            # Late night (0-2 AM)
            daily_factor = -700 - hour * 300
        
        # Weekly pattern: weekends slightly lower
        if dow >= 5:  # Weekend
            weekly_factor = -200
        else:  # Weekday
            weekly_factor = 50
        
        # Seasonal pattern (Delhi has high summer demand, lower winter)
        if month in [5, 6, 7, 8]:  # Peak summer months
            seasonal_factor = 2000 + 500 * np.sin(np.pi * (month - 5) / 3)  # Up to 8500 MW
        elif month in [4, 9]:  # Transition to/from summer
            seasonal_factor = 1000
        elif month in [11, 12, 1, 2]:  # Winter months (current)
            seasonal_factor = 0  # Winter baseline
        else:  # Spring/Fall
            seasonal_factor = 300
        
        # Add realistic noise
        noise = np.random.normal(0, 80)
        
        # Combine all factors
        load = base_load + daily_factor + weekly_factor + seasonal_factor + noise
        
        # Ensure realistic bounds matching Delhi data
        load = max(1900, min(8700, load))
        
        loads.append(load)
    
    # Create dataframe
    df = pd.DataFrame({
        'datetime': [ts.strftime('%d/%m/%Y %H:%M') for ts in timestamps],
        'load': np.round(loads, 2)
    })
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV (without header, matching expected format)
    df.to_csv(output_path, index=False, header=False)
    
    print(f"✓ Generated {len(df)} data points")
    print(f"✓ Date range: {timestamps[0].strftime('%Y-%m-%d')} to {timestamps[-1].strftime('%Y-%m-%d')}")
    print(f"✓ Load range: {min(loads):.2f} - {max(loads):.2f} MW")
    print(f"✓ Mean load: {np.mean(loads):.2f} MW")
    print(f"✓ Saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    # Generate 90 days of training data
    df = generate_realistic_load_data(
        start_date="01/10/2025",
        days=90,
        output_path="data/monthdata1.csv"
    )
    
    # Also create a copy for compatibility
    df.to_csv("data/delhi.csv", index=False, header=False)
    print("✓ Also saved to: data/delhi.csv")
