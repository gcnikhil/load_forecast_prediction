"""
Visualize Load Data
===================
Plot electricity load over time from monthdata1.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Read data
print("Loading data...")
data = pd.read_csv('data/monthdata1.csv', header=None, names=['datetime', 'load'])
data['datetime'] = pd.to_datetime(data['datetime'], format='%d/%m/%Y %H:%M')

print(f"Loaded {len(data)} records")
print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")
print(f"Load range: {data['load'].min():.2f} - {data['load'].max():.2f} MW")

# Create figure with multiple views
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# Plot 1: Full dataset
ax1 = axes[0]
ax1.plot(data['datetime'], data['load'], linewidth=0.8, color='#00FFF6', alpha=0.8)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Load (MW)', fontsize=12)
ax1.set_title('Complete Load Profile (All Data)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 2: Last 7 days
ax2 = axes[1]
last_week = data.tail(7 * 288)  # 7 days × 288 intervals
ax2.plot(last_week['datetime'], last_week['load'], linewidth=1.2, color='#FF2A6D')
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Load (MW)', fontsize=12)
ax2.set_title('Last 7 Days Detail', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
ax2.xaxis.set_major_locator(mdates.DayLocator())
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 3: Single day (most recent)
ax3 = axes[2]
last_day = data.tail(288)  # 1 day = 288 intervals
ax3.plot(last_day['datetime'], last_day['load'], linewidth=2, color='#F9F871', marker='o', markersize=2)
ax3.set_xlabel('Time of Day', fontsize=12)
ax3.set_ylabel('Load (MW)', fontsize=12)
ax3.set_title(f'Single Day Detail: {last_day["datetime"].iloc[0].strftime("%Y-%m-%d")}', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax3.xaxis.set_major_locator(mdates.HourLocator(interval=2))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Add statistics box
stats_text = f"""
Data Statistics:
• Total Points: {len(data):,}
• Days: {len(data) // 288}
• Peak Load: {data['load'].max():.2f} MW
• Min Load: {data['load'].min():.2f} MW
• Avg Load: {data['load'].mean():.2f} MW
• Range: {data['load'].max() - data['load'].min():.2f} MW
"""
fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)

# Save figure
output_file = 'load_visualization.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved to: {output_file}")

# Show plot
plt.show()
