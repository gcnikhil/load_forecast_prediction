"""Analyze historical load pattern."""
import pandas as pd

df = pd.read_csv('data/monthdata1.csv', header=None, names=['datetime', 'load'])
df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M')
df = df.set_index('datetime').sort_index()

# Get a typical day pattern (average by hour)
df['hour'] = df.index.hour
hourly = df.groupby('hour')['load'].mean()
print('Typical hourly load pattern (MW):')
print(hourly.to_string())
print()
print(f'Min load hour: {hourly.idxmin()}:00 ({hourly.min():.0f} MW)')
print(f'Max load hour: {hourly.idxmax()}:00 ({hourly.max():.0f} MW)')
print(f'Daily range: {hourly.max() - hourly.min():.0f} MW')
