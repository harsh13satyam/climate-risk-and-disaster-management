import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print('=== CLIMATE RISK AND DISASTER MANAGEMENT PROJECT ===')
print('Loading and analyzing India weather data...')

df = pd.read_csv('india_weather_data.csv')
print(f'Dataset shape: {df.shape}')

df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['hour'] = df['date'].dt.hour

print(f'Date range: {df["date"].min()} to {df["date"].max()}')
print(f'Total records after cleaning: {len(df)}')

numeric_columns = ['wind_speed', 'cloud_cover', 'precipitation_probability', 'pressure_surface_level', 
                  'dew_point', 'uv_index', 'visibility', 'rainfall', 'solar_radiation', 'snowfall',
                  'max_temperature', 'min_temperature', 'max_humidity', 'min_humidity']

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['max_temperature', 'min_temperature'])
print(f'Records after temperature cleaning: {len(df)}')

df['temperature_range'] = df['max_temperature'] - df['min_temperature']
df['avg_temperature'] = (df['max_temperature'] + df['min_temperature']) / 2
df['humidity'] = (df['max_humidity'] + df['min_humidity']) / 2

heat_wave_threshold = df['max_temperature'].quantile(0.95)
cold_wave_threshold = df['min_temperature'].quantile(0.05)

print(f'Heat wave threshold (95th percentile): {heat_wave_threshold:.2f}°C')
print(f'Cold wave threshold (5th percentile): {cold_wave_threshold:.2f}°C')

df['predicted_heatwave'] = (df['max_temperature'] >= heat_wave_threshold).astype(int)
df['predicted_coldwave'] = (df['min_temperature'] <= cold_wave_threshold).astype(int)

print(f'Heat wave events: {df["predicted_heatwave"].sum()}')
print(f'Cold wave events: {df["predicted_coldwave"].sum()}')
print('Data preprocessing completed successfully!')

# Create visualizations and save them
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
df['max_temperature'].hist(bins=50, alpha=0.7)
plt.title('Max Temperature Distribution')
plt.xlabel('Temperature (°C)')

plt.subplot(2, 3, 2)
df['min_temperature'].hist(bins=50, alpha=0.7, color='blue')
plt.title('Min Temperature Distribution')
plt.xlabel('Temperature (°C)')

plt.subplot(2, 3, 3)
df['predicted_heatwave'].value_counts().plot(kind='bar')
plt.title('Heat Wave Occurrences')
plt.xlabel('Heat Wave (0=No, 1=Yes)')

plt.subplot(2, 3, 4)
df['month'].value_counts().sort_index().plot(kind='bar')
plt.title('Data Distribution by Month')
plt.xlabel('Month')

plt.subplot(2, 3, 5)
df['rainfall'].hist(bins=50, alpha=0.7, color='green')
plt.title('Rainfall Distribution')
plt.xlabel('Rainfall (mm)')

plt.subplot(2, 3, 6)
df['humidity'].hist(bins=50, alpha=0.7, color='orange')
plt.title('Average Humidity Distribution')
plt.xlabel('Humidity (%)')

plt.tight_layout()
plt.savefig('climate_analysis_plots.png', dpi=300, bbox_inches='tight')
plt.close()

print('Visualizations saved as climate_analysis_plots.png')

# Monthly analysis
monthly_avg = df.groupby('month')['avg_temperature'].mean()
monthly_heatwaves = df.groupby('month')['predicted_heatwave'].sum()
monthly_coldwaves = df.groupby('month')['predicted_coldwave'].sum()

print('\n=== MONTHLY ANALYSIS ===')
print('Average Temperature by Month:')
for month, temp in monthly_avg.items():
    print(f'Month {month}: {temp:.2f}°C')

print('\nHeat Wave Events by Month:')
for month, count in monthly_heatwaves.items():
    print(f'Month {month}: {count} events')

print('\nCold Wave Events by Month:')
for month, count in monthly_coldwaves.items():
    print(f'Month {month}: {count} events')

# Machine Learning Models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

feature_columns = ['max_temperature', 'min_temperature', 'humidity', 'rainfall', 
                  'wind_speed', 'pressure_surface_level', 'uv_index', 'cloud_cover',
                  'month', 'day', 'hour']

X = df[feature_columns].fillna(df[feature_columns].mean())
y_heatwave = df['predicted_heatwave']
y_coldwave = df['predicted_coldwave']

X_train, X_test, y_heat_train, y_heat_test = train_test_split(X, y_heatwave, test_size=0.2, random_state=42)
X_train, X_test, y_cold_train, y_cold_test = train_test_split(X, y_coldwave, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Heat Wave Model
rf_heatwave = RandomForestClassifier(n_estimators=100, random_state=42)
rf_heatwave.fit(X_train_scaled, y_heat_train)
y_heat_pred = rf_heatwave.predict(X_test_scaled)
heatwave_accuracy = accuracy_score(y_heat_test, y_heat_pred)

# Cold Wave Model
rf_coldwave = RandomForestClassifier(n_estimators=100, random_state=42)
rf_coldwave.fit(X_train_scaled, y_cold_train)
y_cold_pred = rf_coldwave.predict(X_test_scaled)
coldwave_accuracy = accuracy_score(y_cold_test, y_cold_pred)

print('\n=== MACHINE LEARNING RESULTS ===')
print(f'Heat Wave Prediction Accuracy: {heatwave_accuracy:.2%}')
print(f'Cold Wave Prediction Accuracy: {coldwave_accuracy:.2%}')

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_heatwave.feature_importances_
}).sort_values('importance', ascending=False)

print('\nTop 5 Features for Heat Wave Prediction:')
for idx, row in feature_importance.head().iterrows():
    print(f'{row["feature"]}: {row["importance"]:.4f}')


daily_data = df.groupby(df['date'].dt.date).agg({
    'max_temperature': 'max',
    'min_temperature': 'min',
    'avg_temperature': 'mean',
    'rainfall': 'sum',
    'humidity': 'mean',
    'wind_speed': 'mean',
    'pressure_surface_level': 'mean'
}).reset_index()

daily_data['date'] = pd.to_datetime(daily_data['date'])
daily_data = daily_data.sort_values('date')
daily_data.set_index('date', inplace=True)

print(f'\nDaily data shape: {daily_data.shape}')


from statsmodels.tsa.arima.model import ARIMA

temp_series = daily_data['avg_temperature'].dropna()
if len(temp_series) > 30:
    train_size = int(len(temp_series) * 0.8)
    train_data = temp_series[:train_size]
    
    model = ARIMA(train_data, order=(1, 1, 1))
    fitted_model = model.fit()
    
    forecast_steps = 30
    forecast = fitted_model.forecast(steps=forecast_steps)
    forecast_index = pd.date_range(start=temp_series.index[-1], periods=forecast_steps+1, freq='D')[1:]
    
    print('\n=== 30-DAY TEMPERATURE FORECAST ===')
    print('Next 30 days temperature predictions:')
    for i, (date, temp) in enumerate(zip(forecast_index, forecast)):
        print(f'Day {i+1} ({date.strftime("%Y-%m-%d")}): {temp:.2f}°C')
    
 
    high_heatwave_days = sum(1 for temp in forecast if temp >= heat_wave_threshold)
    high_coldwave_days = sum(1 for temp in forecast if temp <= cold_wave_threshold)
    
    print(f'\n=== RISK ASSESSMENT ===')
    print(f'High heat wave risk days: {high_heatwave_days}')
    print(f'High cold wave risk days: {high_coldwave_days}')
    print(f'Average forecasted temperature: {forecast.mean():.2f}°C')
    print(f'Temperature range: {forecast.min():.2f}°C to {forecast.max():.2f}°C')
    
    
    print(f'\n=== DETAILED RISK ASSESSMENT ===')
    for i, (date, temp) in enumerate(zip(forecast_index, forecast)):
        heat_risk = 'HIGH' if temp >= heat_wave_threshold else 'MEDIUM' if temp >= heat_wave_threshold * 0.9 else 'LOW'
        cold_risk = 'HIGH' if temp <= cold_wave_threshold else 'MEDIUM' if temp <= cold_wave_threshold * 1.1 else 'LOW'
        print(f'{date.strftime("%Y-%m-%d")}: {temp:.2f}°C | Heat Risk: {heat_risk} | Cold Risk: {cold_risk}')

print('\n=== PROJECT COMPLETED SUCCESSFULLY ===')
print('All analysis completed based on Provided dataset(Kaggle)!')
print('Check climate_analysis_plots.png for visualizations.')
