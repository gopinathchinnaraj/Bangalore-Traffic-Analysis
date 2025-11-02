# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Load CSV Data
df = pd.read_csv('Banglore_traffic_Dataset.csv')

# Convert Date to datetime and extract useful time features
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Weekday'] = df['Date'].dt.weekday
df['DayOfYear'] = df['Date'].dt.dayofyear

# Data Cleaning: Handle missing values
df.fillna(method='ffill', inplace=True)  # Forward fill

# Detect outliers using Z-score method
def detect_outliers_zscore(data, threshold=3):
    z_scores = np.abs((data - data.mean()) / data.std())
    return z_scores > threshold

# Remove outliers in the traffic volume data
df['Outlier'] = detect_outliers_zscore(df['Traffic Volume'])
df = df[df['Outlier'] == False]

# Drop the Outlier column after filtering
df = df.drop(columns=['Outlier'])

# Data Normalization: Scaling numeric features
scaler = StandardScaler()
numeric_features = ['Traffic Volume', 'Average Speed', 'Travel Time Index', 'Congestion Level', 
                    'Road Capacity Utilization', 'Public Transport Usage', 'Incident Reports']
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Visualizations: Traffic Volume Over Time
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Date', y='Traffic Volume', label='Traffic Volume')
plt.title('Traffic Volume Over Time')
plt.xticks(rotation=45)
plt.show()

# Visualize the Traffic Volume with Heatmap
plt.figure(figsize=(12, 8))
traffic_pivot = df.pivot_table(index='Month', columns='Year', values='Traffic Volume', aggfunc='mean')
sns.heatmap(traffic_pivot, cmap='coolwarm', annot=True)
plt.title('Average Monthly Traffic Volume by Year')
plt.show()

# Congestion Level Distribution with Kernel Density Estimation (KDE)
plt.figure(figsize=(8, 5))
sns.histplot(df['Congestion Level'], kde=True)
plt.title('Congestion Level Distribution')
plt.show()

# Pairplot: Visualizing relationships between multiple traffic-related variables
sns.pairplot(df[['Traffic Volume', 'Congestion Level', 'Road Capacity Utilization', 'Average Speed']], diag_kind='kde')
plt.show()

# Analyze Weather's Impact on Traffic Volume with a bar chart
weather_impact = df.groupby('Weather Conditions')['Traffic Volume'].mean().sort_values()
plt.figure(figsize=(10,6))
weather_impact.plot(kind='bar')
plt.title('Average Traffic Volume by Weather Conditions')
plt.xticks(rotation=45)
plt.show()

# 3D Scatter Plot using Plotly (Interactive Visualization)
fig = px.scatter_3d(df, x='Average Speed', y='Congestion Level', z='Traffic Volume',
                    color='Weather Conditions', title='3D Traffic Visualization')
fig.show()

# Time Series Forecasting: Predicting Future Traffic Volume with ARIMA

# Setting 'Date' as the index for time series forecasting
df.set_index('Date', inplace=True)

# Use only traffic volume data for ARIMA model
traffic_series = df['Traffic Volume'].resample('M').mean()  # Monthly resampling

# Train ARIMA model (auto_regressive, differencing, moving average)
model_arima = ARIMA(traffic_series, order=(1,1,1))  # p=1, d=1, q=1
model_arima_fit = model_arima.fit()

# Predict the next 12 months of traffic volume
future_forecast = model_arima_fit.forecast(steps=12)
print("Next 12-month forecast:")
print(future_forecast)

# Plot the forecast
plt.figure(figsize=(10,6))
plt.plot(traffic_series.index, traffic_series, label='Actual')
plt.plot(pd.date_range(traffic_series.index[-1], periods=12, freq='M'), future_forecast, label='Forecast')
plt.title('Traffic Volume Forecast for Next 12 Months')
plt.legend()
plt.show()

# Data Preparation for Machine Learning Models

# Define X (features) and y (target)
X = df[['Year', 'Month', 'Day', 'Average Speed', 'Travel Time Index', 'Congestion Level', 
        'Road Capacity Utilization', 'Public Transport Usage', 'Incident Reports']]
y = df['Traffic Volume']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training: Linear Regression
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Predict Traffic Volume using Linear Regression
y_pred_lr = model_lr.predict(X_test)

# Evaluate Linear Regression model
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Linear Regression MSE: {mse_lr}")
print(f"Linear Regression R²: {r2_lr}")

# Model Training: Decision Tree
model_dt = DecisionTreeRegressor(random_state=42)
model_dt.fit(X_train, y_train)

# Predict Traffic Volume using Decision Tree
y_pred_dt = model_dt.predict(X_test)

# Evaluate Decision Tree model
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print(f"Decision Tree MSE: {mse_dt}")
print(f"Decision Tree R²: {r2_dt}")

# Model Training: Random Forest
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Predict Traffic Volume using Random Forest
y_pred_rf = model_rf.predict(X_test)

# Evaluate Random Forest model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest MSE: {mse_rf}")
print(f"Random Forest R²: {r2_rf}")

# Model Comparison
print("Model Comparison:")
print(f"Linear Regression: MSE={mse_lr}, R²={r2_lr}")
print(f"Decision Tree: MSE={mse_dt}, R²={r2_dt}")
print(f"Random Forest: MSE={mse_rf}, R²={r2_rf}")

# Save the best model to disk using joblib
best_model = model_rf if r2_rf > r2_lr and r2_rf > r2_dt else model_lr if r2_lr > r2_dt else model_dt
joblib.dump(best_model, 'best_traffic_volume_model.pkl')

# Predict Traffic Volume for a future date using the best model
future_date = pd.DataFrame({
    'Year': [2024],
    'Month': [10],
    'Day': [10],
    'Average Speed': [40],
    'Travel Time Index': [1.1],
    'Congestion Level': [3],
    'Road Capacity Utilization': [0.80],
    'Incident Reports': [1],
    'Public Transport Usage': [250]
})

# Load the best model and make future predictions
X = df[['Year', 'Month', 'Day', 'Average Speed', 'Travel Time Index', 'Congestion Level',
        'Road Capacity Utilization', 'Public Transport Usage', 'Incident Reports']]

# Ensure future_date has the same columns as X used in training
future_date = pd.DataFrame({
    'Year': [2024],
    'Month': [10],
    'Day': [10],
    'Average Speed': [40],
    'Travel Time Index': [1.1],
    'Congestion Level': [3],
    'Road Capacity Utilization': [0.80],
    'Public Transport Usage': [250],
    'Incident Reports': [1]
}, columns=['Year', 'Month', 'Day', 'Average Speed', 'Travel Time Index',
            'Congestion Level', 'Road Capacity Utilization',
            'Public Transport Usage', 'Incident Reports'])


# Load the best model and make future predictions
best_model_loaded = joblib.load('best_traffic_volume_model.pkl')
future_traffic_volume = best_model_loaded.predict(future_date)
print(f"Predicted Traffic Volume for 2024-10-10: {future_traffic_volume[0]}")


#############
best_model_loaded = joblib.load('best_traffic_volume_model.pkl')
future_traffic_volume = best_model_loaded.predict(future_date)
print(f"Predicted Traffic Volume for 2024-10-10: {future_traffic_volume[0]}")

# Save busiest areas to CSV
busiest_areas = df.groupby('Area Name')['Traffic Volume'].sum().sort_values(ascending=False)
busiest_areas.to_csv('busiest_areas.csv')
