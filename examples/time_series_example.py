import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from models.time_series_models.arima_model import ARIMAModel
from models.time_series_models.prophet_model import ProphetModel
from models.time_series_models.model_manager import TimeSeriesModelManager
#from models.time_series_models.lstm_model import LSTMTimeSeriesModel  # Import the LSTM model

from src.preprocessing.cleaning import TimeSeriesCleaner
import streamlit as st

raw_df = pd.read_csv('data/processed/test_processed.csv', index_col=0)
raw_df.columns = ['value']

series_cleaner = TimeSeriesCleaner(raw_df)

cleaned_df = (
    series_cleaner
    .drop_duplicates()
    .fill_missing()
    .remove_outliers_iqr()
    .get_df()
)

series = cleaned_df

st.dataframe(series)

# Initialize models
arima = ARIMAModel(order=(5, 1, 0))
prophet = ProphetModel()
#lstm = LSTMTimeSeriesModel(look_back=10, epochs=20)  # LSTM model initialization

# Fit models
arima.fit(series)
prophet.fit(series)
#lstm.fit(series)  # Fit the LSTM model

# Forecast future
arima_forecast = arima.forecast(steps=10)
prophet_forecast = prophet.forecast(steps=10)
#lstm_forecast = lstm.forecast(steps=10)  # Forecast using the LSTM model

# Using model manager
manager = TimeSeriesModelManager(models={
    'ARIMA': arima,
    'Prophet': prophet,
    #'LSTM': lstm  # Add LSTM to the manager
})

manager_forecasts = manager.forecast_all(steps=10)

# Output results
print("ARIMA Forecast:", arima_forecast)
print("Prophet Forecast:", prophet_forecast.values[:10])
#print("LSTM Forecast:", lstm_forecast)  # Print the LSTM forecast
print("Manager Forecasts:", manager_forecasts)


def plot_predictions(series, forecasts, model_names, forecast_steps, last_n_steps=500):
    """
    Visualizes the original time series along with predictions from multiple models.
    
    Parameters:
    - series: The original time series data (pandas Series).
    - forecasts: A dictionary of model names and their respective forecasted values.
    - model_names: List of model names to plot.
    - forecast_steps: Number of forecasted steps for each model.
    - last_n_steps: Number of last steps of the series to display.
    """
    
    series.index = pd.to_datetime(series.index)
    
    # Slice the series to get only the last `n` steps
    series_to_plot = series.tail(last_n_steps)
    
    plt.figure(figsize=(12, 6))
    
    # Plot the last `n` steps of the original series
    plt.plot(series_to_plot.index, series_to_plot.values, label='Original Data', color='black', linewidth=2)
    
    # Plot forecasts for each model
    for model_name in model_names:
        forecast = forecasts[model_name]
        
        last_date = pd.to_datetime(series_to_plot.index[-1])
        
        forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps)
        
        # Plot forecasted values
        plt.plot(forecast_index, forecast, label=f'{model_name} Forecast', linestyle='--', marker='o')
    
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(f'Time Series Predictions - Last {last_n_steps} steps')
    plt.legend()
    
    # Show plot
    plt.tight_layout()
    st.pyplot(plt)


# Prepare forecasts for visualization
forecasts = {
    'ARIMA': arima_forecast,
    'Prophet': prophet_forecast.values[:10],  # Adjust if necessary
    #'LSTM': lstm_forecast  # Add LSTM forecast
}

# List of models for plotting
model_names = ['ARIMA', 'Prophet']  # Include LSTM in the list of models

# Visualize the predictions
plot_predictions(series, forecasts, model_names, forecast_steps=10)