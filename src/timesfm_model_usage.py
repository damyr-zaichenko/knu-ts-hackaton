import pandas as pd
import streamlit as st
from model_loader import load_timesfm_model
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

# Define constants
DATA_PATH = 'data/processed/cleaned_train_set.csv'
N_ROWS = 1000  # Number of rows to load for analysis

# Function to load the cleaned dataset
@st.cache_data
def load_data(path, n_rows):
    """
    Loads the dataset from the specified path and returns the first `n_rows` rows.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.head(n_rows)

# Function to get the forecast output from the model
@st.cache_data
def get_forecast(_tfm, forecast_input):
    """
    Fetches the forecast from the trained model for the provided input data.
    """
    forecast_output = _tfm.forecast(forecast_input, freq=[0, 1])
    return forecast_output

# Function to plot the results
def plot_results(df, train, test, mean_forecast, quantile_forecast_2d, forecast_index):
    """
    Plots the input time series data, mean forecast, quantile forecasts, and test data.
    """
    # Create the plot
    fig = go.Figure()

    train = train[-10000:]
    df = df[-10000:]

    # Add the historical input data (Series1)
    fig.add_trace(go.Scatter(x=df.index, y=df.values, mode='lines', name='Input Data (Series1)', line=dict(color='blue')))

    # Add the train data (historical data used for training)
    fig.add_trace(go.Scatter(x=train.index, y=train.values, mode='lines', name='Train Data', line=dict(color='green', dash='dot')))

    # Add the test data (true future data)
    fig.add_trace(go.Scatter(x=test.index, y=test.values, mode='lines', name='Test Data', line=dict(color='red')))

    # Print to verify the content of mean_forecast
    st.code(mean_forecast)
    st.text(f"Mean forecast shape: {mean_forecast.shape}")

    # Convert mean_forecast to DataFrame if necessary
    if isinstance(mean_forecast, np.ndarray):
        mean_forecast = pd.DataFrame(mean_forecast.flatten(), columns=["Mean Forecast"], index=forecast_index)

    # Add the mean forecast to the plot
    fig.add_trace(go.Scatter(x=forecast_index, y=mean_forecast["Mean Forecast"], mode='lines', name='Mean Forecast', line=dict(color='white', dash='dash')))

    # Add the quantile forecasts (each quantile as a separate line)
    quantile_names = [f'Quantile {i+1}' for i in range(quantile_forecast_2d.shape[1])]
    for i in range(quantile_forecast_2d.shape[1]):
        continue
        fig.add_trace(go.Scatter(x=forecast_index, y=quantile_forecast_2d[:, i], mode='lines', name=quantile_names[i], line=dict(dash='dot')))

    # Update layout for better readability
    fig.update_layout(
        title="Time Series Forecast with Quantiles",
        xaxis_title="Date",
        yaxis_title="Value",
        legend_title="Legend",
        template="plotly_dark"
    )

    # Display the plot
    st.plotly_chart(fig)

# Main function to orchestrate the script flow
def main():
    # Load the trained model
    tfm = load_timesfm_model()

    print(help(tfm))

    # Load and prepare the data
    df = load_data(DATA_PATH, n_rows=N_ROWS)
    df = df['Series3']  # Assume 'Series1' is the column to be forecasted

    # Split data into train and test
    train_size = int(0.9 * len(df))  # 80% for training
    train, test = df[:train_size], df[train_size:]

    # Prepare the input data for forecasting (train data)
    forecast_input = [train.values.astype(np.float32)]

    # Fit the model on training data
    tfm._model.fit(train.values.astype(np.float32))

    # Get the forecast results
    mean_forecast, quantile_forecast = get_forecast(tfm, forecast_input)
    quantile_forecast_2d = np.squeeze(quantile_forecast)

    # Generate the forecast time index (starting from the last timestamp of train data)
    forecast_index = pd.date_range(train.index[-1] + pd.Timedelta(minutes=1), periods=mean_forecast.shape[1], freq='T')

    # Display the forecast data
    st.write("Mean forecast:")
    st.dataframe(pd.DataFrame(mean_forecast))

    st.write("Quantile forecast:")
    st.dataframe(pd.DataFrame(quantile_forecast_2d))

    # Plot the results
    plot_results(df, train, test, mean_forecast, quantile_forecast_2d, forecast_index)

# Run the main function when the script is executed
if __name__ == "__main__":
    main()