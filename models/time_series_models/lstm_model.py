import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from .base_time_series_model import BaseTimeSeriesModel


class LSTMTimeSeriesModel(BaseTimeSeriesModel):
    def __init__(self, look_back=10, batch_size=32, epochs=50):
        """
        Initialize the LSTM model.

        Parameters:
        - look_back: Number of previous time steps to use as input features.
        - batch_size: Batch size for model training.
        - epochs: Number of training epochs.
        """
        super().__init__()
        self.look_back = look_back
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.scaler = None

    def _prepare_data(self, series: pd.Series):
        """
        Prepares the time series data for LSTM by scaling and reshaping it.
        """

        values = series.values.reshape(-1, 1)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_values = self.scaler.fit_transform(values)

        # Convert series to supervised learning 
        X, y = [], []
        for i in range(self.look_back, len(scaled_values)):
            X.append(scaled_values[i - self.look_back:i, 0])
            y.append(scaled_values[i, 0])

        X, y = np.array(X), np.array(y)

        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y

    def fit(self, series: pd.Series):
        """
        Fit the LSTM model to the univariate time series.
        
        Parameters:
        - series: The time series data as a pandas Series.
        """
        # Prepare the data
        X, y = self._prepare_data(series)

        # Build the LSTM model
        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=False, input_shape=(X.shape[1], 1)))
        self.model.add(Dense(units=1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.series = series

        # Train the model
        self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs, verbose=1)

    def forecast(self, steps: int) -> np.ndarray:
        """
        Forecast a number of future steps using the trained LSTM model.

        Parameters:
        - steps: Number of future time steps to forecast.
        
        Returns:
        - forecast: Forecasted values as a numpy array.
        """
        if not self.model:
            raise ValueError("The model is not trained. Please call the 'fit' method first.")

        # Start forecasting from the last observed value
        last_values = self.scaler.transform(self.series[-self.look_back:].values.reshape(-1, 1))
        forecast = []

        for _ in range(steps):
            # Reshape the input to match LSTM input format
            last_values = np.reshape(last_values, (1, self.look_back, 1))
            
            # Predict the next value
            predicted_value = self.model.predict(last_values)
            
            # Inverse transform to get the forecast in the original scale
            predicted_value = self.scaler.inverse_transform(predicted_value)
            forecast.append(predicted_value[0, 0])
            
            # Update the last_values array to include the newly predicted value
            last_values = np.append(last_values[:, 1:, :], predicted_value.reshape(1, 1, 1), axis=1)

        return np.array(forecast)

# # Example usage
# if __name__ == "__main__":
#     # Example time series data (univariate)
#     data = pd.Series([100, 102, 105, 107, 109, 111, 115, 118, 120, 122, 123, 125, 127, 130, 132])

#     # Instantiate the LSTM model
#     lstm_model = LSTMTimeSeriesModel(look_back=5, epochs=20)

#     # Fit the model
#     lstm_model.fit(data)

#     # Forecast the next 5 time steps
#     forecast = lstm_model.forecast(steps=5)
#     print("Forecasted values:", forecast)