from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseRegressionModel(ABC):
    """
    Base class for regression models. All models should inherit from this class.
    It ensures that the `fit` and `predict` methods are implemented.
    """

    def __init__(self):
        self.model = None  # This will hold the model (e.g., sklearn, XGBoost)

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit the model to the training data.

        Parameters:
        - X (pd.DataFrame): Feature data.
        - y (pd.Series): Target data.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Parameters:
        - X (pd.DataFrame): Feature data.

        Returns:
        - np.ndarray: Predicted values.
        """
        pass

    def forecast_multiple_steps(self, last_known_data: pd.DataFrame, steps: int = 5) -> list:
        """
        Forecast multiple future time steps using a regression model and lag features.

        Parameters:
        - last_known_data (pd.DataFrame): DataFrame with the most recent lag feature rows (e.g., value_lag_1, value_lag_2).
        - steps (int): Number of future steps to forecast.

        Returns:
        - list: List of forecasted values.
        """
        forecast = []
        input_data = last_known_data.copy()

        for _ in range(steps):
            # Predict next value
            predicted_value = self.predict(input_data)[0]
            forecast.append(predicted_value)

            # Shift lag columns and insert new prediction
            new_row = input_data.iloc[0].copy()
            for i in range(len(new_row)-1, 0, -1):
                new_row[i] = new_row[i-1]
            new_row[0] = predicted_value  # Insert predicted value at lag_1

            input_data = pd.DataFrame([new_row], columns=input_data.columns)

        return forecast