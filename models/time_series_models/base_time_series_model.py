import pandas as pd
import numpy as np

class BaseTimeSeriesModel:
    def __init__(self):
        pass

    def fit(self, series: pd.Series):
        """Fit the model to a univariate time series."""
        raise NotImplementedError

    def forecast(self, steps: int) -> np.ndarray:
        """Forecast a number of future steps."""
        raise NotImplementedError