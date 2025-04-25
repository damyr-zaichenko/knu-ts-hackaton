from statsmodels.tsa.arima.model import ARIMA
from .base_time_series_model import BaseTimeSeriesModel

class ARIMAModel(BaseTimeSeriesModel):
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None
        self.fitted_model = None

    def fit(self, series):
        self.model = ARIMA(series, order=self.order)
        self.fitted_model = self.model.fit()

    def forecast(self, steps):
        return self.fitted_model.forecast(steps=steps)