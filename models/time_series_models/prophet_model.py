from prophet import Prophet
from .base_time_series_model import BaseTimeSeriesModel
import pandas as pd
import numpy as np

class ProphetModel(BaseTimeSeriesModel):
    def __init__(self):
        self.model = Prophet()
        self.last_date = None

    def fit(self, series: pd.Series):
        """
        Fit the model to a univariate time series.
        Converts the series to Prophet format internally.
        """
        # Ensure the series is 1-dimensional
        if isinstance(series, pd.DataFrame):
            if series.shape[1] == 1:
                series = series.iloc[:, 0]
            else:
                raise ValueError("The DataFrame must contain only one column of data.")
        
        if not isinstance(series, pd.Series):
            raise ValueError("The input must be a pandas Series.")
        
        # Check if the series index is a DatetimeIndex
        if not isinstance(series.index, pd.DatetimeIndex):
            series.index = pd.to_datetime(series.index)
        
        # Ensure the series is numeric
        if not np.issubdtype(series.dtype, np.number):
            raise ValueError("The series values must be numerical.")
        
        # Convert the series to a DataFrame with columns 'ds' and 'y'
        df = pd.DataFrame({
            'ds': series.index,  
            'y': series.values 
        })

        self.last_date = series.index[-1]

        import streamlit as st
        st.dataframe(df)

        self.model.fit(df)

    def forecast(self, steps: int) -> pd.Series:
        """
        Forecast a number of future steps.
        Automatically generates the future dataframe.
        """
        future_dates = pd.date_range(start=self.last_date + pd.Timedelta(days=1), periods=steps)
        future_df = pd.DataFrame({'ds': future_dates})
        forecast = self.model.predict(future_df)
        return forecast.set_index('ds')['yhat']