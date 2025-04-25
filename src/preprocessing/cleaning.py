from __future__ import annotations
import pandas as pd
import numpy as np

class TimeSeriesCleaner:
    """
    Perform cleaning operations for time series data.
    """

    def __init__(self, data: pd.Series | pd.DataFrame):

        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0] 
        
        self.data = data.copy()

    def drop_duplicates(self) -> TimeSeriesCleaner:
        """
        Drop duplicate entries from the series or dataframe.
        """
        self.data = self.data.drop_duplicates()
        return self

    def fill_missing(self, strategy='mean') -> TimeSeriesCleaner:
        """
        Fill missing values in the series or dataframe.

        Supported strategies: 'mean', 'median', 'mode', 'zero', 'ffill', 'bfill'.
        """
        if not self.data.isna().any():
            return self

        if strategy == 'mean':
            self.data = self.data.fillna(self.data.mean())
        elif strategy == 'median':
            self.data = self.data.fillna(self.data.median())
        elif strategy == 'mode':
            self.data = self.data.fillna(self.data.mode()[0])
        elif strategy == 'zero':
            self.data = self.data.fillna(0)
        elif strategy == 'ffill':
            self.data = self.data.ffill()
        elif strategy == 'bfill':
            self.data = self.data.bfill()

        return self

    def remove_outliers_iqr(self, multiplier: float = 1.5) -> TimeSeriesCleaner:
        """
        Remove outliers using the IQR method.
        """
        q1 = self.data.quantile(0.25)
        q3 = self.data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        self.data = self.data[(self.data >= lower) & (self.data <= upper)]
        return self

    def get_df(self) -> pd.DataFrame:
        """
        Return the cleaned data as a DataFrame.
        """
        return pd.DataFrame(self.data)

    def get_series(self) -> pd.Series:
        """
        Return the cleaned data as a Series.
        """
        return self.data


# Usage example:
# raw_data = pd.read_csv('data/processed/test_processed.csv', index_col=0)
# series_cleaner = TimeSeriesCleaner(raw_data)

# Clean series
# cleaned_data = (
#     series_cleaner
#     .fill_missing(strategy="median")
#     .remove_outliers_iqr(multiplier=1.5)
#     .get_df()  # or .get_series()
# )