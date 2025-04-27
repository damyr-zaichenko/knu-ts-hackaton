from __future__ import annotations
import pandas as pd
import numpy as np

class TimeSeriesCleaner:
    """
    Perform cleaning operations for time series data and track removed and filled data.
    """

    def __init__(self, data: pd.Series | pd.DataFrame):
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0]
        
        self.original_data = data.copy()
        self.data = data.copy()
        self.outliers_removed = 0
        self.missing_filled = 0

    def drop_duplicates(self) -> TimeSeriesCleaner:
        """
        Drop duplicate entries from the series or dataframe.
        """
        self.data = self.data.drop_duplicates()
        return self

    def fill_missing(self, strategy='mean') -> TimeSeriesCleaner:
        """
        Fill missing values in the series or dataframe.
        """
        missing_before = self.data.isna().sum()

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

        missing_after = self.data.isna().sum()
        self.missing_filled = missing_before - missing_after
        return self

    def remove_outliers_iqr(self, multiplier: float = 1.5) -> TimeSeriesCleaner:
        """
        Remove outliers using the IQR method and replace them with NaN.
        """
        q1 = self.data.quantile(0.25)
        q3 = self.data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        # Track outliers and replace with NaN
        outliers_before = self.data[(self.data < lower) | (self.data > upper)].count()
        self.outliers_removed = outliers_before

        self.data = self.data.apply(lambda x: np.nan if x < lower or x > upper else x)
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

    def get_removal_and_fill_stats(self):
        """
        Return stats about the data removed and filled.
        """
        return self.outliers_removed, self.missing_filled

# Usage example:
# raw_data = pd.read_csv('data/processed/test_processed.csv', index_col=0)
# series_cleaner = TimeSeriesCleaner(raw_data)

# # Clean series
# cleaned_data = (
#     series_cleaner
#     .remove_outliers_iqr(multiplier=1.5)  # Remove outliers and replace them with NaN
#     .fill_missing(strategy="mean")  # Fill NaNs with the mean
#     .get_df()  # or .get_series()
# )
