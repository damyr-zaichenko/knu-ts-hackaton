import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class RegressionFeatureEngineer:
    """
    Handles feature engineering tasks for time series data, including
    lag features, rolling statistics, and polynomial transformations
    for regression tasks.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the RegressionFeatureEngineer.

        Parameters:
        - df (pd.DataFrame): The input time series DataFrame.
        """
        self.df = df.copy()

    def create_lag_features(self, columns: list[str], lags: list[int]):
        """
        Create lag features for specified columns and lags.
        """
        new_cols = {
            f'{col}_lag_{lag}': self.df[col].shift(lag)
            for col in columns
            for lag in lags
        }
        self.df = self.df.assign(**new_cols)
        return self

    def create_rolling_averages(self, columns: list[str], windows: list[int], stats: list[str] = ['mean']):
        """
        Create rolling window statistics (mean, std, etc.) for specified columns.
        """
        new_cols = {}
        for col in columns:
            for window in windows:
                roll = self.df[col].rolling(window)
                for stat in stats:
                    func = getattr(roll, stat)
                    new_cols[f'{col}_roll_{stat}_{window}'] = func()
        self.df = self.df.assign(**new_cols)
        return self

    def create_polynomial_features(self, columns: list[str], degree: int = 2, interaction_only: bool = False):
        """
        Generate polynomial and interaction features from specified columns.
        """
        data = self.df[columns]
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
        poly_features = poly.fit_transform(data)
        feature_names = poly.get_feature_names_out(columns)

        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=self.df.index)

        # Drop original columns and add transformed features
        self.df = self.df.drop(columns=columns)
        self.df = pd.concat([self.df, poly_df], axis=1)
        return self

    def create_exponential_smoothing(self, columns: list[str], alphas: list[float]):
        """
        Create exponentially smoothed versions of specified columns.
        """
        new_cols = {
            f'{col}_ewm_alpha_{alpha}': self.df[col].ewm(alpha=alpha, adjust=False).mean()
            for col in columns
            for alpha in alphas
        }
        self.df = self.df.assign(**new_cols)
        return self

    def get_engineered_data(self) -> pd.DataFrame:
        """
        Return the DataFrame with engineered features.
        """
        return self.df


# # Example usage:

# df = pd.read_csv('data/processed/test_processed.csv', index_col=0)

# from cleaning import TimeSeriesCleaner

# cleaner = TimeSeriesCleaner(df)

# # Clean the series
# cleaned_series = (
#     cleaner
#     .drop_duplicates()
#     .fill_missing()
#     .remove_outliers_iqr()
#     .get_df()
# )

# cleaned_series.columns = ['value']

# # Feature engineering
# feature_engineer = (
#     RegressionFeatureEngineer(cleaned_series)
#     .create_lag_features(columns=['value'], lags=[1, 2, 3])
#     .create_rolling_averages(columns=['value'], windows=[3, 7], stats=['mean', 'std'])
#     .create_polynomial_features(columns=['value'], degree=2)
# )

# df_features = feature_engineer.get_engineered_data()