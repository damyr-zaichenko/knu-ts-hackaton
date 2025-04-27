import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class RegressionFeatureEngineer:
    """
    Handle feature engineering tasks for time series data, including
    lag features, rolling statistics, and polynomial transformations
    for regression tasks.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the RegressionFeatureEngineer class.
        """
        self.df = df.copy()

    def create_lag_features(self, columns: list[str], lags: list[int]):
        """
        Creates lag features for the specified columns.
        """
        lagged_features = []
        for col in columns:
            for lag in lags:
                lagged = self.df[col].shift(lag).rename(f'{col}_lag_{lag}')
                lagged_features.append(lagged)
        
        lagged_df = pd.concat(lagged_features, axis=1)
        self.df = pd.concat([self.df, lagged_df], axis=1)
        return self

    def create_rolling_averages(self, columns: list[str], windows: list[int], stats: list[str] = ['mean']):
        """
        Creates rolling statistics for the specified columns.
        """
        rolling_features = []
        for col in columns:
            for window in windows:
                roll = self.df[col].rolling(window)
                for stat in stats:
                    feature = getattr(roll, stat)().rename(f'{col}_roll_{stat}_{window}')
                    rolling_features.append(feature)
        
        rolling_df = pd.concat(rolling_features, axis=1)
        self.df = pd.concat([self.df, rolling_df], axis=1)
        return self

    def create_polynomial_features(self, columns: list[str], degree: int = 2, interaction_only: bool = False):
        """
        Generates polynomial features from the specified columns.
        """
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
        data = self.df[columns]
        poly_features = poly.fit_transform(data)
        feature_names = poly.get_feature_names_out(columns)
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=self.df.index)

        self.df = pd.concat([self.df.drop(columns, axis=1), poly_df], axis=1)
        return self

    def create_exponential_smoothing(self, columns: list[str], alphas: list[float]):
        """
        Creates exponentially smoothed features for the specified columns.
        """
        ewm_features = []
        for col in columns:
            for alpha in alphas:
                smoothed = self.df[col].ewm(alpha=alpha, adjust=False).mean().rename(f'{col}_ewm_alpha_{alpha}')
                ewm_features.append(smoothed)

        ewm_df = pd.concat(ewm_features, axis=1)
        self.df = pd.concat([self.df, ewm_df], axis=1)
        return self

    def get_engineered_data(self) -> pd.DataFrame:
        """
        Returns the DataFrame with engineered features.
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