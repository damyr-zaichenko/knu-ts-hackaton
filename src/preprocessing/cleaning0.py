import numpy as np
import pandas as pd
from typing import List, Union, Optional
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from sklearn.impute import SimpleImputer
import warnings
from sklearn.preprocessing import RobustScaler as SklearnRobustScaler
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import SplineTransformer
from scipy import stats
from scipy import interpolate
import scikit_posthocs as sp
from src.general.structures import *








class IQRTransformer(BaseTransformer):
    """
    Transformer that handles outliers using Interquartile Range (IQR) method.
    Values beyond Q1 - k*IQR and Q3 + k*IQR are considered outliers.
    """

    def __init__(self, k: float = 1.5, strategy: str = 'clip'):
        """
        Initialize IQRTransformer

        Args:
            k: Factor to multiply IQR by for determining outlier boundaries (default 1.5)
            strategy: How to handle outliers - 'clip' or 'remove' (default 'clip')
        """

        if strategy not in ['clip', 'remove', 'nan']:
            raise ValueError("Strategy must be either 'clip', 'remove' or 'nan'")

        self.k = k
        self.strategy = strategy
        self.boundaries = {}
        self._is_fitted = False

    def fit(self, ts_data: TSdata):
        """
        Calculate IQR boundaries for each target variable, ignoring NaN values

        Args:
            ts_data: TSdata object containing the data to fit on
        """
        if not ts_data.targets:
            raise ValueError("No targets specified in TSdata object")

        self.boundaries = {}
        for target in ts_data.targets:
            # Use only non-NaN values for calculating boundaries
            data = ts_data.data[target].dropna().values

            if len(data) == 0:
                raise ValueError(f"Target {target} contains only NaN values")

            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1

            lower_bound = Q1 - self.k * IQR
            upper_bound = Q3 + self.k * IQR

            self.boundaries[target] = {
                'lower': lower_bound,
                'upper': upper_bound,
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR
            }

        print(f"Calculated IQR boundaries: {self.boundaries}")

        self._is_fitted = True
        return self

    def transform(self, ts_data: TSdata) -> TSdata:
        """
        Transform data by handling outliers according to the chosen strategy.
        NaN values are preserved in the output.

        Args:
            ts_data: TSdata object to transform

        Returns:
            Transformed TSdata object
        """
        if not self._is_fitted:
            raise ValueError("IQRTransformer must be fitted before transform")

        transformed_data = ts_data.data.copy()

        for target in ts_data.targets:
            bounds = self.boundaries[target]
            if self.strategy == 'clip':
                # Create mask for non-NaN values
                non_nan_mask = ~transformed_data[target].isna()


                # Apply clipping only to non-NaN values
                transformed_data.loc[non_nan_mask, target] = \
                    transformed_data.loc[non_nan_mask, target].clip(
                        lower=bounds['lower'],
                        upper=bounds['upper']
                    )
            elif self.strategy == 'remove':  # remove strategy
                # Keep rows where value is either NaN or within bounds
                mask = transformed_data[target].isna() | \
                       ((transformed_data[target] >= bounds['lower']) & \
                        (transformed_data[target] <= bounds['upper']))
                transformed_data = transformed_data[mask]
            elif self.strategy == 'nan':
                # Set outliers to NaN
                print(f"Setting outliers to NaN for {target}")

                transformed_data.loc[transformed_data[target] < bounds['lower'], target] = np.nan
                transformed_data.loc[transformed_data[target] > bounds['upper'], target] = np.nan

        transformed_ts = TSdata(transformed_data)
        transformed_ts.time = ts_data.time
        transformed_ts.targets = ts_data.targets
        transformed_ts.features = ts_data.features

        return transformed_ts

    def fit_transform(self, ts_data: TSdata) -> TSdata:
        """
        Fit and transform the data in one step
        """
        return self.fit(ts_data).transform(ts_data)

    def get_outliers_mask(self, ts_data: TSdata) -> dict:
        """
        Get boolean mask of outliers for each target.
        NaN values are marked as False in the mask.

        Args:
            ts_data: TSdata object to analyze

        Returns:
            Dictionary with target names as keys and boolean arrays as values,
            where True indicates an outlier (NaN values are False)
        """
        if not self._is_fitted:
            raise ValueError("IQRTransformer must be fitted before getting outliers mask")

        outliers_mask = {}
        for target in ts_data.targets:
            bounds = self.boundaries[target]
            # Only mark non-NaN values that are outside bounds as outliers
            mask = ~ts_data.data[target].isna() & \
                   ((ts_data.data[target] < bounds['lower']) | \
                    (ts_data.data[target] > bounds['upper']))
            outliers_mask[target] = mask

        return outliers_mask


class GrubbsOutlierTransformerTS(BaseTransformer):
    """
    Transformer that handles outliers using Grubbs test for normally distributed data.
    First checks for normality using Shapiro-Wilk test, then applies Grubbs test
    for outlier detection if data is normal.
    """

    def __init__(self, alpha: float = 0.05, strategy: str = 'remove'):
        """
        Initialize GrubbsOutlierTransformer

        Args:
            alpha: Significance level for both Shapiro-Wilk and Grubbs tests (default 0.05)
            strategy: How to handle outliers - 'remove' or 'nan' (default 'remove')
        """
        if strategy not in ['remove', 'nan']:
            raise ValueError("Strategy must be either 'remove' or 'nan'")

        self.alpha = alpha
        self.strategy = strategy
        self.test_results = {}
        self._is_fitted = False

    def _check_normality(self, data: np.ndarray) -> tuple:
        """
        Perform Shapiro-Wilk test for normality

        Returns:
            Tuple of (is_normal, statistic, p_value)
        """
        if len(data) < 3:
            return False, np.nan, np.nan

        non_nan_data = data[~np.isnan(data)]
        if len(non_nan_data) < 3:
            return False, np.nan, np.nan

        statistic, p_value = stats.shapiro(non_nan_data)
        is_normal = p_value > self.alpha

        return is_normal, statistic, p_value

    def fit(self, ts_data: TSdata):
        """
        Check normality for each target and identify outliers using Grubbs test
        if data is normally distributed

        Args:
            ts_data: TSdata object containing the data to fit on
        """
        if not ts_data.targets:
            raise ValueError("No targets specified in TSdata object")

        self.test_results = {}

        for target in ts_data.targets:
            data = ts_data.data[target].values
            is_normal, shapiro_stat, shapiro_p = self._check_normality(data)

            result = {
                'is_normal': is_normal,
                'shapiro_statistic': shapiro_stat,
                'shapiro_p_value': shapiro_p,
                'outliers_indices': None
            }

            if is_normal:
                try:
                    # Get outlier indices using Grubbs test
                    non_nan_data = data[~np.isnan(data)]
                    outliers = sp.outliers_grubbs(non_nan_data, alpha=self.alpha)
                    # Map back to original indices
                    original_indices = np.where(~np.isnan(data))[0]
                    outlier_indices = original_indices[outliers]
                    result['outliers_indices'] = outlier_indices
                except Exception as e:
                    result['error'] = str(e)

            self.test_results[target] = result

        self._is_fitted = True
        return self

    def transform(self, ts_data: TSdata) -> TSdata:
        """
        Transform data by handling outliers according to the chosen strategy.
        Only transforms normally distributed variables where Grubbs test was applied.

        Args:
            ts_data: TSdata object to transform

        Returns:
            Transformed TSdata object
        """
        if not self._is_fitted:
            raise ValueError("Transformer must be fitted before transform")

        transformed_data = ts_data.data.copy()

        for target in ts_data.targets:
            result = self.test_results[target]
            print(f"Grubbs test result for {target}: {result}")
            if not result['is_normal'] or result.get('error') or result['outliers_indices'] is None:
                continue

            if self.strategy == 'nan':
                transformed_data.loc[result['outliers_indices'], target] = np.nan
            else:  # remove strategy
                transformed_data = transformed_data.drop(index=result['outliers_indices'])

        transformed_ts = TSdata(transformed_data)
        transformed_ts.time = ts_data.time
        transformed_ts.targets = ts_data.targets
        transformed_ts.features = ts_data.features

        return transformed_ts

    def fit_transform(self, ts_data: TSdata) -> TSdata:
        """
        Fit and transform the data in one step
        """
        return self.fit(ts_data).transform(ts_data)


class SimpleImputerTS(BaseTransformer):
    """
    Transformer that handles missing values in time series data using various strategies.
    """

    def __init__(self, strategy: str = 'mean', fill_value: Optional[Union[str, int, float]] = None):
        """
        Initialize SimpleImputer

        Args:
            strategy: Strategy to use for imputation.
                     Options: 'mean', 'median', 'most_frequent', 'constant', 'forward_fill', 'backward_fill'
            fill_value: Value to use when strategy is 'constant'. Ignored for other strategies.
        """
        super().__init__()
        valid_strategies = ['mean', 'median', 'most_frequent', 'constant', 'forward_fill', 'backward_fill']
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}")

        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = {}

    def fit(self, ts_data: TSdata, y=None):
        """
        Calculate statistics for imputation based on the chosen strategy

        Args:
            ts_data: TSdata object containing the data to fit on
        """
        if not ts_data.targets:
            raise ValueError("No targets specified in TSdata object")

        self.statistics_ = {}

        for target in ts_data.targets:
            data = ts_data.data[target]

            if self.strategy == 'mean':
                self.statistics_[target] = data.mean()
            elif self.strategy == 'median':
                self.statistics_[target] = data.median()
            elif self.strategy == 'most_frequent':
                self.statistics_[target] = data.mode()[0]
            elif self.strategy == 'constant':
                if self.fill_value is None:
                    raise ValueError("fill_value must be specified when strategy is 'constant'")
                self.statistics_[target] = self.fill_value
            elif self.strategy in ['forward_fill', 'backward_fill']:
                # No statistics needed for forward/backward fill
                self.statistics_[target] = None

        self._is_fitted = True
        return self

    def transform(self, ts_data: TSdata) -> TSdata:
        """
        Transform data by imputing missing values according to the chosen strategy.

        Args:
            ts_data: TSdata object to transform

        Returns:
            Transformed TSdata object with imputed values
        """
        self._check_if_fitted()

        transformed_data = ts_data.data.copy()

        for target in ts_data.targets:
            if self.strategy in ['mean', 'median', 'most_frequent', 'constant']:
                transformed_data[target] = transformed_data[target].fillna(self.statistics_[target])
            elif self.strategy == 'forward_fill':
                transformed_data[target] = transformed_data[target].ffill()
            elif self.strategy == 'backward_fill':
                transformed_data[target] = transformed_data[target].bfill()

        transformed_ts = TSdata(transformed_data)
        transformed_ts.time = ts_data.time
        transformed_ts.targets = ts_data.targets
        transformed_ts.features = ts_data.features

        return transformed_ts

    def fit_transform(self, ts_data: TSdata, y=None) -> TSdata:
        """
        Fit and transform the data in one step
        """
        return self.fit(ts_data).transform(ts_data)


class SplineInterpolationImputer(BaseTransformer):
    """
    Transformer that handles missing values in time series data using spline interpolation.
    Uses scipy's interpolate.splrep and splev for spline interpolation.
    """

    def __init__(self, order: int = 3, max_gap: Optional[int] = None):
        """
        Initialize SplineInterpolationImputer

        Args:
            order: The order of the spline interpolation (default is 3 for cubic splines)
                  Must be 1 <= order <= 5
            max_gap: Maximum gap size to interpolate. Gaps larger than this will remain NaN.
                    If None, all gaps will be interpolated regardless of size.
        """
        super().__init__()
        if not 1 <= order <= 5:
            raise ValueError("Spline order must be between 1 and 5")

        self.order = order
        self.max_gap = max_gap
        self.interpolators_ = {}

    def _interpolate_series(self, series: pd.Series) -> pd.Series:
        """
        Interpolate a single series using spline interpolation.

        Args:
            series: pandas Series containing the data to interpolate

        Returns:
            Interpolated series
        """
        # Get non-NaN indices and values
        non_nan_mask = ~series.isna()
        x = np.where(non_nan_mask)[0]
        y = series[non_nan_mask].values

        # Need at least (order + 1) points for interpolation
        if len(x) <= self.order:
            warnings.warn(f"Not enough non-NaN points for order {self.order} interpolation. "
                          f"Returning original series.")
            return series

        # Get indices where we need to interpolate
        nan_indices = np.where(series.isna())[0]

        if len(nan_indices) == 0:
            return series

        if self.max_gap is not None:
            # Find gaps that are too large
            gaps = np.split(nan_indices, np.where(np.diff(nan_indices) > 1)[0] + 1)
            valid_indices = []
            for gap in gaps:
                if len(gap) <= self.max_gap:
                    valid_indices.extend(gap)
            nan_indices = np.array(valid_indices)

        if len(nan_indices) == 0:
            return series

        try:
            # Fit spline
            tck = interpolate.splrep(x, y, k=self.order)

            # Interpolate at missing points
            interpolated_values = interpolate.splev(nan_indices, tck)

            # Create new series with interpolated values
            result = series.copy()
            result.iloc[nan_indices] = interpolated_values

            return result

        except Exception as e:
            warnings.warn(f"Spline interpolation failed: {str(e)}. Returning original series.")
            return series

    def fit(self, ts_data: TSdata, y=None):
        """
        Fit the transformer (no actual fitting needed for interpolation)

        Args:
            ts_data: TSdata object containing the data to fit on
        """
        if not ts_data.targets:
            raise ValueError("No targets specified in TSdata object")

        if ts_data.time is None:
            raise ValueError("Time column must be specified for spline interpolation")

        self._is_fitted = True
        return self

    def transform(self, ts_data: TSdata) -> TSdata:
        """
        Transform data by interpolating missing values using splines.

        Args:
            ts_data: TSdata object to transform

        Returns:
            Transformed TSdata object with interpolated values
        """
        self._check_if_fitted()

        transformed_data = ts_data.data.copy()

        for target in ts_data.targets:
            transformed_data[target] = self._interpolate_series(transformed_data[target])

        transformed_ts = TSdata(transformed_data)
        transformed_ts.time = ts_data.time
        transformed_ts.targets = ts_data.targets
        transformed_ts.features = ts_data.features

        return transformed_ts

    def fit_transform(self, ts_data: TSdata, y=None) -> TSdata:
        """
        Fit and transform the data in one step
        """
        return self.fit(ts_data).transform(ts_data)


