from src.general.structures import *
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from sklearn.preprocessing import PowerTransformer
import numpy as np
import pandas as pd
from typing import List, Union, Optional

class StandardScalerTS(BaseTransformer):
    def __init__(self, with_mean: bool = True, with_std: bool = True):
        super().__init__()
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.scale_ = None
        self._sklearn_scaler = SklearnStandardScaler(
            with_mean=with_mean,
            with_std=with_std
        )

    def fit(self, ts_data: TSdata, y=None):
        if not isinstance(ts_data, TSdata):
            raise ValueError("Input must be a TSdata instance")

        if not ts_data.targets:
            raise ValueError("Targets not specified in TSdata object")

        target_data = ts_data.data[ts_data.targets]
        self._sklearn_scaler.fit(target_data)

        self.mean_ = self._sklearn_scaler.mean_
        self.scale_ = self._sklearn_scaler.scale_

        self._is_fitted = True
        return self

    def transform(self, ts_data: TSdata) -> TSdata:
        if not self._is_fitted:
            raise ValueError("StandardScaler must be fitted before transform")

        if not isinstance(ts_data, TSdata):
            raise ValueError("Input must be a TSdata instance")

        transformed_data = ts_data.data.copy()
        transformed_data[ts_data.targets] = self._sklearn_scaler.transform(transformed_data[ts_data.targets])

        transformed_ts = TSdata(transformed_data)
        transformed_ts.time = ts_data.time
        transformed_ts.targets = ts_data.targets
        transformed_ts.features = ts_data.features

        return transformed_ts

    def inverse_transform(self, ts_data: TSdata) -> TSdata:
        if not self._is_fitted:
            raise ValueError("StandardScaler must be fitted before inverse_transform")

        if not isinstance(ts_data, TSdata):
            raise ValueError("Input must be a TSdata instance")

        inverse_transformed_data = ts_data.data.copy()
        inverse_transformed_data[ts_data.targets] = self._sklearn_scaler.inverse_transform(
            inverse_transformed_data[ts_data.targets]
        )

        inverse_transformed_ts = TSdata(inverse_transformed_data)
        inverse_transformed_ts.time = ts_data.time
        inverse_transformed_ts.targets = ts_data.targets
        inverse_transformed_ts.features = ts_data.features

        return inverse_transformed_ts

    def fit_transform(self, ts_data: TSdata, y=None) -> TSdata:
        return self.fit(ts_data).transform(ts_data)


class PowerTransformerTS(BaseTransformer):
    """
    Transformer that applies power transformation to make data more Gaussian-like.
    Wraps sklearn.preprocessing.PowerTransformer for time series data.

    The power transform finds the optimal parameter λ to make data more Gaussian-like
    using either Yeo-Johnson or Box-Cox transformation.
    """

    def __init__(self, method: str = 'yeo-johnson', standardize: bool = False):
        """
        Initialize PowerTransformerTS

        Args:
            method: The power transform method.
                   Available options:
                   - 'yeo-johnson' [default]: works with both positive and negative values
                   - 'box-cox': only works with strictly positive values
            standardize: Whether to apply zero-mean, unit-variance normalization
                        to the transformed output (default=True)
        """
        super().__init__()
        if method not in ['yeo-johnson', 'box-cox']:
            raise ValueError("method must be either 'yeo-johnson' or 'box-cox'")

        self.method = method
        self.standardize = standardize
        self._transformer = None
        self.lambdas_ = None
        self.mean_ = None
        self.scale_ = None

    def fit(self, ts_data: TSdata, y=None):
        """
        Fit the power transformer.

        Args:
            ts_data: TSdata object containing the data to fit on

        Returns:
            self: The fitted transformer
        """
        if not ts_data.targets:
            raise ValueError("No targets specified in TSdata object")

        # Initialize the sklearn transformer
        self._transformer = PowerTransformer(
            method=self.method,
            standardize=self.standardize
        )

        # Fit transformer on target columns
        target_data = ts_data.data[ts_data.targets]

        # Check for non-positive values when using box-cox
        if self.method == 'box-cox':
            min_values = target_data.min()
            if (min_values <= 0).any():
                raise ValueError(
                    "Box-Cox transformation can only be applied to strictly positive values. "
                    f"Minimum values found: {min_values.to_dict()}"
                )

        try:
            self._transformer.fit(target_data)

            # Store the learned parameters
            self.lambdas_ = self._transformer.lambdas_
            if self.standardize:
                self.mean_ = self._transformer._scaler.mean_
                self.scale_ = self._transformer._scaler.scale_

            self._is_fitted = True
            return self

        except Exception as e:
            raise RuntimeError(f"Power transformation fitting failed: {str(e)}")

    def transform(self, ts_data: TSdata) -> TSdata:
        """
        Apply the power transformation to the data.

        Args:
            ts_data: TSdata object to transform

        Returns:
            Transformed TSdata object
        """
        self._check_if_fitted()

        transformed_data = ts_data.data.copy()

        try:
            # Transform only target columns
            transformed_data[ts_data.targets] = self._transformer.transform(
                transformed_data[ts_data.targets]
            )

        except Exception as e:
            raise RuntimeError(f"Power transformation failed: {str(e)}")

        transformed_ts = TSdata(transformed_data)
        transformed_ts.time = ts_data.time
        transformed_ts.targets = ts_data.targets
        transformed_ts.features = ts_data.features

        return transformed_ts

    def inverse_transform(self, ts_data: TSdata) -> TSdata:
        """
        Apply the inverse power transformation to transform data back to its original space.

        Args:
            ts_data: TSdata object to inverse transform

        Returns:
            Inverse transformed TSdata object
        """
        self._check_if_fitted()

        inverse_transformed_data = ts_data.data.copy()

        try:
            # Inverse transform only target columns
            inverse_transformed_data[ts_data.targets] = self._transformer.inverse_transform(
                inverse_transformed_data[ts_data.targets]
            )

        except Exception as e:
            raise RuntimeError(f"Inverse power transformation failed: {str(e)}")

        inverse_transformed_ts = TSdata(inverse_transformed_data)
        inverse_transformed_ts.time = ts_data.time
        inverse_transformed_ts.targets = ts_data.targets
        inverse_transformed_ts.features = ts_data.features

        return inverse_transformed_ts

    def fit_transform(self, ts_data: TSdata, y=None) -> TSdata:
        """
        Fit the transformer and apply the power transformation to the data.

        Args:
            ts_data: TSdata object to fit and transform

        Returns:
            Transformed TSdata object
        """
        return self.fit(ts_data).transform(ts_data)

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names.

        Args:
            input_features: Input feature names (ignored)

        Returns:
            List of transformed feature names
        """
        self._check_if_fitted()
        return self._transformer.get_feature_names_out()

