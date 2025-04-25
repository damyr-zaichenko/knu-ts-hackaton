import numpy as np
import pandas as pd
from typing import List, Union, Optional

class TSdata:
    """
    Class to handle time series data.
    """

    def __init__(self, data: pd.DataFrame, time = None, targets: Optional[List[str]] = None, features: Optional[List[str]] = None):
        self.data = data
        self.time: str = time
        self.targets: List[str] = targets
        self.features: List[str] = features

class BaseTransformer:
    """
    Base class for all time series transformers.
    Implements basic functionality and interface that all transformers should follow.
    """

    def __init__(self):
        """
        Initialize BaseTransformer.
        Sets the initial fitted state to False.
        """
        self._is_fitted = False

    def fit(self, ts_data: TSdata, y=None):
        """
        Fit the transformer to the data.
        Must be implemented by child classes.

        Args:
            ts_data: TSdata object to fit the transformer on
            y: Optional target variable (kept for scikit-learn compatibility)

        Returns:
            self: The fitted transformer

        Raises:
            NotImplementedError: If not implemented by child class
        """
        raise NotImplementedError("Subclasses must implement fit method")

    def transform(self, ts_data: TSdata) -> TSdata:
        """
        Transform the data.
        Must be implemented by child classes.

        Args:
            ts_data: TSdata object to transform

        Returns:
            TSdata: Transformed data

        Raises:
            NotImplementedError: If not implemented by child class
        """
        if not self._is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        raise NotImplementedError("Subclasses must implement transform method")

    def fit_transform(self, ts_data: TSdata, y=None) -> TSdata:
        """
        Fit the transformer and transform the data in one step.

        Args:
            ts_data: TSdata object to fit and transform
            y: Optional target variable (kept for scikit-learn compatibility)

        Returns:
            TSdata: Transformed data
        """
        return self.fit(ts_data, y).transform(ts_data)

    def _check_if_fitted(self):
        """
        Check if the transformer has been fitted.

        Raises:
            ValueError: If the transformer has not been fitted
        """
        if not self._is_fitted:
            raise ValueError("This transformer is not fitted yet. Call 'fit' before using this method.")

    def _validate_input(self, ts_data: TSdata):
        """
        Validate the input TSdata object.

        Args:
            ts_data: TSdata object to validate

        Raises:
            ValueError: If input is not a TSdata instance or if required attributes are missing
        """
        if not isinstance(ts_data, TSdata):
            raise ValueError("Input must be a TSdata instance")

        if not ts_data.targets:
            raise ValueError("No targets specified in TSdata object")

        if ts_data.data is None or ts_data.data.empty:
            raise ValueError("TSdata object contains no data")

    def _validate_features(self, ts_data: TSdata, required_features: List[str]):
        """
        Validate that required features exist in the data.

        Args:
            ts_data: TSdata object to validate
            required_features: List of feature names that must be present

        Raises:
            ValueError: If any required feature is missing
        """
        missing_features = [f for f in required_features if f not in ts_data.data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

    def inverse_transform(self, ts_data: TSdata) -> TSdata:
        """
        Inverse transform the data.
        Must be implemented by child classes.

        Args:
            ts_data: TSdata object to inverse transform

        Returns:
            TSdata: Inverse transformed data

        Raises:
            NotImplementedError: If not implemented by child class
        """
        if not self._is_fitted:
            raise ValueError("Transformer must be fitted before inverse_transform")
        return ts_data


class Pipeline(BaseTransformer):
    """
    A pipeline of transformers that can be applied sequentially to time series data.
    Implements the same interface as BaseTransformer.
    """

    def __init__(self, steps: List[tuple[str, BaseTransformer]]):
        """
        Initialize Pipeline with a list of (name, transformer) tuples.

        Args:
            steps: List of tuples where each tuple contains:
                  - name (str): A string identifier for the transformer
                  - transformer (BaseTransformer): A transformer instance
        """
        super().__init__()

        if not steps:
            raise ValueError("Steps cannot be empty")

        # Validate steps format
        self._validate_steps(steps)

        self.steps = steps
        self.named_steps = dict(steps)

    def _validate_steps(self, steps):
        """
        Validate the steps parameter.

        Args:
            steps: List of (name, transformer) tuples to validate

        Raises:
            ValueError: If steps format is invalid
        """
        names, transformers = zip(*steps)

        # Check for duplicate names
        if len(set(names)) != len(names):
            raise ValueError("Duplicate transformer names are not allowed")

        # Check that all transformers inherit from BaseTransformer
        for name, transformer in steps:
            if not isinstance(transformer, BaseTransformer):
                raise ValueError(
                    f"Transformer {name} must be an instance of BaseTransformer"
                )

    def fit(self, ts_data: TSdata, y=None):
        """
        Fit all transformers in the pipeline sequentially.

        Args:
            ts_data: TSdata object to fit on
            y: Ignored. Kept for compatibility.

        Returns:
            self: The fitted pipeline
        """
        data = ts_data

        for name, transformer in self.steps:
            try:
                data = transformer.fit_transform(data)
            except Exception as e:
                raise RuntimeError(f"Error fitting transformer {name}: {str(e)}")

        self._is_fitted = True
        return self

    def transform(self, ts_data: TSdata) -> TSdata:
        """
        Apply all transformations in the pipeline sequentially.

        Args:
            ts_data: TSdata object to transform

        Returns:
            TSdata: Transformed data

        Raises:
            ValueError: If pipeline is not fitted
        """
        self._check_if_fitted()

        data = ts_data

        for name, transformer in self.steps:
            try:
                data = transformer.transform(data)
            except Exception as e:
                raise RuntimeError(f"Error in transformer {name}: {str(e)}")

        return data

    def fit_transform(self, ts_data: TSdata, y=None) -> TSdata:
        """
        Fit and transform the data through all steps in the pipeline.

        Args:
            ts_data: TSdata object to fit and transform
            y: Ignored. Kept for compatibility.

        Returns:
            TSdata: Transformed data
        """
        return self.fit(ts_data).transform(ts_data)

    def inverse_transform(self, ts_data: TSdata) -> TSdata:
        """
        Inverse transform the data through all steps in the pipeline.

        Args:
            ts_data: TSdata object to inverse transform

        Returns:
            TSdata: Inverse transformed data
        """
        self._check_if_fitted()

        data = ts_data

        for name, transformer in reversed(self.steps):
            try:
                data = transformer.inverse_transform(data)
            except Exception as e:
                raise RuntimeError(f"Error in transformer {name}: {str(e)}")

        return data


