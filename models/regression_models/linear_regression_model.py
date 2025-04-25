from sklearn.linear_model import LinearRegression
from models.regression_models.base_regression_model import BaseRegressionModel
import pandas as pd
import numpy as np

class LinearRegressionModel(BaseRegressionModel):
    def __init__(self):
        super().__init__()
        self.model = LinearRegression()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)