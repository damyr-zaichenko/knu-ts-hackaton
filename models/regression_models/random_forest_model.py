from sklearn.ensemble import RandomForestRegressor
from models.regression_models.base_regression_model import BaseRegressionModel
import pandas as pd
import numpy as np

class RandomForestModel(BaseRegressionModel):
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        super().__init__()
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)