from models.regression_models.base_regression_model import BaseRegressionModel
import xgboost as xgb
import pandas as pd
import numpy as np

class XGBoostModel(BaseRegressionModel):
    def __init__(self, learning_rate: float = 0.1, n_estimators: int = 100):
        super().__init__()
        self.model = xgb.XGBRegressor(learning_rate=learning_rate, n_estimators=n_estimators)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)