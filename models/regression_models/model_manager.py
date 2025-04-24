from models.regression_models.base_regression_model import BaseRegressionModel
import pandas as pd


class ModelManager:
    """
    A manager class that handles multiple models, allowing for training and prediction.
    """

    def __init__(self, models: list[BaseRegressionModel]):
        self.models = {model.__class__.__name__: model for model in models}

    def fit_models(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit all the models in the manager.

        Parameters:
        - X (pd.DataFrame): Feature data.
        - y (pd.Series): Target data.
        """
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            model.fit(X, y)

    def predict_models(self, X: pd.DataFrame) -> dict:
        """
        Predict using all models in the manager.

        Parameters:
        - X (pd.DataFrame): Feature data.

        Returns:
        - dict: A dictionary of predictions, where keys are model names and values are the predictions.
        """
        predictions = {}
        for model_name, model in self.models.items():
            predictions[model_name] = model.predict(X)
        return predictions