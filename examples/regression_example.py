import pandas as pd
from sklearn.datasets import make_regression

from models.regression_models.linear_regression_model import LinearRegressionModel
from models.regression_models.random_forest_model import RandomForestModel
from models.regression_models.xgboost_regression_model import XGBoostModel
from models.regression_models.model_manager import ModelManager

# ==========================
#   Simple Regression
# ==========================

# Generate example regression dataset
X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

# Initialize models
linear_model = LinearRegressionModel()
rf_model = RandomForestModel()
xgb_model = XGBoostModel()

# Instantiate ModelManager with a list of models
manager = ModelManager([linear_model, rf_model, xgb_model])

# Fit all models
manager.fit_models(X, y)

# Predict with all models
predictions = manager.predict_models(X)

# Print predictions for each model
for model_name, pred in predictions.items():
    print(f"{model_name} predictions: {pred[:5]}")  # Display first 5 predictions



# ==========================
#   Time Series Forecasting 
# ==========================

from src.preprocessing.cleaning import TimeSeriesCleaner
from src.preprocessing.regression_feature_engineering import RegressionFeatureEngineer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import streamlit as st
import os 

raw_df = pd.read_csv('data/processed/test_processed.csv', index_col=0)
raw_df.columns = ['value']

series_cleaner = TimeSeriesCleaner(raw_df)

cleaned_df = (
    series_cleaner
    .drop_duplicates()
    .fill_missing()
    .remove_outliers_iqr()
    .get_df()
)

feature_engineer = RegressionFeatureEngineer(cleaned_df)

df_features = (
    feature_engineer
    .create_lag_features(columns=['value'], lags=[1, 2, 3])
    .create_polynomial_features(columns=['value'], degree=2, interaction_only=True)
    .create_rolling_averages(columns=['value'], windows=[1, 2, 3], stats=['mean'])
    .get_engineered_data()
)

st.text('X:')
st.text(df_features.shape)
st.dataframe(df_features)

st.text('y')
cleaned_df['target'] = cleaned_df['value'].shift(-1)
y = cleaned_df['target']
st.text(y.shape)
st.dataframe(y)


def drop_na(X, y):
    # Combine into one DataFrame to drop any rows with NA jointly
    combined = pd.concat([X, y], axis=1)
    combined = combined.dropna()

    # Separate back into X and y
    X_clean = combined.iloc[:, :-1]
    y_clean = combined.iloc[:, -1]

    return X_clean, y_clean


X, y = drop_na(df_features, y)

manager.fit_models(X, y)
st.text('Models fit')

def evaluate_models(manager, X, y):
    predictions = manager.predict_models(X)
    results = {}

    for model_name, pred in predictions.items():
        r2 = r2_score(y, pred)
        rmse = np.sqrt(mean_squared_error(y, pred))
        mae = mean_absolute_error(y, pred)

        results[model_name] = {
            "R²": r2,
            "RMSE": rmse,
            "MAE": mae
        }

    return results

st.dataframe(evaluate_models(manager, X, y))