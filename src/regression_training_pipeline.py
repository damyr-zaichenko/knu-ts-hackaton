import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor  # ✅ Added LightGBM import
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocessing.regression_feature_engineering import RegressionFeatureEngineer
import numpy as np
from sklearn.neural_network import MLPRegressor

# -------------------------------
# Configs
# -------------------------------
DATA_PATH = 'data/processed/cleaned_train_set.csv'
N_ROWS = 9000000
TEST_SIZE_RATIO = 0.03
SHIFT_STEPS = 240
STEPS_DROP = 50

# Feature Engineering Parameters
LAGS = list(range(1, 1000, 20))
ROLL_WINDOWS = [5, 15, 30, 60, 90, 180, 360, 720]
ROLL_STATS = ['mean', 'std']
POLY_DEGREE = 2
EWM_ALPHAS = [0.0001, 0.001, 0.01, 0.1]

# -------------------------------
# Utility Functions
# -------------------------------
@st.cache_data
def load_data(path, n_rows):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.head(n_rows)

def split_data(df, test_size):
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx], df.iloc[split_idx:]

def shift_targets(df, columns, shift_steps):
    return df[columns].shift(-shift_steps)

def drop_na(X, y):
    X_cols = X.columns
    y_cols = y.columns

    X_renamed = X.add_suffix('_X')
    y_renamed = y.add_suffix('_y')

    combined = pd.concat([X_renamed, y_renamed], axis=1).dropna()

    X_clean = combined[X_renamed.columns].rename(columns=lambda col: col.replace('_X', ''))
    y_clean = combined[y_renamed.columns].rename(columns=lambda col: col.replace('_y', ''))

    return X_clean, y_clean


def calculate_average_metrics(metrics_list):
    """
    Calculates the average of each metric across all dictionaries in the list.

    Args:
        metrics_list (list of dict): List of dictionaries with the same metric keys.

    Returns:
        dict: Dictionary with the average of each metric.
    """
    avg_metrics = {}
    if not metrics_list:
        return avg_metrics

    metric_names = list(metrics_list[0].keys())

    for metric in metric_names:
        values = [d[metric] for d in metrics_list]
        avg_metrics[metric] = np.nanmean(values)  # Use nanmean to ignore possible NaNs

    return avg_metrics


def mean_absolute_percentage_error(y_true, y_pred, threshold=1e-2):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = np.abs(y_true) > threshold  # Only consider "non-near-zero" values
    if not np.any(mask):
        return np.nan  # If all y_true are tiny, return NaN
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# -------------------------------
# Classes
# -------------------------------

class DummyModel:
    def __init__(self):
        self.last_value = None

    def fit(self, X, y):
        self.cols = y.columns

    def predict(self, X):
        # Dummy model predicts the last value for all steps
        st.dataframe(X)
        return X[self.cols]

    def evaluate(self, X, y, dataset_name="Test", shift_steps=0):
        preds = self.predict(X)
        y = y.dropna()
        preds = preds[:len(y)]

        st.subheader(f"{dataset_name} Set: Predictions and Real Values")
        st.text(f"Predictions shape: {preds.shape}")
        st.dataframe(preds)
        st.text(f"Real values shape: {y.shape}")
        st.dataframe(y)

        results = {}
        for col_idx, col_name in enumerate(y.columns):
            y_true_col = y.iloc[:, col_idx]
            y_pred_col = preds[:, col_idx] if preds.ndim > 1 else preds

            results[col_name] = {
                'MAE': mean_absolute_error(y_true_col, y_pred_col),
                'MSE': mean_squared_error(y_true_col, y_pred_col),
                'R2': r2_score(y_true_col, y_pred_col),
                'MAPE': mean_absolute_percentage_error(y_true_col, y_pred_col)
            }

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(y_true_col.index, y_true_col.values, label='Real', color='blue')
            ax.plot(y_true_col.index, y_pred_col, label='Prediction', color='orange', linestyle='--')
            ax.set_title(f'{dataset_name} - Real vs Prediction: {col_name}')
            ax.set_xlabel('Index')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        return results

class FeatureEngineer:
    def __init__(self, lags, roll_windows, roll_stats, poly_degree, ewm_alphas):
        self.lags = lags
        self.roll_windows = roll_windows
        self.roll_stats = roll_stats
        self.poly_degree = poly_degree
        self.ewm_alphas = ewm_alphas

    def transform(self, df, columns):
        engineer = RegressionFeatureEngineer(df)
        df_features = (
            engineer
            .create_lag_features(columns=columns, lags=self.lags)
            .create_rolling_averages(columns=columns, windows=self.roll_windows, stats=self.roll_stats)
            .create_polynomial_features(columns=columns, degree=self.poly_degree)
            .create_exponential_smoothing(columns=columns, alphas=self.ewm_alphas)
            .get_engineered_data()
        )
        return df_features

class RegressionModel:
    def __init__(self, model):
        self.model = model

    def train(self, X, y):
        X, y = drop_na(X, y)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y, dataset_name="Test", shift_steps=0):
        preds = self.predict(X)
        y = y.dropna()
        preds = preds[:len(y)]

        st.subheader(f"{dataset_name} Set: Predictions and Real Values")
        st.text(f"Predictions shape: {preds.shape}")
        st.dataframe(preds)
        st.text(f"Real values shape: {y.shape}")
        st.dataframe(y)

        results = {}
        for col_idx, col_name in enumerate(y.columns):
            y_true_col = y.iloc[:, col_idx]
            y_pred_col = preds[:, col_idx] if preds.ndim > 1 else preds

        # for col_name in y.columns:
        #     y_true_col = y[col_name]
        #     y_pred_col = preds[col_name]

            results[col_name] = {
                'MAE': mean_absolute_error(y_true_col, y_pred_col),
                'MSE': mean_squared_error(y_true_col, y_pred_col),
                'R2': r2_score(y_true_col, y_pred_col),
                'MAPE': mean_absolute_percentage_error(y_true_col, y_pred_col)
            }

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(y_true_col.index, y_true_col.values, label='Real', color='blue')
            ax.plot(y_true_col.index, y_pred_col, label='Prediction', color='orange', linestyle='--')
            ax.set_title(f'{dataset_name} - Real vs Prediction: {col_name}')
            ax.set_xlabel('Index')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        return results
    

def filter_data(X, y, step=10):
    """
    Filters both X and y DataFrames/Series to keep one row every `step` rows.
    
    Parameters:
    - X: pandas DataFrame, features.
    - y: pandas DataFrame or Series, target.
    - step: int, the number of rows to skip. Default is 10.
    
    Returns:
    - Filtered X and y DataFrames/Series with one row every `step` rows.
    """
    X_filtered = X.iloc[::step]
    y_filtered = y.iloc[::step]
    
    return X_filtered, y_filtered


# -------------------------------
# Main Pipeline
# -------------------------------

def main():
    st.title("Regression Models for Time Series Prediction")

    # Model selection
    model_choice = st.selectbox(
        "Choose Regression Model:",
        ("Linear Regression", "Random Forest", "XGBoost", "LightGBM", "Neural Network"
         #"Dummy Model"
        )
    )

    if not st.button('Run'):
        return 0

    if model_choice == "Linear Regression":
        model = LinearRegression()
    elif model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_choice == "XGBoost":
        model = XGBRegressor(
            n_estimators=200,  # Increased to allow more boosting rounds with a lower learning rate
            learning_rate=0.05,  # Slightly higher learning rate to balance with n_estimators
            max_depth=5,  # Optimal depth for modeling interactions without overfitting
            min_child_weight=5,  # Ensures sufficient sample weight for each node
            subsample=0.8,  # Sampling ratio of data to prevent overfitting
            colsample_bytree=0.8,  # Randomly sample columns to prevent overfitting
            reg_alpha=0.1,  # L1 regularization term to prevent overfitting
            reg_lambda=1.0,  # L2 regularization term to avoid overly complex models
            gamma=0.1,  # Minimum loss reduction for making a further partition
            random_state=42,  # Ensures reproducibility
            tree_method="hist",  # Efficient histogram-based training
            objective='reg:squarederror'  # Standard objective for regression tasks
        )
    elif model_choice == "LightGBM":
        model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=42
        )
    elif model_choice == "Neural Network":  # ✅ Add Neural Network option here
        model = MLPRegressor(
            hidden_layer_sizes=(50,),  # Single hidden layer with 100 neurons
            activation='relu',  # ReLU activation function
            solver='adam',  # Adam optimizer
            max_iter=5,  # Maximum number of iterations
            random_state=42,  # Ensures reproducibility
            verbose=True
        )
    elif model_choice == "Dummy Model":
        model = DummyModel()  # Initialize the Dummy Model

    # Load data
    df = load_data(DATA_PATH, N_ROWS)
    initial_columns = df.columns

    # Split data
    train_df, test_df = split_data(df, TEST_SIZE_RATIO)

    # Feature engineering
    feature_engineer = FeatureEngineer(
        lags=LAGS,
        roll_windows=ROLL_WINDOWS,
        roll_stats=ROLL_STATS,
        poly_degree=POLY_DEGREE,
        ewm_alphas=EWM_ALPHAS
    )

    train_features = feature_engineer.transform(train_df, initial_columns)
    test_features = feature_engineer.transform(test_df, initial_columns)

    # Sort columns
    train_features = train_features[sorted(train_features.columns)]
    test_features = test_features[sorted(train_features.columns)]

    assert train_features.columns.equals(test_features.columns), "Feature columns mismatch!"

    # Prepare targets
    train_targets = shift_targets(train_df, initial_columns, SHIFT_STEPS)
    test_targets = shift_targets(test_df, initial_columns, SHIFT_STEPS)

    # Clean
    train_features_clean = train_features.dropna()
    train_targets_clean = train_targets.loc[train_features_clean.index]
    test_features_clean = test_features.dropna()
    test_targets_clean = test_targets.loc[test_features_clean.index]

    train_features_clean, train_targets_clean = filter_data(train_features_clean, train_targets_clean, STEPS_DROP)

    st.text(f"Train feature shape: {train_features_clean.shape}")
    st.text(f"Test feature shape: {test_features_clean.shape}")

    # Train
    reg_model = RegressionModel(model)
    reg_model.train(train_features_clean, train_targets_clean)

    # Evaluate
    train_metrics = reg_model.evaluate(train_features_clean, train_targets_clean, dataset_name="Train", shift_steps=SHIFT_STEPS)
    test_metrics = reg_model.evaluate(test_features_clean, test_targets_clean, dataset_name="Test", shift_steps=SHIFT_STEPS)

    col1, col2 = st.columns([1, 1])

    with col1:

        # Metrics
        st.subheader("Train Set Evaluation Metrics:")
        for target_col, metrics in train_metrics.items():
            st.write(f"**{target_col}**")
            st.json(metrics)

        st.subheader("Train Avg Metrics:")
        train_avg_metrics = calculate_average_metrics(list(train_metrics.values()))
        st.json(train_avg_metrics)

    with col2:

        st.subheader("Test Set Evaluation Metrics:")
        for target_col, metrics in test_metrics.items():
            st.write(f"**{target_col}**")
            st.json(metrics)

        st.subheader("Train Avg Metrics:")
        test_avg_metrics = calculate_average_metrics(list(test_metrics.values()))
        st.json(test_avg_metrics)

    # Plot
    series1_cols = [col for col in train_features.columns if ('Series1_' in col) or (col == 'Series1')]
    fig = px.line(train_features_clean[series1_cols], markers=False)
    st.plotly_chart(fig)


    if model_choice == "Random Forest":
        # Extract feature importances from the trained RandomForest model
        feature_importances = reg_model.model.feature_importances_

        # Create a DataFrame for the feature importances
        importance_df = pd.DataFrame({
            'Feature': train_features_clean.columns,
            'Importance': feature_importances
        })

        st.dataframe(importance_df)

        # Sort the DataFrame by importance
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Plot using Plotly for a more interactive chart
        fig = px.bar(importance_df, x='Feature', y='Importance', title="Feature Importances - Random Forest", 
                    labels={'Feature': 'Feature', 'Importance': 'Importance'})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig)

# Entry point
if __name__ == "__main__":
    main()