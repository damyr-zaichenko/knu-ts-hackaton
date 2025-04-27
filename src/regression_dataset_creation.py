import pandas as pd
import streamlit as st
import plotly.express as px
from preprocessing.regression_feature_engineering import RegressionFeatureEngineer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# -------------------------------
# Parameters
# -------------------------------
DATA_PATH = 'data/processed/cleaned_train_set.csv'
N_ROWS = 50000
TEST_SIZE_RATIO = 0.05
SHIFT_STEPS = 240

LAGS = list(range(1, 60, 2)) + list(range(60, 200, 10))  # finer for small lags, coarser for large
ROLL_WINDOWS = [30, 60, 90, 120, 180, 240]  # added more options in between
ROLL_STATS = ['mean', 'std', 'max', 'min', 'median']  # expanded types of rolling stats
POLY_DEGREES = [1, 2, 3]  # allow trying a simpler (1) or slightly more complex (3) degree
EWM_ALPHAS = [0.001, 0.005, 0.01, 0.05]  # slightly expanded for more smoothing options

# -------------------------------
# Functions
# -------------------------------
@st.cache_data
def load_data(path, n_rows):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.head(n_rows)

def split_data(df, test_size):
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test

def engineer_features(df, columns):
    engineer = RegressionFeatureEngineer(df)
    df_features = (
        engineer
        .create_lag_features(columns=columns, lags=LAGS)
        .create_rolling_averages(columns=columns, windows=ROLL_WINDOWS, stats=ROLL_STATS)
        .create_polynomial_features(columns=columns, degree=POLY_DEGREE)
        .create_exponential_smoothing(columns=columns, alphas=EWM_ALPHAS)
        .get_engineered_data()
    )
    return df_features

def shift_targets(df, columns, shift_steps):
    return df[columns].shift(-shift_steps)

def drop_na(X, y):
    X_cols = X.columns
    y_cols = y.columns
    
    X_renamed = X.add_suffix('_X')
    y_renamed = y.add_suffix('_y')
    
    combined = pd.concat([X_renamed, y_renamed], axis=1)
    combined = combined.dropna()

    X_clean = combined[X_renamed.columns].rename(columns=lambda col: col.replace('_X', ''))
    y_clean = combined[y_renamed.columns].rename(columns=lambda col: col.replace('_y', ''))

    return X_clean, y_clean

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    X_train, y_train = drop_na(X_train, y_train)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X, y, dataset_name="Test"):
    preds = model.predict(X)[:-SHIFT_STEPS]
    y = y.dropna()

    st.subheader(f"{dataset_name} Set: Predictions and Real Values")
    st.text(f"Predictions shape: {preds.shape}")
    st.text(f"Real values shape: {y.shape}")

    results = {}

    for col_idx, col_name in enumerate(y.columns):
        y_true_col = y.iloc[:, col_idx]
        y_pred_col = preds[:, col_idx] if preds.ndim > 1 else preds

        # Metrics
        results[col_name] = {
            'MAE': mean_absolute_error(y_true_col, y_pred_col),
            'MSE': mean_squared_error(y_true_col, y_pred_col),
            'R2': r2_score(y_true_col, y_pred_col)
        }

        # Plot real vs prediction
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

# -------------------------------
# Main Pipeline
# -------------------------------

# Load data
df = load_data(DATA_PATH, N_ROWS)

# Initial Columns
initial_columns = df.columns

# Train/Test Split
train_df, test_df = split_data(df, TEST_SIZE_RATIO)

# Feature Engineering
train_features = engineer_features(train_df, initial_columns)
test_features = engineer_features(test_df, initial_columns)

# Sort columns for consistency
sorted_train_features = train_features[sorted(train_features.columns)]
sorted_test_features = test_features[sorted(train_features.columns)]  # Force same order

assert sorted_train_features.columns.equals(sorted_test_features.columns), "Feature columns mismatch!"

# Target Values (y)
train_targets = shift_targets(train_df, initial_columns, SHIFT_STEPS)
test_targets = shift_targets(test_df, initial_columns, SHIFT_STEPS)

# Align and clean
train_features_clean = sorted_train_features.dropna()
train_targets_clean = train_targets.loc[train_features_clean.index]
test_features_clean = sorted_test_features.dropna()
test_targets_clean = test_targets.loc[test_features_clean.index]

# Debug Outputs
st.text(f"Train feature shape: {train_features_clean.shape}")
st.text(f"Test feature shape: {test_features_clean.shape}")

# Train Model
model = train_linear_regression(train_features_clean, train_targets_clean)

# Evaluate Model on Train Set
train_metrics = evaluate_model(model, train_features_clean, train_targets_clean, dataset_name="Train")

# Evaluate Model on Test Set
test_metrics = evaluate_model(model, test_features_clean, test_targets_clean, dataset_name="Test")

# Metrics Summaries
st.subheader("Train Set Evaluation Metrics:")
for target_col, metrics in train_metrics.items():
    st.write(f"**{target_col}**")
    st.json(metrics)

st.subheader("Test Set Evaluation Metrics:")
for target_col, metrics in test_metrics.items():
    st.write(f"**{target_col}**")
    st.json(metrics)

# Plot an Example Series
series1_cols = [col for col in sorted_train_features.columns if ('Series1_' in col) or (col == 'Series1')]
fig = px.line(train_features_clean[series1_cols], markers=False)
st.plotly_chart(fig)

