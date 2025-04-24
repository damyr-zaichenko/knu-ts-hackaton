import streamlit as st
import pandas as pd
import numpy as np

df = pd.read_csv('data/raw/test_data.csv', index_col=0)

st.dataframe(df)

np.random.seed(42)

outlier_indices = np.random.choice(df.index, size=5, replace=False)
df.loc[outlier_indices, 'x'] = df['x'].mean() * 1000

missing_indices = np.random.choice(df.index, size=5, replace=False)
df.loc[missing_indices, 'x'] = np.nan

values = df['x'].values
date_range = pd.date_range(start='2025-01-01', periods=len(values), freq='D')

series = pd.Series(values, index=date_range)

st.line_chart(series)

st.dataframe(series)

series.to_csv('data/processed/test_processed.csv')