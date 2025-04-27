import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Load the data
df = pd.read_csv('data/raw/train_set.csv', index_col=0, parse_dates=True)

st.title("Time Series Plots")

# Plot each column separately
for column in df.columns:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df.index, y=df[column], mode='lines', name=column)
    )
    fig.update_layout(
        title=column,
        xaxis_title='Date',
        yaxis_title=column,
        height=400,
        width=900
    )
    st.plotly_chart(fig)