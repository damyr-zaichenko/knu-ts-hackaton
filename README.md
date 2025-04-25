# ⏱️ Time Series Forecasting Hackathon Project

Welcome to our project for the KNU TS Hackathon 2025! This repo contains our work on time series forecasting using Python and ML.

## 📁 Structure

- `notebooks/` — Jupyter notebooks for EDA and modeling experiments  
- `examples/` — Python scripts demonstrating usage of models and preprocessing steps  
- `src/`  
  - `preprocessing/` — Cleaning functions and regression feature engineering utilities  
- `models/`  
  - `regression_models/` — Modular regression models (Linear, Random Forest, XGBoost) and manager class  
- `data/` — Input data files (keep it lightweight or use symlinks)  
- `requirements.txt` — Python dependency list

## 🚀 Setup

```bash
git clone https://github.com/damyr-zaichenko/knu-ts-hackaton.git
cd knu-ts-hackathon
python3.10 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows
pip install -r requirements.txt