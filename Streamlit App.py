import sys
sys.path.append(r"C:\Program Files\Python313\Lib\site-packages")

import joblib  # Try importing again

import streamlit as st
import pandas as pd

# Load model
model = joblib.load("xgboost_production_forecast.pkl")

st.title("Reservoir Production Forecasting")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your reservoir data (CSV)", type="csv")

if uploaded_file:
    new_data = pd.read_csv(uploaded_file)
    predictions = model.predict(new_data)
    new_data["Predicted_Production"] = predictions
    st.write(new_data)
