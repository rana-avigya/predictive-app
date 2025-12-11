import streamlit as st
import pandas as pd
import joblib

st.title("Predictive Model")

# CONFIG
PROBLEM = "classification"  # must match the trained model
DROP_COLS = ["track_id"]

# Load model
model = joblib.load("model.pkl")
df = pd.read_csv("data.csv")
input_cols = df.drop(columns=["genre"] + DROP_COLS, errors='ignore').columns

# --- Single input ---
single_input = {}
for col in input_cols:
    if df[col].dtype in ['int64','float64']:
        single_input[col] = st.number_input(col, value=float(df[col].mean()))
    else:
        single_input[col] = st.selectbox(col, options=df[col].unique())

if st.button("Predict"):
    pred = model.predict(pd.DataFrame([single_input]))[0]
    if PROBLEM == "classification":
        st.success(f"Prediction: {pred}")
    else:
        st.success(f"Prediction (value): {pred:.2f}")

