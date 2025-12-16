import streamlit as st
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

st.title(" Predictive Model App")

data = joblib.load("model.pkl")
model = data["model"]

TARGET_COL = data["target"]

ml_model = model.named_steps['model']
if hasattr(ml_model, 'predict_proba') or hasattr(ml_model, 'classes_'):
    PROBLEM = 'classification'
else:
    PROBLEM = 'regression'

st.sidebar.write(f"Model type detected: {ml_model.__class__.__name__}")
st.sidebar.write(f"Problem type: {PROBLEM}")


df = pd.read_csv("data.csv")
DROP_COLS = ["customer_id", TARGET_COL] 
input_cols = df.drop(columns=DROP_COLS, errors='ignore').columns

st.header("Input Prediction")
input = {}
for col in input_cols:
    if df[col].dtype in ['int64','float64']:
        input[col] = st.number_input(col, value=float(df[col].mean()))
    else:
        input[col] = st.selectbox(col, options=df[col].unique())

if st.button("Predict"):
    pred = model.predict(pd.DataFrame([input]))[0]
    if PROBLEM == "classification":
        st.success(f"Prediction (class): {pred}")
    else:
        st.success(f"Prediction (value): {pred:.2f}")