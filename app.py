import streamlit as st
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

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

st.header("Exploratory Data Analysis")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Column datatypes")
st.write(df.dtypes)

st.subheader("Summary")
st.write(df.describe())

st.subheader("Feature distribution")
select_col = st.selectbox("Select column" ,df.columns)
fig, ax = plt.subplots()

if df[select_col].dtype in ["int64","float64"]:
    sns.histplot(df[select_col], kde=True, ax=ax)
    ax.set_ylabel(select_col)
    ax.set_xlabel("Count")
    ax.set_title(f"Distribution of {select_col}")
else:
    sns.countplot(df[select_col])
    ax.set_ylabel(select_col)
    ax.set_xlabel("Count")
    ax.set_title(f"Distribution of {select_col}")
st.pyplot(fig)

numeric_cols = df.select_dtypes(include=["int64", "float64"])

if len(numeric_cols.columns) > 1:
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

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
        st.success(f"Prediction of {TARGET_COL}: {pred}")
    else:
        st.success(f"Prediction of {TARGET_COL}: {pred:.2f}")