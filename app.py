import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

#using streamlit to create new page title layout for page
st.set_page_config(page_title = "Predictive App", layout="wide")
st.title(" Predictive Model App")
#loads the model generated in train.py using joblib
data = joblib.load("model.pkl")
model = data["model"]

target_column = data["target"]

ml_model = model.named_steps['model']
if hasattr(ml_model, 'predict_proba') or hasattr(ml_model, 'classes_'):
    PROBLEM = 'classification'
else:
    PROBLEM = 'regression'

df = pd.read_csv("data.csv")

df = df.drop(columns=["post_id","upload_date"])
DROP_COLS = ["customer_id", target_column]
input_cols = df.drop(columns=DROP_COLS, errors='ignore').columns

tabs = st.tabs(["EDA", "Predictive"])

with tabs[0]:
    st.header("Exploratory Data Analysis")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Column datatypes")
    st.write(df.dtypes)

    st.subheader("Summary")
    st.write(df.describe())
    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())

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


    st.subheader("Target vs Feature")
    target_col = st.selectbox("Select Target Column", df.columns, key="target_eda")
    feature_col = st.selectbox("Select Feature Column", [c for c in df.columns if c != target_col], key="feature_eda")
    
    fig, ax = plt.subplots()
    if df[target_col].dtype in ['int64', 'float64']: 
        if df[feature_col].dtype in ['int64', 'float64']:
            sns.scatterplot(x=df[feature_col], y=df[target_col], ax=ax)
        else:
            sns.boxplot(x=df[feature_col], y=df[target_col], ax=ax)
    else:  
        if df[feature_col].dtype in ['int64', 'float64']:
                sns.boxplot(x=df[target_col], y=df[feature_col], ax=ax)
        else:
            sns.countplot(x=feature_col, hue=target_col, data=df, palette="Set1", ax=ax)
    st.pyplot(fig)

with tabs[1]:
    st.header(f"Predict {target_column} using {ml_model.__class__.__name__}")
    input = {}
    for col in input_cols:
        if df[col].dtype in ['int64','float64']:
            input[col] = st.number_input(col, value=float(df[col].mean()))
        else:
            input[col] = st.selectbox(col, options=df[col].unique())

    if st.button("Predict"):
        pred = model.predict(pd.DataFrame([input]))[0]
        if PROBLEM == "classification":
            st.success(f"Prediction of {target_column}: {pred}")
        else:
            st.success(f"Prediction of {target_column}: {pred:.2f}")