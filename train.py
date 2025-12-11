import argparse

from pathlib import Path
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

def column_types(df, exclude=None):
    if exclude is None:
        exclude = []
    numeric = df.select_dtypes(include=['number']).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric and c not in exclude]
    return numeric, categorical

def build_pipeline(numeric_features, categorical_features, problem = 'classification'):
    numeric_transformer = Pipeline(steps=[
        ('imputer',SimpleImputer(strategy='median')),
        ('scaler',StandardScaler())
    ])

    categorical_transformer = Pipeline(steps = [
        ('imputer',SimpleImputer(   strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore',sparse=False))
        
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat',categorical_transformer, categorical_features)
    ])

    if problem == 'classification':
        model = RandomForestClassifier(n_estimators= 150, random_state=42, n_jobs =-1)
    else:
        model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    return pipeline

def main():
    df = pd.read_csv('shopping_behavior_updated.csv')
    X = df.drop(columns)