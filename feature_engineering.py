import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_features(df, target_col, drop_cols=None):
    
    if drop_cols is None:
        drop_cols = []
    drop_cols = list(drop_cols) + [target_col]
    
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df[target_col]
    
    numeric_cols = X.select_dtypes(include=['int64','float64']).columns
    categorical_cols = X.select_dtypes(include=['object','category']).columns
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])
    
    return X, y, preprocessor
