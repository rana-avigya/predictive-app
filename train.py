# import pandas as pd
# import argparse
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
# import joblib
# import math
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
#
# #asks user for input for type of pronblem and what algorithm to use
# user_input = argparse.ArgumentParser(description="Train a predictive model")
# user_input.add_argument('--target', type=str, required=True, help='Target column name')
# user_input.add_argument('--problem', type=str, choices=['classification','regression'], required=True, help='Problem type')
# user_input.add_argument('--algorithm', type=str, required=True
#                     )
# #this variable parses user inputs on terminal
# args = user_input.parse_args()
#
#
# df = pd.read_csv("data.csv")
#
# #these features will be dropped while training the model
# columns_to_drop = ["post_id", "upload_date"]
#
# #defines a new function that is used as a preprocessor
# def preprocess_features(df, target_col, drop_cols=None):
#     if drop_cols is None:
#         drop_cols = []
#     drop_cols = list(drop_cols) + [target_col]
#
#     X = df.drop(columns=drop_cols, errors='ignore')
#     y = df[target_col]
#
#     numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
#     categorical_cols = X.select_dtypes(include=['object', 'category']).columns
#     preprocessor = ColumnTransformer([
#         ('num', StandardScaler(), numeric_cols),
#         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
#     ])
#
#     return X, y, preprocessor
#
#
# X, y, preprocessor = preprocess_features(df, args.target, drop_cols=columns_to_drop)
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
#
# #if else block to know what algorithm to use
# if args.problem == "classification":
#     if args.algorithm == "RandomForest":
#         model = RandomForestClassifier(random_state=42, class_weight="balanced")
#     elif args.algorithm == "GradientBoosting":
#         model = GradientBoostingClassifier(random_state=42)
#     elif args.algorithm == "LogisticRegression":
#         model = LogisticRegression(random_state=0)
#     else:
#         raise ValueError("Invalid algorithm for classification")
# else:
#     if args.algorithm == "RandomForest":
#         model = RandomForestRegressor(random_state=42)
#     elif args.algorithm == "GradientBoosting":
#         model = GradientBoostingRegressor(random_state=42)
#     elif args.algorithm == "LinearRegression":
#         model = LinearRegression()
#     else:
#         raise ValueError("Invalid algorithm for regression")
#
# #creates a pipeline to be used for model
# pipeline = Pipeline([
#     ('preprocess', preprocessor),
#     ('model', model)
# ])
#
# #hyperparameter tuning using n_estimators, max_depth
# param_grid = {
#     "model_fit_intercept":[True, False]
# }
# #uses grid search cv to know the accuracy
# grid_search = GridSearchCV(
#     pipeline,
#     param_grid=param_grid,
#     cv=5,
#     scoring="neg_mean_squared_error",
#     n_jobs=-1
# )
# grid_search.fit(X_train, y_train)
#
# pipeline.fit(X_train, y_train)
#
# # Prediction of the model
# y_pred = pipeline.predict(X_test)
# if args.problem == "classification":
#     print("Accuracy:", accuracy_score(y_test, y_pred))
#     print(classification_report(y_test, y_pred))
# else:
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = math.sqrt(mse)
#     print("Best Params:", grid.best_params_)
#     print("RMSE:", rmse)
#
# #creates a trained model file named model.pkl which will be accessed in the main app file
# joblib.dump(
#     {
#         "model": pipeline,
#         "target": args.target
#     },
#     "model.pkl"
# )

import argparse
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, mean_squared_error

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression


def get_column_types(X):
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    return numeric_cols, categorical_cols

def main():
    parser = argparse.ArgumentParser(description="Train ML model")
    parser.add_argument("--csv", default="data.csv")
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--problem",
        choices=["classification", "regression"],
        default="regression"
    )
    parser.add_argument("--out", default="model.pkl")
    args = parser.parse_args()

    #loading data
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    df = pd.read_csv(csv_path)
    columns_to_drop = ["post_id", "upload_date"]

    df.drop( columns_to_drop, axis=1)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not in dataset")

    X = df.drop(columns=[args.target])
    y = df[args.target]
    numeric_cols, categorical_cols = get_column_types(X)

    #preprocessing starts here
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    #model selection here
    if args.problem == "classification":
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    else:
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    #train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if args.problem == "classification" else None,
    )

 #model training
    pipeline.fit(X_train, y_train)

    #evaluation
    y_pred = pipeline.predict(X_test)

    if args.problem == "classification":
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
    else:
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        print(f"RMSE: {rmse:.4f}")

    joblib.dump(
        {
            "model": pipeline,
            "target": args.target
        },
        "model.pkl")


if __name__ == "__main__":
    main()
