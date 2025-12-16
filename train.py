# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
# import joblib
# from feature_engineering import preprocess_features
# import math

# # CONFIG
# TARGET = "customer_rating"          # update with your target column
# PROBLEM = "regression" # "classification" or "regression"
# DROP_COLS = ["customer_id"]  # columns to ignore

# # Load dataset
# df = pd.read_csv("data.csv").sample(5000, random_state=42)
# X, y, preprocessor = preprocess_features(df, TARGET, drop_cols=DROP_COLS)

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Select model
# if PROBLEM == "classification":
#     model = RandomForestClassifier(random_state=42)
# else:
#     model = RandomForestRegressor(random_state=42)

# # Pipeline
# pipeline = Pipeline([
#     ('preprocess', preprocessor),
#     ('model', model)
# ])



# # Evaluate
# y_pred = grid.predict(X_test)
# if PROBLEM == "classification":
#     print("Best Params:", grid.best_params_)
#     print("Accuracy:", accuracy_score(y_test, y_pred))
#     print(classification_report(y_test, y_pred))
# else:
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = math.sqrt(mse)
#     print("Best Params:", grid.best_params_)
#     print("RMSE:", rmse)

# # Save model
# joblib.dump(grid.best_estimator_, "model.pkl")
# print("Model saved as model.pkl")


import pandas as pd
import argparse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import joblib
from feature_engineering import preprocess_features
import math


parser = argparse.ArgumentParser(description="Train a predictive model")
parser.add_argument('--csv', type=str, default='data.csv', help='Path to CSV file')
parser.add_argument('--target', type=str, required=True, help='Target column name')
parser.add_argument('--problem', type=str, choices=['classification','regression'], required=True, help='Problem type')
parser.add_argument('--algorithm', type=str, required=True,
                    choices=['RandomForest','GradientBoosting','LogisticRegression','LinearRegression'],
                    help='Machine learning algorithm')
parser.add_argument('--out', type=str, default='model.pkl', help='Output model filename')

args = parser.parse_args()


df = pd.read_csv(args.csv)

DROP_COLS = ["post_id", "upload_date"]
X, y, preprocessor = preprocess_features(df, args.target, drop_cols=DROP_COLS)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


if args.problem == "classification":
    if args.algorithm == "RandomForest":
        model = RandomForestClassifier(random_state=42)
    elif args.algorithm == "GradientBoosting":
        model = GradientBoostingClassifier(random_state=42)
    elif args.algorithm == "LogisticRegression":
        model = LogisticRegression(max_iter=1000)
    else:
        raise ValueError("Invalid algorithm for classification")
else:
    if args.algorithm == "RandomForest":
        model = RandomForestRegressor(random_state=42)
    elif args.algorithm == "GradientBoosting":
        model = GradientBoostingRegressor(random_state=42)
    elif args.algorithm == "LinearRegression":
        model = LinearRegression()
    else:
        raise ValueError("Invalid algorithm for regression")

#pipeline build
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', model)
])

#hyperparameter tuning
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 5, 10],
    'model__min_samples_split': [2, 5]
}

grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)


pipeline.fit(X_train, y_train)

# Prediction
y_pred = pipeline.predict(X_test)
if args.problem == "classification":
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
else:
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    print("Best Params:", grid.best_params_)
    print("RMSE:", rmse)

# model save using joblib
# joblib.dump(pipeline, args.out)
# print(f"Model saved as {args.out} using {args.algorithm}")

joblib.dump(
    {
        "model": pipeline,
        "target": args.target
    },
    args.out
)
