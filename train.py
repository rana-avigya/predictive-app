import pandas as pd
import argparse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import joblib
import math
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

#asks user for input for type of pronblem and what algorithm to use
user_input = argparse.ArgumentParser(description="Train a predictive model")
user_input.add_argument('--target', type=str, required=True, help='Target column name')
user_input.add_argument('--problem', type=str, choices=['classification','regression'], required=True, help='Problem type')
user_input.add_argument('--algorithm', type=str, required=True
                    )

args = user_input.parse_args()


df = pd.read_csv(args.csv)

#these features will be dropped while training the model
columns_to_drop = ["post_id", "upload_date"]

#defines a new function that is used as a preprocessor
def preprocess_features(df, target_col, drop_cols=None):
    if drop_cols is None:
        drop_cols = []
    drop_cols = list(drop_cols) + [target_col]

    X = df.drop(columns=drop_cols, errors='ignore')
    y = df[target_col]

    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

    return X, y, preprocessor


X, y, preprocessor = preprocess_features(df, args.target, drop_cols=columns_to_drop)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)




#if else block to know what algorithm to use
if args.problem == "classification":
    if args.algorithm == "RandomForest":
        model = RandomForestClassifier(n_estimators=300,
                                       max_depth =15,
                                       min_samples_split = 5, 
                                       min_samples_leaf=2,
                                       max_features='sqrt',
                                       class_weight='balanced',
                                       random_state=42)
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

#creates a pipeline to be used for model
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', model)
])

#hyperparameter tuning using n_estimators, max_depth
param_grid = {
    'model__n_estimators': [200, 300, 500],
    'model__max_depth': [10, 15, 20, None],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2']
}
#uses grid search cv to know the accuracy
grid = GridSearchCV(
    estimator=pipeline,   
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

pipeline.fit(X_train, y_train)

# Prediction of the model
y_pred = pipeline.predict(X_test)
if args.problem == "classification":
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
else:
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    print("Best Params:", grid.best_params_)
    print("RMSE:", rmse)

#creates a trained model file named model.pkl which will be accessed in the main app file
joblib.dump(
    {
        "model": pipeline,
        "target": args.target
    },
    "model.pkl"
)