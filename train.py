import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import joblib
from feature_engineering import preprocess_features

# CONFIG
TARGET = "genre"          # update with your target column
PROBLEM = "classification" # "classification" or "regression"
DROP_COLS = ["track_id"]  # columns to ignore

# Load dataset
df = pd.read_csv("data.csv").sample(500, random_state=42)
X, y, preprocessor = preprocess_features(df, TARGET, drop_cols=DROP_COLS)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Select model
if PROBLEM == "classification":
    model = RandomForestClassifier(random_state=42)
else:
    model = RandomForestRegressor(random_state=42)

# Pipeline
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', model)
])

# Optional: Hyperparameter tuning
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 5, 10],
    'model__min_samples_split': [2, 5]
}

grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

# Evaluate
y_pred = grid.predict(X_test)
if PROBLEM == "classification":
    print("Best Params:", grid.best_params_)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
else:
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("Best Params:", grid.best_params_)
    print("RMSE:", rmse)

# Save model
joblib.dump(grid.best_estimator_, "model.pkl")
print("Model saved as model.pkl")
