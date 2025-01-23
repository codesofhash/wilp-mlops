import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os
from datetime import datetime
import joblib  # Import joblib for model saving

# Define the path for logging (MLflow local directory)
# mlflow.set_tracking_uri('http://127.0.0.1:5000/')  # This stores the logs and models in a local directory called 'mlruns'

dataset_name = 'california_housing.csv'  # dataset's name for logging

# Load dataset
df = pd.read_csv(dataset_name)

# Assuming the target variable is 'median_house_value' and the rest are features
X = df.drop(columns=['target'])  # Drop the target column
y = df['target']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models and hyperparameter grids
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Gradient Boosting Regressor": GradientBoostingRegressor(),
    "Random Forest Regressor": RandomForestRegressor(random_state=42)
}

param_grids = {
    "Linear Regression": {},
    "Ridge Regression": {
        'alpha': [0.1, 1, 10, 100]
    },
    "Lasso Regression": {
        'alpha': [0.1, 1, 10, 100]
    },
    "Gradient Boosting Regressor": {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7],
    },
    "Random Forest Regressor": {
        'n_estimators': [50,100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
    }
}

# Set the experiment
experiment_name = 'wilp-mlops'

# Check if the experiment exists and is active, otherwise create a new one or restore it
existing_experiment = mlflow.get_experiment_by_name(experiment_name)

if existing_experiment is None:
    # Experiment doesn't exist, create a new one
    mlflow.create_experiment(experiment_name)
elif existing_experiment.lifecycle_stage == "deleted":
    # If the experiment was deleted, recreate it
    mlflow.delete_experiment(existing_experiment.experiment_id)
    mlflow.create_experiment(experiment_name)

mlflow.set_experiment(experiment_name)

# Generate a timestamped run name
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"run_{timestamp}"

# Start an MLflow run with a custom name
with mlflow.start_run(run_name=run_name):
    # Log the dataset used in both the tags and parameters
    mlflow.set_tag("dataset_used", dataset_name)  # Log as a tag (shows in Runs Dashboard)

    # You can also log the dataset path as a parameter
    mlflow.log_param("dataset_used",dataset_name)  # Log as a parameter (shows in Overview > Details)
    
    for model_name, model in models.items():
        print(f"Training model: {model_name}")
        
        # Get the parameter grid for the model
        param_grid = param_grids[model_name]

        # If no hyperparameters to tune (e.g., Linear Regression), perform GridSearchCV with default settings
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

        # Fit the model
        grid_search.fit(X_train, y_train)

        # Get the best model and its parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_ if param_grid else "N/A"  # For Linear Regression, no parameters to tune
        best_score = grid_search.best_score_

        # Log the best model, parameters, and score to MLflow
        mlflow.log_param(f"{model_name}_best_params", str(best_params))
        mlflow.log_metric(f"{model_name}_best_score", best_score)

        # Evaluate the best model on the test set
        predictions = best_model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        # Log the test MSE
        mlflow.log_metric(f"{model_name}_test_mse", mse)

        # Log the best model itself
        mlflow.sklearn.log_model(best_model, model_name)

        # Save the trained model using joblib
        model_filename = f"{model_name.replace(' ', '_')}_model.pkl"
        joblib.dump(best_model, model_filename)  # Save model to a file

        print(f"Best Params: {best_params}")
        print(f"Best Score (CV): {best_score}")
        print(f"Test MSE: {mse}")
        print(f"Saved model as: {model_filename}")
