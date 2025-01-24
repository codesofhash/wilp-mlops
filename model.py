import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime
from mlflow.models import infer_signature

# Set the MLflow tracking URI
# mlflow.set_tracking_uri('http://127.0.0.1:5000/')

# Load the dataset
dataset_name = "california_housing.csv"
df = pd.read_csv(dataset_name)

# Split the data into features and target variable
X = df.drop(columns=["target"])
y = df["target"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models and hyperparameter grids
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Gradient Boosting Regressor": GradientBoostingRegressor(),
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
}

param_grids = {
    "Linear Regression": {},
    "Ridge Regression": {"alpha": [0.1, 1, 10, 100]},
    "Lasso Regression": {"alpha": [0.1, 1, 10, 100]},
    "Gradient Boosting Regressor": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5, 7],
    },
    "Random Forest Regressor": {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 20, 30],
        "min_samples_split": [2, 5, 10],
    },
}

# Set the experiment
experiment_name = "wilp-mlops"

# Check if the experiment exists, otherwise create a new one
existing_experiment = mlflow.get_experiment_by_name(experiment_name)

if existing_experiment is None:
    mlflow.create_experiment(experiment_name)
else:
    mlflow.set_experiment(experiment_name)

# Create a timestamped run name
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"run_{timestamp}"

# Start an MLflow run
with mlflow.start_run(run_name=run_name):
    mlflow.set_tag("dataset_used", dataset_name)  # Log the dataset used as a tag
    mlflow.log_param("dataset_used", dataset_name)  # Log dataset as a parameter

    best_model_name = None
    best_model = None
    best_mse = float("inf")

    for model_name, model in models.items():
        print(f"Training model: {model_name}")

        # Get the parameter grid for the model
        param_grid = param_grids[model_name]

        # Perform GridSearchCV with cross-validation
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            scoring="neg_mean_squared_error",
        )

        # Fit the model
        grid_search.fit(X_train, y_train)

        # Get the best model and its parameters
        best_model_temp = grid_search.best_estimator_
        best_params = grid_search.best_params_ if param_grid else "N/A"
        best_score = grid_search.best_score_

        # Log the best model's parameters and score
        mlflow.log_param(f"{model_name}_best_params", str(best_params))
        mlflow.log_metric(f"{model_name}_best_score", best_score)

        # Evaluate the best model on the test set
        predictions = best_model_temp.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        # Log the test MSE
        mlflow.log_metric(f"{model_name}_test_mse", mse)

        # Track the best performing model
        if mse < best_mse:
            best_model_name = model_name
            best_model = best_model_temp
            best_mse = mse

    print(f"Best Model: {best_model_name} with MSE: {best_mse}")

    # Log the best model with MLflow, including input example and signature
    input_example = X_train.iloc[0].to_dict()  # Use the first row as an example for input
    signature = infer_signature(X_train, y_train)  # Automatically infer the signature

    # Log the best model with MLflow, including input example and signature
    mlflow.sklearn.log_model(best_model, "model", input_example=input_example, signature=signature)

    # Register the best model
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    print(f"Registering {best_model_name} model...")

    # Register the model in the Model Registry
    mlflow.register_model(model_uri, best_model_name)

    # Wait for the model version to be created
    client = mlflow.tracking.MlflowClient()
    latest_version = client.get_latest_versions(best_model_name, stages=["None"])[0].version

    # Transition the model to production
    print(f"Model {best_model_name} with version {latest_version} moved to Production")
    client.transition_model_version_stage(
        name=best_model_name,
        version=latest_version,
        stage="Production",
    )
