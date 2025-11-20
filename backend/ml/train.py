import mlflow
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

mlflow.set_experiment("MLflow seconde exp")

# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

# Enable autologging for scikit-learn
mlflow.sklearn.autolog()

# Start an MLflow run
with mlflow.start_run() as run:
    # Log the hyperparameters
    mlflow.log_params(params)

    # Train the model
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)

    # Log the model
    mlflow.sklearn.log_model(sk_model=lr, name="iris_model")

    # Predict on the test set
    y_pred = lr.predict(X_test)

    # Log metrics
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision_score(y_test, y_pred, average="macro"))
    mlflow.log_metric("recall", recall_score(y_test, y_pred, average="macro"))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average="macro"))

    # Optional: Set a tag
    mlflow.set_tag("Training Info", "Basic LR model for iris data")

    # Model URI for current run
    model_uri = f"runs:/{run.info.run_id}/iris_model"

# Load the model back
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Predictions
predictions = loaded_model.predict(X_test)

# Results dataframe
iris_feature_names = datasets.load_iris().feature_names
result = pd.DataFrame(X_test, columns=iris_feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions

print(result[:4])
