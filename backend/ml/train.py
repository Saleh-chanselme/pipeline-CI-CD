import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path

MODEL_LOCAL_DIR = Path("model_local")
MODEL_LOCAL_DIR.mkdir(exist_ok=True)

mlflow.set_experiment("MLflow Quickstart")

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

mlflow.sklearn.autolog()

lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.sklearn.log_model(sk_model=lr, name="iris_model")
    mlflow.sklearn.save_model(lr, MODEL_LOCAL_DIR)
    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.set_tag("Training Info", "Basic LR model for iris data")

loaded_model = mlflow.pyfunc.load_model(MODEL_LOCAL_DIR)

iris_feature_names = datasets.load_iris().feature_names
result = pd.DataFrame(X_test, columns=iris_feature_names)
result["actual_class"] = y_test
result["predicted_class"] = loaded_model.predict(X_test)

print(result.head(4))
