# LogisticRegression Model Documentation (Iris Dataset with MLflow)

## 1. Model Description

This project uses a **LogisticRegression** model from **scikit-learn** to classify Iris flowers into three species:

- setosa
- versicolor
- virginica

The model is trained, logged, and tracked using **MLflow**, including metrics and hyperparameters.

---

## 2. Dataset

- **Dataset**: Iris dataset (from scikit-learn)
- **Features**:
  - sepal length (cm)
  - sepal width (cm)
  - petal length (cm)
  - petal width (cm)
- **Target**: species of the flower

The data is split into **training (80%)** and **test (20%)** sets using `train_test_split`.

---

## 3. Training and MLflow Tracking

### Hyperparameters

```python
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}
```

## Training Code Example:

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
import mlflow
import mlflow.sklearn

# Load dataset

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Enable MLflow autologging

mlflow.sklearn.autolog()

# Train LogisticRegression model

lr = LogisticRegression(\*\*params)
lr.fit(X_train, y_train)

# Start MLflow run

with mlflow.start_run():
mlflow.log_params(params)
mlflow.sklearn.log_model(sk_model=lr, name="iris_model")

## Example of logging metrics:

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred = lr.predict(X_test)
mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
mlflow.log_metric("precision", precision_score(y_test, y_pred, average="weighted"))
mlflow.log_metric("recall", recall_score(y_test, y_pred, average="weighted"))
mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average="weighted"))

## Save Model Locally:

from pathlib import Path

MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)
mlflow.sklearn.save_model(lr, MODEL_DIR)

## Load Model for Predictions:

import mlflow.pyfunc

model_url = MODEL_DIR
loaded_model = mlflow.pyfunc.load_model(model_url)
predictions = loaded_model.predict(X_test)
