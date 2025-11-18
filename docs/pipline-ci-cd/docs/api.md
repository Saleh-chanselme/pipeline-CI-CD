# API Documentation — Iris LogisticRegression Model

## 1. Overview

This API is built with **FastAPI** and serves predictions from a **LogisticRegression** model trained on the Iris dataset.

The model is tracked and loaded dynamically from **MLflow**.

It provides two endpoints:

1. **GET /** → Health check
2. **POST /predict** → Predict the class of Iris flowers based on input features

All API events are logged into `app.log`.

---

## 2. GET /

Endpoint to check that the API is running.

### Request

### Response Example

```json
{
  "status": "Ok"
}
```

## 3. POST /predict:

Endpoint to send input features to the LogisticRegression model and receive predictions.

# Request Body

{
"features": [5.1, 3.5, 1.4, 0.2]
}

# Response Example:

{
"predictions": [0]
}

## Loading the Model:

The model is loaded dynamically from MLflow using the latest run of a given experiment:
import mlflow
import mlflow.pyfunc

def load_mlflow_model(experiment_id: str):
client = mlflow.MlflowClient()
runs = client.search_runs(
experiment_ids=[experiment_id], order_by=["start_time DESC"], max_results=1
)
latest_run = runs[0]
run_id = latest_run.info.run_id
model_uri = f"runs:/{run_id}/model"
model = mlflow.pyfunc.load_model(model_uri)
return model

The API uses experiment_id="399503080388943605".
If the model fails to load, the /predict endpoint will return an error.

## run API

fastapi dev app/ml_api.py
