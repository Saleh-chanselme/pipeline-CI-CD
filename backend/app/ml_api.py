from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
import logging
import pandas as pd


logging.basicConfig(level=logging.INFO, filename="app.log")
logger = logging.getLogger(__name__)
logger.info("App Started")

app = FastAPI()


class PredictRequest(BaseModel):
    features: list


def load_mlflow_model(experiment_id: str):
    try:
        client = mlflow.MlflowClient()
        runs = client.search_runs(
            experiment_ids=[experiment_id], order_by=["start_time DESC"], max_results=1
        )
        latest_run = runs[0]
        run_id = latest_run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Model loaded successfully from MLflow: run_id={run_id}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from MLflow: {e}")
        return None


model = load_mlflow_model("399503080388943605")


@app.get("/")
def root():
    logger.info("Root endpoint works")
    return {"status": "Ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        logger.error("Prediction requested but model is not loaded")
        raise HTTPException("No model found !")
    try:
        data = pd.DataFrame([req.features])
        logger.info(f"Received data for prediction: {req.features}")
        predictions = model.predict(data)
        logger.info(f"Prediction result : {predictions.to_list()}")
        return {"predictions": predictions.to_list()}
    except Exception as e:
        logger.error(f"Error during prediction {e}")
        raise HTTPException(status_code=500, detail=str(e))
