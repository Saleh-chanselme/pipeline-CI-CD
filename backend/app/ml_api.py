from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
import logging
import pandas as pd
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, filename="app.log")
logger = logging.getLogger(__name__)
logger.info("App Started")


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pipeline-frontend.azurewebsites.net"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    features: list[float]


MODEL_DIR = Path("model_local")


def load_local_mlflow_model(model_path):
    try:
        model = mlflow.pyfunc.load_model(MODEL_DIR)
        logger.info(f"Model loaded successfully from local path: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading local MLflow model: {e}")
        return None


model = load_local_mlflow_model("../model/")


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
        columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        data = pd.DataFrame([req.features], columns=columns)

        logger.info(f"Received data for prediction: {req.features}")
        predictions = model.predict(data)

        logger.info(f"Prediction result: {predictions.tolist()}")
        return {"predictions": predictions.tolist()}
    except Exception as e:
        logger.error(f"Error during prediction {e}")
        raise HTTPException(status_code=500, detail=str(e))
