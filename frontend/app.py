import requests
import streamlit as st  # type: ignore
import logging
from dotenv import load_dotenv
import os

logging.basicConfig(level=logging.INFO, filename="app.log")
logger = logging.getLogger(__name__)

load_dotenv()
API_URL = os.getenv("BACKEND_URL", "http://localhost:8000/predict")

logger.info("App started")
st.title("Welcome to our application")
st.header(":red[ML Model Prediction App]")

sepal_length = st.slider(
    "Sepal Length (e.g., 5.1)", min_value=0.0, max_value=100.0, step=0.1
)
sepal_width = st.slider(
    "Sepal Width (e.g., 3.5)", min_value=0.0, max_value=100.0, step=0.1
)
petal_length = st.slider(
    "Petal Length (e.g., 1.4)", min_value=0.0, max_value=100.0, step=0.1
)
petal_width = st.slider(
    "Petal Width (e.g., 0.2)", min_value=0.0, max_value=100.0, step=0.1
)

if st.button("Predict"):
    try:
        input_data = {
            "features": [sepal_length, sepal_width, petal_length, petal_width]
        }

        response = requests.post(url=API_URL, json=input_data)
        response.raise_for_status()
        if response.status_code == 200:
            predictions = response.json().get("predictions")
            logger.info(f"Predict class {predictions}")
            st.success(f"Predicted class: {predictions}")
        else:
            st.error(f"API Error: {response.text}")
            logger.error(f"API error: {response.text}")
    # except ValueError:
    #     st.error("Please enter valid numeric values for all features.")
    #     logger.error("Features values are not valid")
    except requests.exceptions.RequestException as e:
        st.error(f"Cannot reach API: {e}")
        logger.error(f"can not reach api {e}")
