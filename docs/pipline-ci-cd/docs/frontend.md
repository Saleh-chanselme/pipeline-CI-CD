## Pipeline CI/CD

# ML Model Prediction App - Frontend

## Overview

This frontend is built using **Streamlit**. It allows users to input features for the Iris dataset and get predictions from the ML model via a FastAPI backend.

---

## Features

- Input sliders for the following features:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- A **Predict** button to send data to the backend API.
- Display predicted class returned by the ML model.
- Error handling and logging for invalid inputs or API issues.

---

## Code Example

```python
import requests
import streamlit as st
import logging

logger = logging.basicConfig(level=logging.INFO, filename="app.log")
logger = logging.getLogger(__name__)

API_URL = "http://127.0.0.1:8000/predict"

st.title("Welcome to our application")
st.header(":red[ML Model Prediction App]")

sepal_length = st.slider("Sepal Length (e.g., 5.1)", 0.0, 100.0, 0.1)
sepal_width = st.slider("Sepal Width (e.g., 3.5)", 0.0, 100.0, 0.1)
petal_length = st.slider("Petal Length (e.g., 1.4)", 0.0, 100.0, 0.1)
petal_width = st.slider("Petal Width (e.g., 0.2)", 0.0, 100.0, 0.1)

if st.button("Predict"):
    try:
        input_data = {"features": [sepal_length, sepal_width, petal_length, petal_width]}
        response = requests.post(url=API_URL, json=input_data)
        response.raise_for_status()
        if response.status_code == 200:
            predictions = response.json().get("predictions")
            st.success(f"Predicted class: {predictions}")
        else:
            st.error(f"API Error: {response.text}")
    except ValueError:
        st.error("Please enter valid numeric values for all features.")
    except requests.exceptions.RequestException as e:
        st.error(f"Cannot reach API: {e}")
```

## run APP Example :

streamlit run app.py
