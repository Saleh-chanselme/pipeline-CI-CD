# Pipeline CI-CD Progect

## Machine Learning Pipeline — Streamlit + FastAPI + MLflow

This project is a complete machine-learning application built with **FastAPI** and **Streamlit**, packaged using **Docker**, automated with **GitHub Actions**, and **fully deployed on Azure Web App Service**.

It includes:

- A **FastAPI backend** that loads a trained MLflow model.
- A **Streamlit frontend** for user interaction.
- Two separate Docker images (frontend & backend) orchestrated with **Docker Compose**.
- **GitHub Actions CI/CD** to automatically build and push Docker images to Docker Hub.
- The entire application is **running on Azure Web App Service**.

---

## Features

### **Backend (FastAPI)**

- Loads an ML model exported from **MLflow**.
- Exposes a `/predict` endpoint for inference.
- Dockerized for local and cloud deployment.
- Deployed on **Azure Web App Service**.

### **Frontend (Streamlit)**

- Clean UI for user inputs.
- Sends inference requests to the backend using `requests.post`.
- Reads backend URL from environment variables.
- Deployed on **Azure Web App Service**.

### **CI/CD (GitHub Actions)**

- Automatically:
  - Builds backend & frontend Docker images
  - Pushes images to **Docker Hub**
- Ensures deployment consistency between local and production.

### **Deployment**

- Backend and frontend Docker images deployed to **Azure Web App Service**.
- Managed through Azure web service + environment variables.

---

## note

- to run the full project use docker compose :
docker-compose up --build
