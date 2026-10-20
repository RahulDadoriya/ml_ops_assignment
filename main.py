
import os

import mlflow
from fastapi import FastAPI

import controller.heart_disease_controller as hc


app = FastAPI(
    title="Heart Disease MLOps Production Engine",
    version="1.0.0"
)

# Configure MLflow before initializing the controller.
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

mlflow.set_tracking_uri(TRACKING_URI)

# Create the controller once when the application starts.
# It will load the model assigned the "champion" alias.
controller = hc.HeartDiseaseAPIController()

# Register the application's endpoints.
app.include_router(hc.router)
app.include_router(controller.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000
    )
