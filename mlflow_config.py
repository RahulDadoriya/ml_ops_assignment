import os
import mlflow

TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "sqlite:////mlflow/mlflow.db"
)

mlflow.set_tracking_uri(TRACKING_URI)
