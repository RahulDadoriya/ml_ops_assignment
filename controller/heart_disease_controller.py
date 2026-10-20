
import logging

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.tracking import MlflowClient
from fastapi import APIRouter, HTTPException

from dto.patient_request import PatientRequestPayload


logger = logging.getLogger(__name__)

# Main application router
router = APIRouter()


class HeartDiseaseAPIController:
    """Load the champion model and provide prediction endpoints."""

    def __init__(self):
        """Initialize the controller and load the registered model."""

        self.model_name = "HeartDiseaseModel"
        self.model_alias = "champion"
        self.model_path = (
            f"models:/{self.model_name}@{self.model_alias}"
        )

        self.router = APIRouter()

        logger.info(
            "Connecting to MLflow Tracking Server: %s",
            mlflow.get_tracking_uri()
        )

        try:
            # Read the model version currently assigned to champion.
            client = MlflowClient(
                tracking_uri=mlflow.get_tracking_uri()
            )

            model_version = client.get_model_version_by_alias(
                name=self.model_name,
                alias=self.model_alias
            )

            # Store the real version and originating run ID.
            self.model_version = str(model_version.version)
            self.run_id = model_version.run_id

            logger.info(
                "Loading model %s, version %s, run ID %s",
                self.model_name,
                self.model_version,
                self.run_id
            )

            # Load the model selected by the MLflow alias.
            self.model = mlflow.sklearn.load_model(
                self.model_path
            )

            logger.info("Champion model loaded successfully.")

        except Exception as e:
            logger.exception(
                "Failed to load the champion model from MLflow."
            )

            raise RuntimeError(
                "Failed to load the champion model "
                "from the MLflow Model Registry."
            ) from e

        # Register the prediction endpoint.
        self.router.add_api_route(
            "/predict",
            self.predict,
            methods=["POST"],
            name="predict_heart_disease"
        )

    def predict(self, payload: PatientRequestPayload):
        """Predict the class and probability for one patient."""

        try:
            # Step 1: Convert request data into a DataFrame.
            input_data = payload.model_dump()
            input_df = pd.DataFrame([input_data])

            # Step 2: Predict the class.
            predictions = self.model.predict(input_df)
            raw_prediction = int(predictions[0])

            # Step 3: Predict class probabilities.
            probabilities = self.model.predict_proba(input_df)

            # Step 4: Find the probability for the predicted class.
            model_classes = list(self.model.classes_)
            class_index = model_classes.index(raw_prediction)

            confidence = float(
                probabilities[0, class_index]
            )

            # Step 5: Map classes to their descriptions.
            label_mapping = {
                0: "Absence of Heart Disease (Normal Profile)",
                1: "Presence of Heart Disease Detected",
                2: "Presence of Heart Disease Detected (Stage 2)",
                3: "Presence of Heart Disease Detected (Stage 3)",
                4: "Presence of Heart Disease Detected (Stage 4)"
            }

            prediction_label = label_mapping.get(
                raw_prediction,
                "Unknown Status"
            )

            # Step 6: Return prediction and model metadata.
            return {
                "status": "success",
                "prediction": raw_prediction,
                "prediction_label": prediction_label,
                "probability": round(confidence, 4),
                "model_meta": {
                    "model_name": self.model_name,
                    "model_alias": self.model_alias,
                    "model_version": self.model_version,
                    "mlflow_run_id": self.run_id
                }
            }

        except Exception as e:
            logger.exception(
                "Heart disease prediction failed."
            )

            raise HTTPException(
                status_code=500,
                detail="Prediction failed. Check the server logs."
            ) from e


@router.get("/")
def greeting():
    """Welcome endpoint."""

    return {
        "message": "Welcome to Heart Disease Predictor!"
    }
