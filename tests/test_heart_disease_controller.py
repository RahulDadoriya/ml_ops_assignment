from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from fastapi import HTTPException
from controller.heart_disease_controller import HeartDiseaseAPIController, greeting

def build_controller(model=None):
    if model is None:
        model = MagicMock()
        model.predict.return_value = np.array([1])
        model.predict_proba.return_value = np.array([[0.2, 0.8]])
        model.classes_ = np.array([0, 1])

    fake_version = SimpleNamespace(version="7", run_id="test-run-123")
    with patch("controller.heart_disease_controller.mlflow.get_tracking_uri",
               return_value="http://test-mlflow"), \
         patch("controller.heart_disease_controller.MlflowClient") as client_class, \
         patch("controller.heart_disease_controller.mlflow.sklearn.load_model",
               return_value=model):
        client_class.return_value.get_model_version_by_alias.return_value = fake_version
        controller = HeartDiseaseAPIController()
    return controller, model

def test_controller_loads_champion_model():
    controller, _ = build_controller()
    assert controller.model_name == "HeartDiseaseModel"
    assert controller.model_alias == "champion"
    assert controller.model_version == "7"
    assert controller.run_id == "test-run-123"

def test_predict_returns_prediction_and_metadata(valid_patient_payload):
    controller, model = build_controller()
    result = controller.predict(valid_patient_payload)
    assert result["status"] == "success"
    assert result["prediction"] == 1
    assert result["prediction_label"] == "Presence of Heart Disease Detected"
    assert result["probability"] == 0.8
    assert result["model_meta"]["model_version"] == "7"
    assert result["model_meta"]["mlflow_run_id"] == "test-run-123"
    model.predict.assert_called_once()
    model.predict_proba.assert_called_once()

def test_predict_returns_unknown_status_for_unmapped_class(valid_patient_payload):
    model = MagicMock()
    model.predict.return_value = np.array([9])
    model.predict_proba.return_value = np.array([[1.0]])
    model.classes_ = np.array([9])
    controller, _ = build_controller(model)
    result = controller.predict(valid_patient_payload)
    assert result["prediction_label"] == "Unknown Status"

def test_predict_wraps_model_error_in_http_500(valid_patient_payload):
    controller, model = build_controller()
    model.predict.side_effect = ValueError("model failed")
    with pytest.raises(HTTPException) as error:
        controller.predict(valid_patient_payload)
    assert error.value.status_code == 500
    assert error.value.detail == "Prediction failed. Check the server logs."

def test_controller_raises_runtime_error_when_registry_fails():
    with patch("controller.heart_disease_controller.mlflow.get_tracking_uri",
               return_value="http://test-mlflow"), \
         patch("controller.heart_disease_controller.MlflowClient") as client_class:
        client_class.return_value.get_model_version_by_alias.side_effect = RuntimeError(
            "registry unavailable"
        )
        with pytest.raises(RuntimeError, match="Failed to load the champion model"):
            HeartDiseaseAPIController()

def test_greeting_endpoint():
    assert greeting() == {"message": "Welcome to Heart Disease Predictor!"}
