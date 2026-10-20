from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest
from model.heart_disease_trainer import HeartDiseaseTrainer

def make_trainer():
    # Prevent the test from changing real MLflow settings.
    with patch("model.heart_disease_trainer.mlflow.set_tracking_uri"), \
         patch("model.heart_disease_trainer.mlflow.set_experiment"), \
         patch("model.heart_disease_trainer.mlflow.sklearn.autolog"):
        return HeartDiseaseTrainer(
            tracking_uri="file:///tmp/test-mlruns",
            experiment_name="pytest-heart-disease",
        )

def test_process_and_split_data_returns_expected_shapes():
    trainer = make_trainer()
    row_count = 100
    data = pd.DataFrame({
        "age": list(range(20, 20 + row_count)),
        "chol": list(range(150, 150 + row_count)),
        "target": [0, 1] * 50,
    })
    X_train, X_test, y_train, y_test = trainer.process_and_split_data(data)
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20
    assert "target" not in X_train.columns
    assert set(np.unique(y_train)) == {0, 1}
    assert set(np.unique(y_test)) == {0, 1}

def test_process_and_split_data_raises_when_target_missing():
    trainer = make_trainer()
    data = pd.DataFrame({"age": [40, 50, 60], "chol": [180, 200, 220]})
    with pytest.raises(KeyError):
        trainer.process_and_split_data(data)

def test_training_pipeline_registers_best_model():
    trainer = make_trainer()
    X_train = pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 50, 55, 60],
        "chol": [160, 170, 180, 190, 220, 230, 240, 250],
    })
    y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    X_test = pd.DataFrame({
        "age": [28, 58, 33, 53], "chol": [165, 245, 175, 235]
    })
    y_test = np.array([0, 1, 0, 1])
    run_ids = ["run-logistic", "run-forest"]
    run_objects = [
        MagicMock(info=MagicMock(run_id=run_ids[0])),
        MagicMock(info=MagicMock(run_id=run_ids[1])),
    ]

    class FakeRunContext:
        def __init__(self, run_object):
            self.run_object = run_object
        def __enter__(self):
            return self.run_object
        def __exit__(self, exc_type, exc_value, traceback):
            return False

    with patch("model.heart_disease_trainer.mlflow.start_run",
               side_effect=[FakeRunContext(run_objects[0]), FakeRunContext(run_objects[1])]), \
         patch("model.heart_disease_trainer.mlflow.log_param"), \
         patch("model.heart_disease_trainer.mlflow.log_metric"), \
         patch("model.heart_disease_trainer.mlflow.sklearn.log_model"), \
         patch("model.heart_disease_trainer.mlflow.register_model",
               return_value=MagicMock(version="3")) as register_model, \
         patch("model.heart_disease_trainer.mlflow.tracking.MlflowClient") as client_class:
        result = trainer.run_training_pipeline(X_train, X_test, y_train, y_test)

    assert result == {
        "Logistic Regression": "run-logistic",
        "Random Forest": "run-forest",
    }
    register_model.assert_called_once()
    client_class.return_value.set_registered_model_alias.assert_called_once_with(
        name="HeartDiseaseModel", alias="champion", version="3"
    )
