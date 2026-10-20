import pytest

@pytest.fixture
def valid_patient_data():
    return {
        "age": 52, "sex": 1, "cp": 2, "trestbps": 130, "chol": 240,
        "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
        "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 3,
    }

@pytest.fixture
def valid_patient_payload(valid_patient_data):
    from dto.patient_request import PatientRequestPayload
    return PatientRequestPayload(**valid_patient_data)
