import pytest
from pydantic import ValidationError
from dto.patient_request import PatientRequestPayload

def test_valid_payload_is_accepted(valid_patient_data):
    payload = PatientRequestPayload(**valid_patient_data)
    assert payload.age == 52
    assert payload.sex == 1
    assert payload.ca == 0

def test_optional_fields_can_be_omitted(valid_patient_data):
    valid_patient_data.pop("ca")
    valid_patient_data.pop("thal")
    payload = PatientRequestPayload(**valid_patient_data)
    assert payload.ca is None
    assert payload.thal is None

@pytest.mark.parametrize("field,value", [
    ("age", 0), ("age", 120), ("sex", 2), ("sex", -1),
    ("cp", 0), ("cp", 5), ("trestbps", 0), ("chol", -1),
    ("fbs", 2), ("restecg", 3), ("thalach", 0), ("exang", 2),
    ("oldpeak", -0.1), ("slope", 0), ("slope", 4),
    ("ca", 4), ("thal", 2),
])
def test_out_of_range_values_are_rejected(valid_patient_data, field, value):
    valid_patient_data[field] = value
    with pytest.raises(ValidationError):
        PatientRequestPayload(**valid_patient_data)

def test_missing_required_field_is_rejected(valid_patient_data):
    valid_patient_data.pop("age")
    with pytest.raises(ValidationError):
        PatientRequestPayload(**valid_patient_data)
