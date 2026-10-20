from pydantic import BaseModel, Field
from typing import Optional

class PatientRequestPayload(BaseModel):
    """
    Component 3: Plain Data Request Payload (DTO).
    Enforces strict structural data types and value validations before calculations.
    """
    age: float = Field(..., description="Age in years", gt=0, lt=120)
    sex: int = Field(..., description="Sex (1 = male; 0 = female)", ge=0, le=1)
    cp: int = Field(..., description="Chest pain type (values 1, 2, 3, 4)", ge=1, le=4)
    trestbps: float = Field(..., description="Resting blood pressure in mm Hg", gt=0)
    chol: float = Field(..., description="Serum cholesterol in mg/dl", gt=0)
    fbs: int = Field(..., description="Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)", ge=0, le=1)
    restecg: int = Field(..., description="Resting electrocardiographic results (0, 1, 2)", ge=0, le=2)
    thalach: float = Field(..., description="Maximum heart rate achieved", gt=0)
    exang: int = Field(..., description="Exercise-induced angina (1 = yes; 0 = no)", ge=0, le=1)
    oldpeak: float = Field(..., description="ST depression induced by exercise relative to rest", ge=0.0)
    slope: int = Field(..., description="The slope of the peak exercise ST segment", ge=1, le=3)

    # Optional fields default to None.
    # This allows the SimpleImputer in your trained Pipeline to fill them in natively!
    ca: Optional[float] = Field(None, description="Number of major vessels (0-3) colored by fluoroscopy", ge=0, le=3)
    thal: Optional[float] = Field(None, description="Thalassemia (3 = normal; 6 = fixed; 7 = reversible)", ge=3, le=7)
