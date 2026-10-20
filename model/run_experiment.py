import os
from ucimlrepo import fetch_ucirepo
from model.heart_disease_trainer import HeartDiseaseTrainer

# Force MLflow to allow the local file directory tracking backend
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

heart_disease = fetch_ucirepo(id=45)
df = heart_disease.data.features.copy()
df["target"] = heart_disease.data.targets.iloc[:, 0]

# 2. Configure Local MLOps Storage Variables
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)
#LOCAL_URI = "sqlite:////home/dodo/Documents/Bits-AIML/sem3/ml_ops_assignment/mlflow.db"
EXP_NAME = "Heart Disease Prediction"

trainer = HeartDiseaseTrainer(tracking_uri=TRACKING_URI, experiment_name=EXP_NAME)

trainer.generate_and_log_eda(df)

X_train, X_test, y_train, y_test = trainer.process_and_split_data(df)
run_ids = trainer.run_training_pipeline(X_train, X_test, y_train, y_test)

print("\n--- Production run registry output ---")
print(run_ids)