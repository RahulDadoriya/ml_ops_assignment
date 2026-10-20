# Heart Disease Prediction — MLOps Project

An end-to-end Machine Learning Operations (MLOps) project for serving heart disease predictions through a REST API, tracking experiments with MLflow, managing model versions through the MLflow Model Registry, and deploying the application on Kubernetes.

## 1. Project Overview

This project demonstrates how a trained machine learning model can be developed, tracked, versioned, containerized, deployed, and served through a production-style API.

The application uses FastAPI to expose prediction endpoints. MLflow manages experiment tracking and model versions, PostgreSQL stores MLflow metadata, and persistent artifact storage holds trained model artifacts.

The application is deployed on Kubernetes using Minikube for local development and testing.

### Key Features

- **Machine Learning:** Train and evaluate heart disease classification models.
- **Experiment Tracking:** Track model runs and associated metrics using MLflow.
- **Model Registry:** Register models and select a production version using the `champion` alias.
- **REST API:** Serve predictions through FastAPI.
- **Containerization:** Package the API and MLflow server using Docker.
- **Orchestration:** Deploy and manage application services using Kubernetes.
- **Persistent Storage:** Preserve PostgreSQL data and MLflow model artifacts across Pod restarts.
- **API Documentation:** Test prediction endpoints using automatically generated Swagger UI.
- **Model Version Metadata:** Return the registered model name, alias, version, and originating MLflow run ID in prediction responses.

### Run project
- **Clone the project locally**
- **Navigate to root directory of project**
- **Run cmd: ```sudo chmod +x run_script.sh```**
- **Execute the script using cmd: ```./run_script.sh```**
- **If you see error for port-forwarding then run below 2 cmd:**
  - ***```kubectl port-forward --address 127.0.0.1 service/heart-disease-api-service 8000:8000```***
  - ***```kubectl port-forward --address 127.0.0.1 service/mlflow-tracking 5000:5000```***
- **Finaly you can access mlflow-ui & fast-api at below url:**
  - ***ML flow ui: ```http://127.0.0.1:5000```***
  - ***Fast api: ```http://127.0.0.1:8000/docs```***
## 2. System Architecture

```text
                 Model Training
                       |
                       v
              MLflow Tracking Server
                       |
             +---------+----------+
             |                    |
             v                    v
        PostgreSQL         Model Artifacts
        Metadata           Persistent Storage
             |
             v
       MLflow Model Registry
       HeartDiseaseModel
       Alias: champion
             |
             v
        FastAPI Application
             |
             v
       POST /predict
             |
             v
       Prediction Response
```

### Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Programming language | Python | Model development and API implementation |
| Machine learning | scikit-learn | Model training and prediction |
| Experiment tracking | MLflow | Track runs, metrics, and models |
| Model Registry | MLflow Model Registry | Register and manage model versions |
| Backend database | PostgreSQL | Store MLflow metadata |
| Artifact storage | Persistent volume | Store model artifacts |
| API framework | FastAPI | Expose prediction endpoints |
| API server | Uvicorn | Run the FastAPI application |
| Containerization | Docker | Build application images |
| Orchestration | Kubernetes | Manage application services |
| Local Kubernetes | Minikube | Run Kubernetes locally |
| API testing | Swagger UI | Test API endpoints |

## 3. Prerequisites

Install the following tools before running the project.

### Required software

1. **Python 3.14 or a version supported by all project dependencies.**
2. **pip** for Python package installation.
3. **Docker Engine** for building container images.
4. **Minikube** for local Kubernetes.
5. **kubectl** for managing Kubernetes resources.
6. **Git** for source control.

Verify the installations:

```bash
python --version
python -m pip --version
docker --version
minikube version
kubectl version --client
git --version
```

Ensure Docker is running and your user has permission to access it.

### Start Minikube

The following commands assume that Minikube uses the Docker driver:

```bash
minikube start --driver=docker
minikube status
kubectl get nodes
```

The Kubernetes node should report `Ready`.

> **Note:** Use the same Linux user and Minikube profile consistently. If Minikube was created under your normal user, avoid running subsequent Minikube commands with `sudo`, as this may select a different profile.

## 4. Clone the Repository

Clone the repository and enter the project directory:

```bash
git clone <YOUR_GIT_REPOSITORY_URL>
cd ml_ops_assignment
```

Replace `<YOUR_GIT_REPOSITORY_URL>` with the URL of your Git repository.

## 5. Project Structure

The main files and directories are expected to follow this general structure. Actual training script filenames may differ depending on the final repository.

```text
ml_ops_assignment/
│
├── main.py
├── requirements.txt
├── Dockerfile
├── Dockerfile.mlflow
├── deployment.yaml
├── service.yaml
├── mlflow-stack.yaml
├── .gitignore
├── README.md
│
├── controller/
│   └── heart_disease_controller.py
│
├── dto/
│   └── patient_request.py
│
├── model/
│   └── <training scripts>
│
├── mlruns/                 # Local MLflow data; not committed
├── mlflow.db               # Local SQLite database; not committed
└── backups/                # Local backups; not committed
```

The Kubernetes deployment uses PostgreSQL and persistent artifact storage. The local SQLite database and `mlruns/` directory are not required for the Kubernetes tracking server.

## 6. Local Development Setup

### Step 1 — Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Linux:

```bash
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

Ensure that `requirements.txt` includes the packages used by the application, such as FastAPI, Uvicorn, MLflow, pandas, and scikit-learn.

### Step 3 — Configure the MLflow Tracking URI

For local development, set the tracking URI to the MLflow server you intend to use.

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

The URI above assumes that the MLflow Tracking Server is running locally or is accessible through a Kubernetes port-forward.

Do not use `http://mlflow-tracking:5000` from your ordinary host terminal. That service name is intended for communication between Pods inside the Kubernetes cluster.

## 7. Model Training and Experiment Tracking

Train the heart disease classification models using the training scripts in the `model/` directory.

For example, if your training entry point is `model/run_experiment.py`:

```bash
source .venv/bin/activate

export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

python model/run_experiment.py
```

Replace the example script path with the actual training entry point in your repository.

The training pipeline should:

1. Load the dataset.
2. Perform the required preprocessing and feature engineering.
3. Train the selected classification models.
4. Evaluate model performance.
5. Log relevant metrics and model artifacts to MLflow.
6. Associate the runs with the appropriate MLflow experiment.

### MLflow Experiment Tracking

Open the MLflow UI:

[http://localhost:5000](http://localhost:5000)

The interface allows you to inspect experiments, compare runs, review metrics, and access logged artifacts.

For the Kubernetes deployment, start a port-forward in a separate terminal:

```bash
kubectl port-forward service/mlflow-tracking 5000:5000
```

Keep the terminal open while accessing the UI.

## 8. MLflow Model Registry

The project uses the following registered model:

**Registered model:** `HeartDiseaseModel`

**Alias:** `champion`

The FastAPI controller loads the model using this MLflow model URI:

```python
models:/HeartDiseaseModel@champion
```

This approach separates the model's identity from its specific run ID.

### Registering and Promoting a Model

1. Open the MLflow UI.
2. Navigate to the experiment containing the trained model.
3. Open the relevant run and register its model artifact as `HeartDiseaseModel`.
4. Open the registered model in the Model Registry.
5. Select the version you want to serve.
6. Assign the alias `champion` to that version.

The FastAPI application loads the version currently assigned to this alias when the application starts.

**Important:** Changing the alias does not automatically replace a model already loaded into a running FastAPI process. Restart or roll out the API Deployment after promoting a new version.

## 9. Build Docker Images

Build the FastAPI image from the project root:

```bash
docker build -t heart-disease-api:v3 .
```

Build the MLflow Tracking Server image:

```bash
docker build -f Dockerfile.mlflow -t heart-disease-mlflow:v1 .
```

If the MLflow image is already built and has not changed, it does not need to be rebuilt.

### Load Images into Minikube

Minikube has its own container runtime image store. Loading a Docker image into Minikube makes it available to Pods configured with `imagePullPolicy: Never`.

```bash
minikube image load heart-disease-api:v3
minikube image load heart-disease-mlflow:v1
```

Verify the images:

```bash
minikube image ls
```

Ensure that the image names and tags match those referenced in the Kubernetes manifests.

## 10. Deploy the Infrastructure on Kubernetes

The infrastructure manifest `mlflow-stack.yaml` defines the components required for MLflow tracking:

- Kubernetes Secret for the PostgreSQL credentials.
- PostgreSQL Deployment and Service.
- Persistent volume claim for PostgreSQL.
- Persistent volume claim for MLflow artifacts.
- MLflow Tracking Server Deployment and Service.

### Step 1 — Apply the MLflow infrastructure

```bash
kubectl apply -f mlflow-stack.yaml
```

Check the resources:

```bash
kubectl get pods
kubectl get services
kubectl get pvc
```

Wait for PostgreSQL and the MLflow Tracking Server to become ready.

```bash
kubectl rollout status deployment/postgres
kubectl rollout status deployment/mlflow-tracking
```

If the PostgreSQL Deployment has a different name in your manifest, use that name instead.

### Step 2 — Access the MLflow UI

```bash
kubectl port-forward service/mlflow-tracking 5000:5000
```

Open:

[http://localhost:5000](http://localhost:5000)

The MLflow Tracking Server uses PostgreSQL for metadata and persistent artifact storage for model files.

### Step 3 — Configure the API Deployment

Ensure `deployment.yaml` references the intended API image:

```yaml
image: heart-disease-api:v3
imagePullPolicy: Never
```

The API must use the internal Kubernetes service address for MLflow:

```yaml
env:
  - name: MLFLOW_TRACKING_URI
    value: "http://mlflow-tracking:5000"
```

The current implementation loads the `champion` alias at application startup. A hardcoded `PRODUCTION_RUN_ID` is not needed when using the registered model URI.

### Step 4 — Deploy the API

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

Check rollout status:

```bash
kubectl rollout status deployment/heart-disease-api
kubectl get pods
kubectl get services
```

The API Pod should become `Running` and report `1/1` ready.

### Step 5 — Inspect application logs

```bash
kubectl logs deployment/heart-disease-api --tail=100
```

Inspect MLflow logs if necessary:

```bash
kubectl logs deployment/mlflow-tracking --tail=100
```

## 11. Access the FastAPI Application

Start port forwarding:

```bash
kubectl port-forward service/heart-disease-api-service 8000:8000
```

Open the Swagger UI:

[http://localhost:8000/docs](http://localhost:8000/docs)

The root endpoint is available at:

[http://localhost:8000/](http://localhost:8000/)

Keep the port-forward process running while testing the API.

## 12. Prediction API

### Endpoint

```http
POST /predict
```

The endpoint accepts a JSON payload defined by `PatientRequestPayload`.

The input field names, data types, and required values must match the actual schema in `dto/patient_request.py`.

### How to Test

1. Open the Swagger UI.
2. Expand `POST /predict`.
3. Select **Try it out**.
4. Enter a valid patient payload according to the schema.
5. Select **Execute**.
6. Review the prediction and model metadata.

### Example Response

The following is an illustrative response. Actual prediction values, labels, probabilities, and run IDs depend on the model and request.

```json
{
  "status": "success",
  "prediction": 0,
  "prediction_label": "Absence of Heart Disease (Normal Profile)",
  "probability": 0.85,
  "model_meta": {
    "model_name": "HeartDiseaseModel",
    "model_alias": "champion",
    "model_version": "1",
    "mlflow_run_id": "actual-run-id"
  }
}
```

The probability represents the model's predicted probability for the predicted class. It should not be interpreted as a clinically validated assessment of an individual's health.

## 13. Updating the Deployed Model

To promote a new model version:

1. Train and evaluate a new model.
2. Register the model version in MLflow.
3. Assign the `champion` alias to the desired version.
4. Restart the API Deployment to load the newly selected model.

```bash
kubectl rollout restart deployment/heart-disease-api
kubectl rollout status deployment/heart-disease-api
```

Verify the API response again and confirm that `model_meta.model_version` matches the intended registered version.

The current workflow requires this restart because the controller loads the model into memory at startup. Automated promotion-triggered deployment can be added later through a CI/CD pipeline or another deployment automation mechanism.

## 14. Troubleshooting

### Issue 1 — Minikube cannot find a Docker image

Load the image into Minikube:

```bash
minikube image load heart-disease-api:v3
```

Check the image tag in `deployment.yaml`:

```yaml
image: heart-disease-api:v3
imagePullPolicy: Never
```

Make sure the loaded image tag exactly matches the configured Deployment image.

### Issue 2 — Docker permission denied

Check Docker access:

```bash
docker ps
```

If your user needs Docker group membership, an administrator can add it:

```bash
sudo usermod -aG docker "$USER"
```

Log out and back in, or start a new shell with the updated group membership. Docker group membership grants powerful privileges on the machine; use it only on a trusted development system.

### Issue 3 — MLflow rejects the Host header

An error such as `Invalid Host header - possible DNS rebinding attack detected` indicates that the MLflow server rejected the request's host name.

Check the MLflow Tracking Server configuration in `mlflow-stack.yaml`. Configure its supported allowed-hosts setting to include the actual Kubernetes service host names and local development host names you use.

Keep host validation enabled. Do not disable security checks or use unrestricted host wildcards just to bypass the error.

After changing the manifest, apply it and wait for the rollout:

```bash
kubectl apply -f mlflow-stack.yaml
kubectl rollout status deployment/mlflow-tracking
```

### Issue 4 — The API cannot load the champion model

Check that the model exists in the Registry and that its `champion` alias is assigned.

Confirm the API is configured with:

```text
MLFLOW_TRACKING_URI=http://mlflow-tracking:5000
```

Check the API logs:

```bash
kubectl logs deployment/heart-disease-api --tail=100
```

Check the MLflow Tracking Server logs:

```bash
kubectl logs deployment/mlflow-tracking --tail=100
```

### Issue 5 — The API serves an older model version

Changing the `champion` alias alone does not refresh the in-memory model.

Restart the API:

```bash
kubectl rollout restart deployment/heart-disease-api
kubectl rollout status deployment/heart-disease-api
```

Then verify the model version in the prediction response.

### Issue 6 — A Pod is not starting

Inspect the Pods and recent events:

```bash
kubectl get pods
kubectl describe pod <POD_NAME>
kubectl logs <POD_NAME>
```

Replace `<POD_NAME>` with the actual Pod name.

### Issue 7 — MLflow or PostgreSQL data is unavailable

Check the persistent volume claims:

```bash
kubectl get pvc
kubectl describe pvc <PVC_NAME>
```

Do not delete PostgreSQL or artifact-storage volumes as a routine troubleshooting step. They may contain important experiment history and trained model artifacts.

## 15. Security and Configuration

Before deploying outside a local development environment:

- Do not commit passwords, tokens, or other secrets to Git.
- Use Kubernetes Secrets or an appropriate secret-management solution for credentials.
- Restrict access to the MLflow UI and Tracking Server.
- Use HTTPS and suitable authentication for externally accessible endpoints.
- Configure resource requests and limits for Kubernetes workloads.
- Use persistent storage for PostgreSQL and model artifacts.
- Back up important database metadata and artifacts.
- Pin and validate dependency versions.
- Review access permissions for any public API endpoint.
- Do not expose PostgreSQL publicly unless there is a specific, secured requirement.

The example manifests are intended for development and demonstration. A production deployment requires additional security, monitoring, backup, and availability controls.

## 16. Git and Repository Hygiene

Local MLflow databases, model artifacts, environment directories, and backups should generally not be committed to Git.

Recommended `.gitignore` entries:

```gitignore
mlruns/
mlflow.db
mlflow.db-journal
mlflow.db-shm
mlflow.db-wal
backups/
artifacts/
.venv/
venv/
__pycache__/
*.py[cod]
*.log
.env
.env.*
!.env.example
```

Keep source code, Dockerfiles, Kubernetes manifests, dependency files, documentation, and appropriate example configuration in the repository.

If sensitive files were already committed, adding them to `.gitignore` does not remove them from existing Git history. Rotate any exposed credentials and follow the appropriate repository-cleanup procedure.

## 17. Cleanup

To stop the application while retaining its persistent data, remove or scale down the application Deployments as appropriate for your setup.

For a temporary local environment, Minikube can be stopped using:

```bash
minikube stop
```

To start it again:

```bash
minikube start
```

**Warning:** Deleting Kubernetes resources or persistent volume claims can remove data depending on the storage configuration and reclaim policy. Do not delete the PostgreSQL or MLflow artifact PVCs if you want to preserve the registry, experiment history, or models.

## 18. Future Improvements

Potential improvements include:

- Automated model retraining and evaluation.
- CI/CD pipelines for testing and deployment.
- Automatic Kubernetes rollout when a model is promoted.
- Model performance and data-drift monitoring.
- API authentication and authorization.
- Centralized logging and metrics.
- Automated database and artifact backups.
- Deployment to a cloud-hosted Kubernetes environment.
- Integration tests and model validation gates before promotion.

## 19. Conclusion

This project demonstrates a practical MLOps workflow for a heart disease classification application. It integrates machine learning, experiment tracking, model version management, REST API development, containerization, and Kubernetes deployment.

By separating application code, model metadata, and model artifacts, the project provides a foundation for reproducible model management and controlled deployment of updated machine learning models.

---

**Project:** Heart Disease Prediction — MLOps

**API documentation:** `http://localhost:8000/docs`

**MLflow UI:** `http://localhost:5000`

*These URLs are available when the corresponding Kubernetes port-forward commands are running.*
