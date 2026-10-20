#!/usr/bin/env bash
set -e

INFO="[INFO]"
SUCCESS="[SUCCESS]"
ERROR="[ERROR]"

echo "========================================================================="
echo "   HEART DISEASE MLOPS PROJECT: AUTOMATED ONE-CLICK DEPLOYMENT ENGINE    "
echo "========================================================================="

# 1. PREREQUISITE ENVIRONMENT CHECKS
echo "$INFO Step 1/7: Validating system applications..."
for cmd in python3 docker minikube kubectl git; do
    if ! command -v "$cmd" &> /dev/null; then
        echo "$ERROR '$cmd' is missing from the system."
        exit 1
    fi
done

if ! docker info &> /dev/null; then
    echo "$ERROR Docker daemon is not running."
    exit 1
fi

# 2. LOCAL VIRTUAL ENVIRONMENT & PIP BUILD
echo "$INFO Step 2/7: Creating isolated Python environment..."
[ ! -d ".venv" ] && python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
[ -f "requirements.txt" ] && pip install -r requirements.txt || { echo "$ERROR requirements.txt missing."; exit 1; }

# 3. KUBERNETES ENGINE ORCHESTRATION
echo "$INFO Step 3/7: Initializing local Minikube cluster..."
[ "$(minikube status --format='{{.Host}}' 2>/dev/null)" != "Running" ] && minikube start --driver=docker
docker build -t heart-disease-api:v3 .
docker build -f Dockerfile.mlflow -t heart-disease-mlflow:v1 .
minikube image load heart-disease-api:v3
minikube image load heart-disease-mlflow:v1

# 4. INFRASTRUCTURE INSTANTIATION
echo "$INFO Step 4/7: Launching backend infrastructure..."
kubectl apply -f mlflow-stack.yaml
kubectl rollout status deployment/postgres --timeout=120s
kubectl rollout status deployment/mlflow-tracking --timeout=120s

# 5. AUTOMATED BACKGROUND PORT-FORWARD FOR TRAINING
echo "$INFO Step 5/7: Configuring background channels..."
lsof -ti:5000 | xargs kill -9 2>/dev/null || true
kubectl port-forward service/mlflow-tracking 5000:5000 > /dev/null 2>&1 &
PF_MLFLOW_PID=$!
sleep 3

# 6. PIPELINE TRAINING & MODEL REGISTRATION
echo "$INFO Step 6/7: Commencing automated Model Training Pipeline..."
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
TRAIN_SCRIPT=$(find model/ -name "run_experiment.py" -o -name "train.py" -o -name "*.py" | head -n 1)
if [ -n "$TRAIN_SCRIPT" ]; then
    python "$TRAIN_SCRIPT"
else
    echo "$ERROR No training script found."
    kill $PF_MLFLOW_PID
    exit 1
fi

# 7. SERVICE APPLICATION DEPLOYMENT
echo "$INFO Step 7/7: Deploying FastAPI REST Production API Gateway..."
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl rollout status deployment/heart-disease-api --timeout=120s

lsof -ti:8000 | xargs kill -9 2>/dev/null || true
kubectl port-forward service/heart-disease-api-service 8000:8000 > /dev/null 2>&1 &
PF_API_PID=$!

echo "========================================================================="
echo "   $SUCCESS SETUP COMPLETE! ALL ML INFRASTRUCTURE INTERFACES ARE ONLINE"
echo "========================================================================="
echo " 🌐 MLflow Tracking Dashboard:   http://localhost:5000"
echo " 🌐 FastAPI Interactive Docs:    http://localhost:8000/docs"
echo "========================================================================="

trap "kill $PF_MLFLOW_PID $PF_API_PID 2>/dev/null; exit 0" INT
while true; do sleep 1; done
