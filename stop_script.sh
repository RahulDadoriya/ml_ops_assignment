#!/usr/bin/env bash

INFO="[INFO]"
SUCCESS="[SUCCESS]"
ERROR="[ERROR]"

echo "========================================================================="
echo "        HEART DISEASE MLOPS PROJECT: SHUTDOWN ENGINE                     "
echo "========================================================================="

# 1. TERMINATE PORT-FORWARD BACKGROUND OPERATIONS
echo "$INFO Disconnecting API and MLflow network tunnels..."
# Find and terminate background port-forwards listening on 8000 and 5000
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:5000 | xargs kill -9 2>/dev/null || true

# 2. DELETE RUNNING KUBERNETES APPLICATION PODS
echo "$INFO Removing application service deployments from Kubernetes..."
if [ -f "deployment.yaml" ] && [ -f "service.yaml" ]; then
    kubectl delete -f deployment.yaml -f service.yaml --ignore-not-found=true
else
    echo "$ERROR Deployment manifests not found. Skipping application teardown."
fi

# 3. TEARDOWN INFRASTRUCTURE WORKLOADS (Optional but recommended for deep cleanup)
echo "$INFO Removing infrastructure components (FastAPI & MLflow)..."
if [ -f "mlflow-stack.yaml" ]; then
    kubectl delete -f mlflow-stack.yaml --ignore-not-found=true
else
    echo "$ERROR mlflow-stack.yaml not found. Skipping infrastructure teardown."
fi

# Note: The database volumes (PVCs) are safely preserved so model history isn't lost.

# 4. POWER DOWN MINIKUBE ENGINE
echo "$INFO Powering down local Minikube cluster engine..."
if command -v minikube &> /dev/null; then
    minikube stop
else
    echo "$ERROR Minikube command not found."
fi

echo "========================================================================="
echo "   $SUCCESS SHUTDOWN COMPLETE! All background services have been stopped."
echo "========================================================================="
