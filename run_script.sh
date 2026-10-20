#!/usr/bin/env bash

set -Eeuo pipefail

INFO="[INFO]"
SUCCESS="[SUCCESS]"
WARN="[WARN]"
ERROR="[ERROR]"

# ============================================================
# Heart Disease MLOps - Portable Linux Deployment Script
# Supported: Debian, Ubuntu, Fedora, RHEL, Rocky, AlmaLinux,
#            openSUSE, SUSE, Arch, Manjaro and Alpine.
# Architecture: x86_64/amd64 and aarch64/arm64.
# ============================================================

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

VENV_DIR="$PROJECT_DIR/.venv"
LOCAL_BIN="$HOME/.local/bin"

PF_MLFLOW_PID=""
PF_API_PID=""
TEMP_DIR=""

info()    { printf '%s %s\n' "$INFO" "$*"; }
success() { printf '%s %s\n' "$SUCCESS" "$*"; }
warn()    { printf '%s %s\n' "$WARN" "$*" >&2; }
die()     { printf '%s %s\n' "$ERROR" "$*" >&2; exit 1; }

cleanup() {
    local exit_code=$?

    trap - EXIT INT TERM

    if [[ -n "$PF_API_PID" ]]; then
        kill "$PF_API_PID" 2>/dev/null || true
        wait "$PF_API_PID" 2>/dev/null || true
    fi

    if [[ -n "$PF_MLFLOW_PID" ]]; then
        kill "$PF_MLFLOW_PID" 2>/dev/null || true
        wait "$PF_MLFLOW_PID" 2>/dev/null || true
    fi

    if [[ -n "$TEMP_DIR" && -d "$TEMP_DIR" ]]; then
        rm -rf -- "$TEMP_DIR"
    fi

    if [[ "$exit_code" -ne 0 ]]; then
        warn "Deployment stopped with exit code $exit_code."
        warn "Review the error above, fix the issue, and run this script again."
    fi

    exit "$exit_code"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ------------------------------------------------------------
# OS and architecture detection
# ------------------------------------------------------------

detect_os() {
    if [[ ! -r /etc/os-release ]]; then
        die "Cannot identify Linux distribution: /etc/os-release is missing."
    fi

    # shellcheck disable=SC1091
    . /etc/os-release

    OS_ID="${ID:-unknown}"
    OS_NAME="${PRETTY_NAME:-$OS_ID}"
    OS_FAMILY="${ID_LIKE:-$OS_ID}"

    case "$(uname -m)" in
        x86_64|amd64)
            ARCH="amd64"
            ;;
        aarch64|arm64)
            ARCH="arm64"
            ;;
        *)
            die "Unsupported CPU architecture: $(uname -m). Supported: x86_64 and aarch64."
            ;;
    esac

    info "Detected operating system: $OS_NAME"
    info "Detected architecture: $ARCH"
}

# ------------------------------------------------------------
# Basic utilities and installation guidance
# ------------------------------------------------------------


show_package_hint() {
    local package="$1"

    case " $OS_ID $OS_FAMILY " in
        *" ubuntu "*|*" debian "*|*" linuxmint "*|*" pop "*)
            printf 'Try: sudo apt-get update && sudo apt-get install -y %s\n' "$package"
            ;;
        *" fedora "*|*" rhel "*|*" centos "*|*" rocky "*|*" almalinux "*)
            printf 'Try: sudo dnf install -y %s\n' "$package"
            ;;
        *" opensuse "*|*" opensuse-leap "*|*" sles "*|*" suse "*)
            printf 'Try: sudo zypper install %s\n' "$package"
            ;;
        *" arch "*|*" manjaro "*)
            printf 'Try: sudo pacman -Sy --needed %s\n' "$package"
            ;;
        *" alpine "*)
            printf 'Try: sudo apk add %s\n' "$package"
            ;;
        *)
            printf 'Install "%s" using your operating system package manager.\n' "$package"
            ;;
    esac
}

require_command() {
    local command_name="$1"
    local package_name="${2:-$1}"

    if ! command -v "$command_name" >/dev/null 2>&1; then
        warn "Required command '$command_name' is missing."
        show_package_hint "$package_name"
        die "Install '$command_name' and rerun the script."
    fi
}

# ------------------------------------------------------------
# Install kubectl from the official Kubernetes release endpoint.
# Uses the downloaded SHA-256 checksum before installation.
# ------------------------------------------------------------

install_kubectl() {
    info "kubectl is missing. Attempting official binary installation."

    require_command curl curl
    require_command install coreutils
    require_command mktemp coreutils

    if command -v sha256sum >/dev/null 2>&1; then
        :
    elif command -v shasum >/dev/null 2>&1; then
        :
    else
        die "No SHA-256 verification utility found. Install coreutils or a shasum provider."
    fi

    TEMP_DIR="$(mktemp -d)"
    local version
    local binary_url
    local checksum_url
    local binary_file="$TEMP_DIR/kubectl"
    local checksum_file="$TEMP_DIR/kubectl.sha256"

    version="$(curl -fsSL --retry 3 https://dl.k8s.io/release/stable.txt)" ||
        die "Could not retrieve the stable kubectl version."

    [[ "$version" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] ||
        die "Unexpected kubectl version response: $version"

    binary_url="https://dl.k8s.io/release/$version/bin/linux/$ARCH/kubectl"
    checksum_url="$binary_url.sha256"

    curl -fsSL --retry 3 "$binary_url" -o "$binary_file" ||
        die "Failed to download kubectl."

    curl -fsSL --retry 3 "$checksum_url" -o "$checksum_file" ||
        die "Failed to download the kubectl checksum."

    local expected actual
    expected="$(tr -d '[:space:]' < "$checksum_file")"

    if command -v sha256sum >/dev/null 2>&1; then
        actual="$(sha256sum "$binary_file" | awk '{print $1}')"
    else
        actual="$(shasum -a 256 "$binary_file" | awk '{print $1}')"
    fi

    [[ "$expected" =~ ^[[:xdigit:]]{64}$ ]] ||
        die "The downloaded checksum has an invalid format."

    [[ "$actual" == "$expected" ]] ||
        die "kubectl checksum verification failed. The binary was not installed."

    chmod +x "$binary_file"

    if [[ -w /usr/local/bin ]]; then
        install -m 0755 "$binary_file" /usr/local/bin/kubectl
    elif command -v sudo >/dev/null 2>&1; then
        info "Administrator access is required to install kubectl in /usr/local/bin."
        if sudo -n true 2>/dev/null; then
            sudo install -m 0755 "$binary_file" /usr/local/bin/kubectl
        else
            sudo install -m 0755 "$binary_file" /usr/local/bin/kubectl ||
                die "Could not install kubectl. Grant sudo access or install it manually."
        fi
    else
        mkdir -p "$LOCAL_BIN"
        install -m 0755 "$binary_file" "$LOCAL_BIN/kubectl"
        export PATH="$LOCAL_BIN:$PATH"
        warn "kubectl installed in $LOCAL_BIN. Add this directory to PATH in future shells."
    fi

    rm -rf -- "$TEMP_DIR"
    TEMP_DIR=""

    command -v kubectl >/dev/null 2>&1 ||
        die "kubectl was installed but is not available on PATH. Add its installation directory to PATH."

    success "kubectl installed and verified."
}

# ------------------------------------------------------------
# Install Minikube from its official release endpoint.
# ------------------------------------------------------------

install_minikube() {
    info "Minikube is missing. Attempting official binary installation."

    require_command curl curl
    require_command install coreutils
    require_command mktemp coreutils

    TEMP_DIR="$(mktemp -d)"

    local binary_file="$TEMP_DIR/minikube"
    local binary_url="https://storage.googleapis.com/minikube/releases/latest/minikube-linux-$ARCH"

    curl -fsSL --retry 3 "$binary_url" -o "$binary_file" ||
        die "Failed to download Minikube. Install it manually using the official Minikube documentation."

    chmod +x "$binary_file"

    if [[ -w /usr/local/bin ]]; then
        install -m 0755 "$binary_file" /usr/local/bin/minikube
    elif command -v sudo >/dev/null 2>&1; then
        info "Administrator access is required to install Minikube in /usr/local/bin."
        if sudo -n true 2>/dev/null; then
            sudo install -m 0755 "$binary_file" /usr/local/bin/minikube
        else
            sudo install -m 0755 "$binary_file" /usr/local/bin/minikube ||
                die "Could not install Minikube. Grant sudo access or install it manually."
        fi
    else
        mkdir -p "$LOCAL_BIN"
        install -m 0755 "$binary_file" "$LOCAL_BIN/minikube"
        export PATH="$LOCAL_BIN:$PATH"
        warn "Minikube installed in $LOCAL_BIN. Add this directory to PATH in future shells."
    fi

    rm -rf -- "$TEMP_DIR"
    TEMP_DIR=""

    command -v minikube >/dev/null 2>&1 ||
        die "Minikube was installed but is not available on PATH."

    success "Minikube installed."
}

# ------------------------------------------------------------
# Port and HTTP readiness helpers
# ------------------------------------------------------------

check_port_available() {
    local port="$1"

    if command -v python3 >/dev/null 2>&1; then
        if python3 - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket()
try:
    sock.bind(("127.0.0.1", port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
        then
            return 0
        fi

        die "Port $port is already in use. Stop the application using that port and retry."
    fi

    die "Python 3 is required to check local port availability."
}

wait_for_http() {
    local url="$1"
    local description="$2"
    local attempts="${3:-60}"
    local count=0

    require_command curl curl

    while (( count < attempts )); do
        if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
            success "$description is responding at $url"
            return 0
        fi

        sleep 2
        count=$((count + 1))
    done

    die "$description did not respond at $url within the expected time."
}

start_port_forward() {
    local service="$1"
    local local_port="$2"
    local remote_port="$3"
    local description="$4"
    local log_file="$PROJECT_DIR/.${description}-port-forward.log"

    check_port_available "$local_port"

    info "Starting port-forward for $description..."

    kubectl port-forward \
        --address 127.0.0.1 \
        "service/$service" \
        "$local_port:$remote_port" >"$log_file" 2>&1 &

    local pid=$!

    if [[ "$description" == "mlflow" ]]; then
        PF_MLFLOW_PID="$pid"
    else
        PF_API_PID="$pid"
    fi

    local attempt=0

    while (( attempt < 30 )); do
        if ! kill -0 "$pid" 2>/dev/null; then
            cat "$log_file" >&2 || true
            die "Port-forward for $description exited unexpectedly."
        fi

        if curl -fsS --max-time 2 "http://127.0.0.1:$local_port/" >/dev/null 2>&1; then
            success "$description port-forward is active."
            return 0
        fi

        sleep 1
        attempt=$((attempt + 1))
    done

    cat "$log_file" >&2 || true
    die "Port-forward for $description did not become ready."
}

# ============================================================
# STEP 1: PREREQUISITE ENVIRONMENT CHECKS
# ============================================================

echo "========================================================================="
echo "   HEART DISEASE MLOPS PROJECT: PORTABLE DEPLOYMENT ENGINE              "
echo "========================================================================="

info "Step 1/7: Detecting operating system and checking prerequisites..."

[[ "$(uname -s)" == "Linux" ]] ||
    die "This script targets Linux distributions. For macOS or Windows, use a platform-specific setup."

detect_os

require_command bash bash
require_command python3 python3
require_command git git
require_command docker docker
require_command curl curl

if ! command -v kubectl >/dev/null 2>&1; then
    install_kubectl
fi

if ! command -v minikube >/dev/null 2>&1; then
    install_minikube
fi

kubectl version --client >/dev/null 2>&1 ||
    die "kubectl is installed but cannot run. Check its version and CPU architecture."

minikube version >/dev/null 2>&1 ||
    die "Minikube is installed but cannot run."

docker info >/dev/null 2>&1 || {
    warn "Docker is installed but its daemon is unavailable to this user."
    warn "Start Docker and verify that 'docker info' works without errors."
    die "Docker must be running and accessible before deployment."
}

if [[ "$(docker info --format '{{.OSType}}')" != "linux" ]]; then
    die "Docker must use Linux containers for this project."
fi

for required_file in requirements.txt Dockerfile Dockerfile.mlflow mlflow-stack.yaml deployment.yaml service.yaml; do
    [[ -f "$required_file" ]] ||
        die "Required project file is missing: $required_file"
done

success "Prerequisite checks passed."

# ============================================================
# STEP 2: LOCAL VIRTUAL ENVIRONMENT AND PYTHON DEPENDENCIES
# ============================================================

info "Step 2/7: Creating isolated Python environment..."

if [[ ! -d "$VENV_DIR" ]]; then
    if ! python3 -m venv "$VENV_DIR"; then
        warn "Python virtual-environment support may not be installed."
        case "$OS_ID" in
            ubuntu|debian|linuxmint|pop)
                show_package_hint python3-venv
                ;;
            fedora|rhel|centos|rocky|almalinux)
                show_package_hint python3
                ;;
            opensuse*|sles|suse)
                show_package_hint python3-venv
                ;;
            arch|manjaro)
                show_package_hint python
                ;;
            alpine)
                show_package_hint py3-virtualenv
                ;;
            *)
                warn "Install the Python venv/virtualenv package provided by your distribution."
                ;;
        esac
        die "Could not create the Python virtual environment."
    fi
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

success "Python dependencies installed."

# ============================================================
# STEP 3: KUBERNETES ENGINE ORCHESTRATION
# ============================================================

info "Step 3/7: Initializing local Minikube cluster..."

if [[ "$(minikube status --format='{{.Host}}' 2>/dev/null || true)" != "Running" ]]; then
    minikube start --driver=docker
fi

minikube status ||
    die "Minikube cluster is not healthy."

kubectl config use-context minikube >/dev/null ||
    die "Could not switch kubectl to the Minikube context."

kubectl cluster-info ||
    die "Kubernetes API server is not responding."

info "Building FastAPI image..."
docker build -t heart-disease-api:v3 .

info "Building MLflow image..."
docker build -f Dockerfile.mlflow -t heart-disease-mlflow:v1 .

info "Loading images into Minikube..."
minikube image load heart-disease-api:v3
minikube image load heart-disease-mlflow:v1

success "Minikube is ready and project images are loaded."

# ============================================================
# STEP 4: INFRASTRUCTURE INSTANTIATION
# ============================================================

info "Step 4/7: Launching backend infrastructure..."

kubectl apply -f mlflow-stack.yaml

kubectl rollout status deployment/postgres --timeout=180s
kubectl rollout status deployment/mlflow-tracking --timeout=180s

success "PostgreSQL and MLflow deployments are ready."

# ============================================================
# STEP 5: BACKGROUND PORT-FORWARD FOR TRAINING
# ============================================================

info "Step 5/7: Configuring MLflow access..."

start_port_forward \
    "mlflow-tracking" \
    5000 \
    5000 \
    "mlflow"

wait_for_http "http://127.0.0.1:5000/" "MLflow dashboard" 30


# ============================================================
# STEP 6: MODEL TRAINING AND REGISTRATION
# ============================================================

info "Step 6/7: Running the model training pipeline..."

export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"

info "Running model training from the project root..."

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Select the training script before executing it.
TRAIN_SCRIPT=""

if [[ -f "$PROJECT_DIR/model/run_experiment.py" ]]; then
    TRAIN_SCRIPT="$PROJECT_DIR/model/run_experiment.py"
elif [[ -f "$PROJECT_DIR/model/train.py" ]]; then
    TRAIN_SCRIPT="$PROJECT_DIR/model/train.py"
else
    warn "No preferred training filename found. Searching model/ for Python scripts."

    while IFS= read -r candidate; do
        if [[ "$(basename "$candidate")" != "__init__.py" ]]; then
            TRAIN_SCRIPT="$candidate"
            break
        fi
    done < <(
        find "$PROJECT_DIR/model" \
            -type f \
            -name '*.py' \
            -not -path '*/__pycache__/*' \
            2>/dev/null | sort
    )
fi

# Confirm that a valid training script was found.
if [[ -z "$TRAIN_SCRIPT" || ! -f "$TRAIN_SCRIPT" ]]; then
    die "No Python training script found under model/. Expected run_experiment.py or train.py."
fi

info "Selected training script: ${TRAIN_SCRIPT#"$PROJECT_DIR"/}"

# Execute the selected training script exactly once.
python "$TRAIN_SCRIPT"

success "Model training pipeline completed."


# ============================================================
# STEP 7: FASTAPI DEPLOYMENT
# ============================================================

info "Step 7/7: Deploying the FastAPI REST API..."

kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

kubectl rollout status deployment/heart-disease-api --timeout=180s

start_port_forward \
    "heart-disease-api-service" \
    8000 \
    8000 \
    "api"

wait_for_http "http://127.0.0.1:8000/docs" "FastAPI documentation" 30

echo "========================================================================="
echo "   $SUCCESS SETUP COMPLETE!                                             "
echo "========================================================================="
echo "   Linux distribution: $OS_NAME"
echo "   Kubernetes context: minikube"
echo "   MLflow dashboard:  http://localhost:5000"
echo "   FastAPI docs:      http://localhost:8000/docs"
echo "========================================================================="
echo "   Press Ctrl+C to stop the port-forward processes."
echo "   The Minikube cluster will remain available for future runs."
echo "========================================================================="

# Keep both port-forwards running until the user stops the script.
while true; do
    if [[ -n "$PF_MLFLOW_PID" ]] && ! kill -0 "$PF_MLFLOW_PID" 2>/dev/null; then
        die "The MLflow port-forward stopped unexpectedly."
    fi

    if [[ -n "$PF_API_PID" ]] && ! kill -0 "$PF_API_PID" 2>/dev/null; then
        die "The API port-forward stopped unexpectedly."
    fi

    sleep 2
done
