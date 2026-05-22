#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# setup_venv.sh — 创建 Python 虚拟环境并安装依赖
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
PYTHON="${PYTHON:-python3.10}"
REQ_FILE="${PROJECT_DIR}/requirements.txt"

echo "======================================"
echo "  Game Fraud Detection — Setup venv"
echo "======================================"

# Check Python version >= 3.10
if ! command -v "$PYTHON" &>/dev/null; then
    echo "[ERROR] $PYTHON not found. Please install Python 3.10+ or set PYTHON env var."
    exit 1
fi

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo "[ERROR] Python 3.10+ required, found ${PY_VERSION}"
    exit 1
fi

echo "[INFO] Using Python ${PY_VERSION} (${PYTHON})"

# Create venv
if [ -d "$VENV_DIR" ]; then
    echo "[INFO] Virtual environment already exists: ${VENV_DIR}"
    echo "[INFO] To recreate, delete it first: rm -rf ${VENV_DIR}"
else
    echo "[INFO] Creating virtual environment: ${VENV_DIR}"
    "$PYTHON" -m venv "$VENV_DIR"
    echo "[OK]   Virtual environment created"
fi

# Activate and install
echo "[INFO] Installing dependencies..."
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel -q
pip install --prefer-binary -r "$REQ_FILE"

echo ""
echo "======================================"
echo "  Setup complete!"
echo "======================================"
echo ""
echo "Activate manually:  source ${VENV_DIR}/bin/activate"
echo "Run service:        ./run.sh"
