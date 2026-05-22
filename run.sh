#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# run.sh — 一键启动风控检测服务（后台运行）
#
# 功能:
#   1. 自动检测并创建虚拟环境（首次运行）
#   2. 激活虚拟环境
#   3. 安装/更新依赖
#   4. 后台启动 FastAPI 服务，输出写入日志文件
#
# 用法:
#   ./run.sh              # 默认启动 (0.0.0.0:8000)
#   ./run.sh --port 9000  # 指定端口
#   ./run.sh --host 127.0.0.1 --port 9000
#   ./run.sh --reload     # 开发模式（前台运行，代码修改自动重载）
#   ./run.sh stop         # 停止服务
#   ./run.sh status       # 查看服务状态
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
PYTHON="${PYTHON:-python3}"
REQ_FILE="${PROJECT_DIR}/requirements.txt"
LOG_DIR="${PROJECT_DIR}/logs"
PID_FILE="${PROJECT_DIR}/.service.pid"
STDOUT_LOG="${LOG_DIR}/stdout.log"

mkdir -p "$LOG_DIR"

# ── Helper: stop ─────────────────────────────────────────────────────────
do_stop() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "[INFO] Stopping service (PID: $PID)..."
            kill "$PID"
            sleep 1
            # Force kill if still alive
            if kill -0 "$PID" 2>/dev/null; then
                kill -9 "$PID" 2>/dev/null || true
            fi
            echo "[OK]   Service stopped"
        else
            echo "[INFO] Process $PID not running"
        fi
        rm -f "$PID_FILE"
    else
        echo "[INFO] No PID file found, service not running"
    fi
}

# ── Helper: status ───────────────────────────────────────────────────────
do_status() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "[OK]   Service is running (PID: $PID)"
            return 0
        else
            echo "[WARN] PID file exists but process $PID not running"
            rm -f "$PID_FILE"
            return 1
        fi
    else
        echo "[INFO] Service is not running"
        return 1
    fi
}

# ── Handle stop / status commands ────────────────────────────────────────
if [[ "${1:-}" == "stop" ]]; then
    do_stop
    exit 0
fi

if [[ "${1:-}" == "status" ]]; then
    do_status
    exit $?
fi

# ── Parse arguments ──────────────────────────────────────────────────────
HOST=""
PORT=""
RELOAD=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            HOST="$2"; shift 2 ;;
        --port)
            PORT="$2"; shift 2 ;;
        --reload)
            RELOAD="1"; shift ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ── Banner ───────────────────────────────────────────────────────────────
echo "======================================"
echo "  Game Fraud Detection Service"
echo "======================================"

# ── Step 1: Ensure virtual environment exists ────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Virtual environment not found, running setup..."
    bash "${PROJECT_DIR}/setup_venv.sh"
fi

# ── Step 2: Activate virtual environment ─────────────────────────────────
echo "[INFO] Activating virtual environment..."
source "${VENV_DIR}/bin/activate"

# ── Step 3: Check dependencies are up to date ───────────────────────────
STAMP_FILE="${VENV_DIR}/.deps_installed"
if [ ! -f "$STAMP_FILE" ] || [ "$REQ_FILE" -nt "$STAMP_FILE" ]; then
    echo "[INFO] Installing/updating dependencies..."
    pip install --upgrade pip setuptools wheel -q
    pip install --prefer-binary -r "$REQ_FILE" -q
    touch "$STAMP_FILE"
    echo "[OK]   Dependencies up to date"
else
    echo "[OK]   Dependencies already up to date"
fi

# ── Step 4: Build env overrides from CLI args ────────────────────────────
if [ -n "$HOST" ]; then
    export FRAUD_API_HOST="$HOST"
fi
if [ -n "$PORT" ]; then
    export FRAUD_API_PORT="$PORT"
fi

DISPLAY_HOST="${HOST:-0.0.0.0}"
DISPLAY_PORT="${PORT:-8000}"

# ── Step 5: Stop existing instance if running ────────────────────────────
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[INFO] Stopping old instance (PID: $OLD_PID)..."
        kill "$OLD_PID"
        sleep 1
        kill -0 "$OLD_PID" 2>/dev/null && kill -9 "$OLD_PID" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
fi

# ── Step 6: Start the service ────────────────────────────────────────────
cd "$PROJECT_DIR"

if [ -n "$RELOAD" ]; then
    # Dev mode: foreground with auto-reload
    echo ""
    echo "[INFO] Development mode (foreground, auto-reload)"
    echo "[INFO] http://${DISPLAY_HOST}:${DISPLAY_PORT}"
    echo "[INFO] Press Ctrl+C to stop"
    echo "======================================"
    exec python -m uvicorn api.app:create_app --factory \
        --host "${DISPLAY_HOST}" \
        --port "${DISPLAY_PORT}" \
        --reload \
        --log-level info \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
else
    # Production mode: background with log output
    echo ""
    echo "[INFO] Starting in background..."
    echo "[INFO] http://${DISPLAY_HOST}:${DISPLAY_PORT}"
    echo "[INFO] Log file: ${STDOUT_LOG}"

    nohup python main.py "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
        >> "$STDOUT_LOG" 2>&1 &

    PID=$!
    echo "$PID" > "$PID_FILE"

    # Wait briefly and check if process is still alive
    sleep 2
    if kill -0 "$PID" 2>/dev/null; then
        echo "[OK]   Service started (PID: $PID)"
        echo "[INFO] Use './run.sh stop' to stop"
        echo "[INFO] Use './run.sh status' to check"
        echo "[INFO] Use 'tail -f ${STDOUT_LOG}' to follow logs"
    else
        echo "[ERROR] Service failed to start, check ${STDOUT_LOG}"
        rm -f "$PID_FILE"
        exit 1
    fi
    echo "======================================"
fi
