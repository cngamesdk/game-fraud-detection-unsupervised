"""Game Fraud Detection Service - Entry Point."""
from __future__ import annotations

import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from loguru import logger

from config import settings

# ---------------------------------------------------------------------------
# Loguru configuration: all sinks include trace_id for request tracing
# ---------------------------------------------------------------------------

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Ensure trace_id always exists in extra (default "-" for system logs)
def _patch_trace_id(record):
    record["extra"].setdefault("trace_id", "-")

logger.remove()

LOG_FMT_CONSOLE = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{extra[trace_id]}</cyan> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

LOG_FMT_FILE = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {extra[trace_id]} | "
    "{name}:{function}:{line} - {message}"
)

logger.configure(patcher=_patch_trace_id)

# Console
logger.add(sys.stderr, level="INFO", format=LOG_FMT_CONSOLE)

# File: daily rotation
logger.add(
    os.path.join(LOG_DIR, "fraud_{time:YYYY-MM-DD}.log"),
    rotation="00:00",
    retention="7 days",
    level="DEBUG",
    encoding="utf-8",
    format=LOG_FMT_FILE,
)

# File: size rotation
logger.add(
    os.path.join(LOG_DIR, "fraud_latest.log"),
    rotation="100 MB",
    retention="7 days",
    level="DEBUG",
    encoding="utf-8",
    format=LOG_FMT_FILE,
)


def main():
    uvicorn.run(
        "api.app:create_app",
        factory=True,
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        log_level="info",
    )


if __name__ == "__main__":
    main()
