from __future__ import annotations

import zoneinfo

from pydantic_settings import BaseSettings, SettingsConfigDict

# 中国时间
TZ = zoneinfo.ZoneInfo("Asia/Shanghai")


class Settings(BaseSettings):
    """Central configuration, loaded from environment variables (prefix FRAUD_) or .env file."""

    model_config = SettingsConfigDict(env_prefix="FRAUD_", env_file=".env", extra="ignore")

    # ── MySQL ────────────────────────────────────────────────────────────
    MYSQL_HOST: str = "your-mysql-host"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "your-mysql-user"
    MYSQL_PASSWORD: str = "your-mysql-password"
    MYSQL_DB: str = "your-mysql-database"
    MYSQL_POOL_SIZE: int = 10

    # ── Platforms ───────────────────────────────────────────────────────────
    PLATFORMS: list[str] = []

    # ── Table names ──────────────────────────────────────────────────────
    TABLE_ACTIVATION: str = "your-table-name-activation"
    TABLE_REGISTRATION: str = "your-table-name-registration"
    TABLE_LOGIN: str = "your-table-name-login"
    TABLE_ROLE_CREATION: str = "your-table-name-role-creation"
    TABLE_INGAME_EVENT: str = "your-table-name-ingame-event"
    TABLE_PAYMENT: str = "your-table-name-payment"
    TABLE_BLOCKLIST: str = "your-table-name-blocklist"

    # ── Blocklist ────────────────────────────────────────────────────────
    BLOCKLIST_REFRESH_MINUTES: int = 10

    # ── Query ───────────────────────────────────────────────────────────
    QUERY_BATCH_SIZE: int = 50000

    # ── Feature engineering ──────────────────────────────────────────────
    FEATURE_WINDOW_DAYS: int = 30
    FEATURE_PREDICT_WINDOW_DAYS: int = 2
    CROSS_ACCOUNT_IP_WINDOW_HOURS: int = 24

    # ── Isolation Forest ─────────────────────────────────────────────────
    IF_N_ESTIMATORS: int = 300
    IF_CONTAMINATION: float = 0.02
    IF_MAX_SAMPLES: str = "auto"
    IF_RANDOM_STATE: int = 42

    # ── Scoring ──────────────────────────────────────────────────────────
    ZSCORE_THRESHOLD: float = 3.0
    IF_WEIGHT: float = 0.7
    ZSCORE_WEIGHT: float = 0.3
    RISK_THRESHOLD_HIGH: float = 0.90
    RISK_THRESHOLD_MEDIUM: float = 0.50
    TRACE_TOP_N: int = 5

    # ── Model storage ────────────────────────────────────────────────────
    MODEL_DIR: str = "saved_models"
    MODEL_PREFIX: str = "fraud_detector"

    # ── Scheduler ────────────────────────────────────────────────────────
    TRAINING_CRON_HOUR: int = 3
    TRAINING_CRON_MINUTE: int = 0

    # ── API ──────────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1

    # ── Security ──────────────────────────────────────────────────────────
    SIGN_SECRETS: dict[str, str] = {"test": "your-test-secrets"}   # source → secret 映射，为空时跳过校验
    SIGN_EXPIRE_SECONDS: int = 300      # 签名时间窗口 (秒)


settings = Settings()
