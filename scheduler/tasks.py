from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from config import settings, TZ
from features.engineering import FeatureEngineer
from model.detector import FraudDetector
from model.storage import ModelStorage
from trace.explainer import RiskExplainer
from listmanager import ListManager


def start_scheduler(app) -> AsyncIOScheduler:
    """Create and start the APScheduler with the daily incremental training job."""

    scheduler = AsyncIOScheduler()

    async def incremental_train_job():
        """Scheduled incremental training with retry on failure."""
        feature_engineer: FeatureEngineer = app.state.feature_engineer
        storage: ModelStorage = app.state.storage
        lm: ListManager = app.state.listmanager

        trace_id = f"train-{uuid.uuid4().hex[:8]}"
        max_retries = settings.TRAINING_MAX_RETRIES
        retry_delay = settings.TRAINING_RETRY_DELAY_SECONDS

        with logger.contextualize(trace_id=trace_id):
            for attempt in range(1, max_retries + 1):
                try:
                    since = datetime.now(TZ) - timedelta(days=settings.FEATURE_WINDOW_DAYS)
                    logger.info(
                        f"Scheduled training started | attempt={attempt}/{max_retries} | "
                        f"window_since={since.isoformat()}"
                    )

                    exclude = lm.list_all("uid", "whitelist") or None
                    features_df = await feature_engineer.extract_features(since=since, exclude_uids=exclude)
                    if features_df.empty:
                        logger.warning("No data in sliding window. Skipping training.")
                        return

                    logger.info(f"Features extracted | users={len(features_df)}")

                    new_detector = FraudDetector()
                    result = await asyncio.to_thread(new_detector.train, features_df)

                    path = await asyncio.to_thread(storage.save, new_detector)
                    logger.info(f"Model saved: {path}")

                    app.state.detector = new_detector
                    app.state.explainer = RiskExplainer(new_detector)

                    removed = await asyncio.to_thread(storage.cleanup, 5)
                    if removed:
                        logger.info(f"Cleaned up {removed} old model file(s)")

                    logger.info(
                        f"Scheduled training complete | version={result['version']} | "
                        f"samples={result['sample_count']}"
                    )
                    return  # success — exit retry loop

                except Exception as e:
                    if attempt < max_retries:
                        delay = retry_delay * (2 ** (attempt - 1))
                        logger.warning(
                            f"Training attempt {attempt}/{max_retries} failed: {e}. "
                            f"Retrying in {delay}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.exception(
                            f"Scheduled training failed after {max_retries} attempts"
                        )

    scheduler.add_job(
        incremental_train_job,
        CronTrigger(
            hour=settings.TRAINING_CRON_HOUR,
            minute=settings.TRAINING_CRON_MINUTE,
        ),
        id="incremental_train",
        name="Daily incremental model training",
        replace_existing=True,
    )

    async def refresh_blocklist_job():
        trace_id = f"blref-{uuid.uuid4().hex[:8]}"
        with logger.contextualize(trace_id=trace_id):
            lm: ListManager = app.state.listmanager
            try:
                await lm.load()
            except Exception:
                logger.exception("Blocklist refresh failed")

    from apscheduler.triggers.interval import IntervalTrigger

    scheduler.add_job(
        refresh_blocklist_job,
        IntervalTrigger(minutes=settings.BLOCKLIST_REFRESH_MINUTES),
        id="blocklist_refresh",
        name="Periodic blocklist refresh",
        replace_existing=True,
    )

    scheduler.start()
    logger.info(
        f"Scheduler started | training runs daily at "
        f"{settings.TRAINING_CRON_HOUR:02d}:{settings.TRAINING_CRON_MINUTE:02d}"
    )
    return scheduler
