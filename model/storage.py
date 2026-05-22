from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import joblib
from loguru import logger

from config import settings, TZ


class ModelStorage:
    """Versioned model persistence with joblib."""

    def __init__(
        self,
        model_dir: str | None = None,
        prefix: str | None = None,
    ):
        self.model_dir = Path(model_dir or settings.MODEL_DIR)
        self.prefix = prefix or settings.MODEL_PREFIX
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def save(self, detector) -> str:
        """
        Serialize the detector and create a 'latest' symlink.
        Returns the path to the saved file.
        """
        version = detector.version or datetime.now(TZ).strftime("%Y%m%d_%H%M%S")
        filename = f"{self.prefix}_{version}.joblib"
        filepath = self.model_dir / filename

        # Serialize all state needed to reconstruct the detector
        state = {
            "isolation_forest": detector.isolation_forest,
            "feature_means": detector.feature_means,
            "feature_medians": detector.feature_medians,
            "feature_stds": detector.feature_stds,
            "if_score_min": detector.if_score_min,
            "if_score_max": detector.if_score_max,
            "is_fitted": detector.is_fitted,
            "last_trained_at": detector.last_trained_at,
            "training_sample_count": detector.training_sample_count,
            "feature_names": detector.feature_names,
            "version": detector.version,
        }
        joblib.dump(state, filepath)
        logger.info(f"Model saved: {filepath}")

        # Update 'latest' symlink
        latest_link = self.model_dir / "latest.joblib"
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(filepath.name)

        return str(filepath)

    def load_latest(self):
        """
        Load the model pointed to by 'latest.joblib'.
        Returns a FraudDetector instance or None.
        """
        latest_link = self.model_dir / "latest.joblib"
        if not latest_link.exists():
            return None
        return self._load_from_path(latest_link)

    def load_version(self, version: str):
        """Load a specific model version."""
        filepath = self.model_dir / f"{self.prefix}_{version}.joblib"
        if not filepath.exists():
            raise FileNotFoundError(f"Model version not found: {filepath}")
        return self._load_from_path(filepath)

    def list_versions(self) -> list[dict]:
        """List all saved model versions with metadata."""
        versions = []
        for f in sorted(self.model_dir.glob(f"{self.prefix}_*.joblib")):
            try:
                state = joblib.load(f)
                versions.append({
                    "version": state.get("version", "unknown"),
                    "file": f.name,
                    "trained_at": (
                        state["last_trained_at"].isoformat()
                        if state.get("last_trained_at")
                        else None
                    ),
                    "sample_count": state.get("training_sample_count", 0),
                    "size_bytes": f.stat().st_size,
                })
            except Exception as e:
                logger.warning(f"Failed to read model file {f}: {e}")
        return versions

    def cleanup(self, keep_n: int = 5) -> int:
        """Remove old model files, keeping the N most recent. Returns count removed."""
        files = sorted(
            self.model_dir.glob(f"{self.prefix}_*.joblib"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        removed = 0
        for f in files[keep_n:]:
            f.unlink()
            removed += 1
            logger.info(f"Removed old model: {f.name}")
        return removed

    def _load_from_path(self, filepath: Path):
        """Reconstruct a FraudDetector from a saved state dict."""
        from model.detector import FraudDetector
        from features.registry import FEATURE_NAMES

        state = joblib.load(filepath)

        saved_features = state.get("feature_names", [])
        if saved_features != FEATURE_NAMES:
            logger.warning(
                f"Feature mismatch in {filepath.name}: "
                f"saved {len(saved_features)} features, current {len(FEATURE_NAMES)}. "
                f"Model outdated, retrain required."
            )
            return None

        detector = FraudDetector()
        detector.isolation_forest = state["isolation_forest"]
        detector.feature_means = state["feature_means"]
        detector.feature_medians = state["feature_medians"]
        detector.feature_stds = state["feature_stds"]
        detector.if_score_min = state["if_score_min"]
        detector.if_score_max = state["if_score_max"]
        detector.is_fitted = state["is_fitted"]
        detector.last_trained_at = state["last_trained_at"]
        detector.training_sample_count = state["training_sample_count"]
        detector.feature_names = state["feature_names"]
        detector.version = state["version"]
        logger.info(f"Model loaded: {filepath} (version={detector.version})")
        return detector
