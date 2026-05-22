from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import Settings, settings, TZ
from features.registry import FEATURE_NAMES, FEATURE_WEIGHTS


class FraudDetector:
    """Isolation Forest + Z-score ensemble for unsupervised fraud detection."""

    def __init__(self, cfg: Settings | None = None):
        self.settings = cfg or settings
        self.isolation_forest: IsolationForest | None = None
        self.feature_means: pd.Series | None = None
        self.feature_medians: pd.Series | None = None
        self.feature_stds: pd.Series | None = None
        self.if_score_min: float = -0.5
        self.if_score_max: float = 0.5
        self.is_fitted: bool = False
        self.last_trained_at: datetime | None = None
        self.training_sample_count: int = 0
        self.feature_names: list[str] = FEATURE_NAMES
        self.version: str | None = None

    # ── Training ─────────────────────────────────────────────────────────

    def train(self, features_df: pd.DataFrame) -> dict:
        """
        Train the ensemble on a feature DataFrame (uid as index, FEATURE_NAMES as columns).
        Returns training metadata.
        """
        # Validate columns
        missing = set(self.feature_names) - set(features_df.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")

        X = features_df[self.feature_names].values

        # Store population statistics for Z-score and traceability
        self.feature_means = features_df[self.feature_names].mean()
        self.feature_medians = features_df[self.feature_names].median()
        self.feature_stds = features_df[self.feature_names].std().replace(0, 1e-10)

        # Fit Isolation Forest
        self.isolation_forest = IsolationForest(
            n_estimators=self.settings.IF_N_ESTIMATORS,
            contamination=self.settings.IF_CONTAMINATION,
            max_samples=self.settings.IF_MAX_SAMPLES,
            random_state=self.settings.IF_RANDOM_STATE,
            n_jobs=-1,
        )
        self.isolation_forest.fit(X)

        # Store score range from training data for normalization
        raw_scores = self.isolation_forest.decision_function(X)
        self.if_score_min = float(raw_scores.min())
        self.if_score_max = float(raw_scores.max())

        # Metadata
        now = datetime.now(TZ)
        self.is_fitted = True
        self.last_trained_at = now
        self.training_sample_count = len(features_df)
        self.version = now.strftime("%Y%m%d_%H%M%S")

        return {
            "trained_at": now.isoformat(),
            "sample_count": self.training_sample_count,
            "version": self.version,
            "feature_count": len(self.feature_names),
        }

    # ── Prediction ───────────────────────────────────────────────────────

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict risk for one or more users.

        Returns DataFrame with: uid, risk_score, risk_label, if_score, zscore_component
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not trained. Call train() first.")

        X = features_df[self.feature_names].values
        uids = [str(u) for u in features_df.index]

        # Isolation Forest raw scores (more negative = more anomalous)
        raw_scores = self.isolation_forest.decision_function(X)

        # Normalize to [0, 1] where higher = more anomalous
        score_range = self.if_score_max - self.if_score_min
        if score_range == 0:
            score_range = 1e-10
        if_scores = 1.0 - (raw_scores - self.if_score_min) / score_range
        if_scores = np.clip(if_scores, 0.0, 1.0)

        # Z-score component
        zscore_components = self._compute_zscore_component(features_df)

        # Combined score
        combined = (
            self.settings.IF_WEIGHT * if_scores
            + self.settings.ZSCORE_WEIGHT * zscore_components
        )
        combined = np.clip(combined, 0.0, 1.0)

        # Risk labels
        labels = []
        for score in combined:
            if score >= self.settings.RISK_THRESHOLD_HIGH:
                labels.append("high")
            elif score >= self.settings.RISK_THRESHOLD_MEDIUM:
                labels.append("medium")
            else:
                labels.append("low")

        return pd.DataFrame({
            "uid": uids,
            "risk_score": np.round(combined, 4),
            "risk_label": labels,
            "if_score": np.round(if_scores, 4),
            "zscore_component": np.round(zscore_components, 4),
        })

    def _compute_zscore_component(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        For each user, compute per-feature Z-scores weighted by FEATURE_WEIGHTS,
        take the mean of the top-5 absolute Z-scores, and normalize to [0, 1].
        """
        X = features_df[self.feature_names].values
        means = self.feature_means.values
        stds = self.feature_stds.values
        weights = np.array(FEATURE_WEIGHTS, dtype=float)

        # Weighted Z-scores for all features: shape (n_users, n_features)
        zscores = np.abs((X - means) / stds) * weights

        # Top-K Z-scores per user
        top_k = min(self.settings.TRACE_TOP_N, zscores.shape[1])
        top_zscores = np.sort(zscores, axis=1)[:, -top_k:]
        mean_top_z = top_zscores.mean(axis=1)

        # Normalize: clip to [0, 10] then divide by 10
        return np.clip(mean_top_z / 10.0, 0.0, 1.0)

    # ── Status ───────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        return {
            "is_fitted": self.is_fitted,
            "last_trained_at": self.last_trained_at.isoformat() if self.last_trained_at else None,
            "training_sample_count": self.training_sample_count,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "version": self.version,
            "config": {
                "if_n_estimators": self.settings.IF_N_ESTIMATORS,
                "if_contamination": self.settings.IF_CONTAMINATION,
                "risk_threshold_high": self.settings.RISK_THRESHOLD_HIGH,
                "risk_threshold_medium": self.settings.RISK_THRESHOLD_MEDIUM,
                "if_weight": self.settings.IF_WEIGHT,
                "zscore_weight": self.settings.ZSCORE_WEIGHT,
            },
        }
