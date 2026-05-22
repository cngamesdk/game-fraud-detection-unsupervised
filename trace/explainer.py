from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from config import settings
from features.registry import FEATURE_DESCRIPTIONS, FEATURE_FRAUD_TYPE_MAPPING

# ── Fraud type descriptions ──────────────────────────────────────────────

FRAUD_TYPE_DESCRIPTIONS: dict[str, str] = {
    "多开": "同一用户或设备运行多个游戏客户端",
    "工作室": "利用多设备批量操作牟利的职业团体",
    "批量注册": "短时间内大量注册账号",
    "脚本/外挂": "使用自动化脚本或修改器",
    "机器人": "自动化操控的非真人账号",
    "盗刷": "使用盗取的支付信息进行消费",
    "洗钱": "通过游戏内交易进行资金清洗",
    "账号交易": "买卖游戏账号",
    "代练": "他人代为游戏操作",
}


class RiskExplainer:
    """Produce human-readable explanations for why a user is flagged as risky."""

    def __init__(self, detector):
        self.detector = detector

    def explain(self, uid: str, user_features: pd.Series, risk_score: float, risk_label: str) -> dict:
        """
        Analyze which features deviate most from the population and map to fraud types.

        Args:
            uid: User ID
            user_features: Series of feature values for this user
            risk_score: Pre-computed risk score
            risk_label: Pre-computed risk label

        Returns:
            Structured explanation dict.
        """
        means = self.detector.feature_means
        medians = self.detector.feature_medians
        stds = self.detector.feature_stds
        feature_names = self.detector.feature_names

        # Compute per-feature Z-scores
        anomalous_features = []
        for feat in feature_names:
            value = float(user_features.get(feat, 0))
            mean = float(means[feat])
            median = float(medians[feat])
            std = float(stds[feat])
            if std < 1e-10:
                z = 0.0
            else:
                z = (value - mean) / std

            anomalous_features.append({
                "feature": feat,
                "description": FEATURE_DESCRIPTIONS.get(feat, feat),
                "user_value": round(value, 4),
                "population_median": round(median, 4),
                "population_mean": round(mean, 4),
                "z_score": round(z, 4),
                "abs_z_score": abs(z),
                "direction": "above" if z > 0 else "below",
                "suspected_fraud_types": FEATURE_FRAUD_TYPE_MAPPING.get(feat, []),
            })

        # Sort by absolute Z-score descending, take top-N
        anomalous_features.sort(key=lambda x: x["abs_z_score"], reverse=True)
        top_features = anomalous_features[: settings.TRACE_TOP_N]

        # Remove internal field
        for f in top_features:
            del f["abs_z_score"]

        # Aggregate suspected fraud types
        type_counter: Counter[str] = Counter()
        for f in top_features:
            for ft in f["suspected_fraud_types"]:
                type_counter[ft] += 1
        primary_risk_types = [t for t, _ in type_counter.most_common()]

        # Generate summary
        summary = self._generate_summary(uid, top_features, primary_risk_types, risk_score, risk_label)

        return {
            "uid": uid,
            "risk_score": round(risk_score, 4),
            "risk_label": risk_label,
            "primary_risk_types": primary_risk_types,
            "top_anomalous_features": top_features,
            "summary": summary,
        }

    def _generate_summary(
        self,
        uid: str,
        top_features: list[dict],
        primary_risk_types: list[str],
        risk_score: float,
        risk_label: str,
    ) -> str:
        """Generate a natural language summary in Chinese."""
        if risk_label == "low":
            return f"用户 {uid} 风险评分 {risk_score:.2f}，行为模式正常，未发现明显异常。"

        risk_type_str = "、".join(primary_risk_types[:3]) if primary_risk_types else "未知类型"
        level_str = "高风险" if risk_label == "high" else "中风险"

        lines = [f"用户 {uid} 被识别为{level_str}(评分 {risk_score:.2f})，疑似{risk_type_str}行为："]

        for feat in top_features[:3]:
            direction_str = "高于" if feat["direction"] == "above" else "低于"
            lines.append(
                f"  - {feat['description']}为 {feat['user_value']:.2f}，"
                f"{direction_str}群体中位数 {feat['population_median']:.2f}"
                f"(Z-score={feat['z_score']:.1f})"
            )

        return "\n".join(lines)
