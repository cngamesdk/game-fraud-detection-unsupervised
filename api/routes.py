from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import aiomysql
from fastapi import APIRouter, HTTPException, Query, Request
from loguru import logger

from config import settings, TZ
from db.connection import execute_query
from db import queries
from features.engineering import FeatureEngineer
from model.detector import FraudDetector
from model.storage import ModelStorage
from trace.explainer import RiskExplainer

from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    BlocklistEntry,
    BlocklistRequest,
    BlocklistResponse,
    DevicePredictRequest,
    DevicePredictResponse,
    EntryType,
    ErrorResponse,
    IpPredictRequest,
    IpPredictResponse,
    ListType,
    ModelStatusResponse,
    PredictRequest,
    PredictResponse,
    TraceResponse,
    TrainRequest,
    TrainResponse,
)

router = APIRouter(prefix="/api/v1", tags=["fraud-detection"])


def _get_state(request: Request):
    """Extract shared objects from app.state."""
    return (
        request.app.state.detector,
        request.app.state.feature_engineer,
        request.app.state.explainer,
        request.app.state.storage,
    )


def _whitelisted_response(uid: str) -> PredictResponse:
    return PredictResponse(
        uid=uid, risk_score=0.0, risk_label="low",
        if_score=0.0, zscore_component=0.0, is_whitelisted=True,
    )


def _blacklisted_response(uid: str) -> PredictResponse:
    return PredictResponse(
        uid=uid, risk_score=1.0, risk_label="high",
        if_score=1.0, zscore_component=1.0, is_blacklisted=True,
    )


def _build_predict_response(
    row, features_df, explainer: RiskExplainer, detector: FraudDetector, **extra,
) -> PredictResponse:
    """Build PredictResponse with trace info when model is fitted."""
    uid = row["uid"]
    trace_fields: dict = {}
    if detector.is_fitted and detector.feature_means is not None and uid in features_df.index:
        try:
            explanation = explainer.explain(
                uid=uid,
                user_features=features_df.loc[uid],
                risk_score=float(row["risk_score"]),
                risk_label=row["risk_label"],
            )
            trace_fields = {
                "primary_risk_types": explanation["primary_risk_types"],
                "top_anomalous_features": explanation["top_anomalous_features"],
                "summary": explanation["summary"],
            }
        except Exception:
            logger.debug(f"Trace explain failed for uid={uid}", exc_info=True)

    return PredictResponse(
        uid=uid,
        risk_score=row["risk_score"],
        risk_label=row["risk_label"],
        if_score=row["if_score"],
        zscore_component=row["zscore_component"],
        **trace_fields,
        **extra,
    )


# ── POST /predict ────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={503: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def predict(body: PredictRequest, request: Request):
    """Predict risk for a single user."""
    lm = request.app.state.listmanager

    if lm.is_blacklisted(body.uid, "uid"):
        return _blacklisted_response(body.uid)
    if lm.is_whitelisted(body.uid, "uid"):
        return _whitelisted_response(body.uid)

    detector, feature_engineer, explainer, _ = _get_state(request)

    if not detector.is_fitted:
        raise HTTPException(503, detail="模型未训练，请先调用 POST /api/v1/train")

    features_df = await feature_engineer.extract_features(uids=[body.uid])
    if features_df.empty:
        raise HTTPException(404, detail=f"未找到用户 {body.uid} 的行为日志")

    result_df = detector.predict(features_df)
    row = result_df.iloc[0]
    return _build_predict_response(row, features_df, explainer, detector)


# ── POST /batch-predict ─────────────────────────────────────────────────

@router.post(
    "/batch-predict",
    response_model=BatchPredictResponse,
    responses={503: {"model": ErrorResponse}},
)
async def batch_predict(body: BatchPredictRequest, request: Request):
    """Predict risk for up to 500 users."""
    detector, feature_engineer, explainer, _ = _get_state(request)
    lm = request.app.state.listmanager

    if not detector.is_fitted:
        raise HTTPException(503, detail="模型未训练，请先调用 POST /api/v1/train")

    wl_uids, bl_uids, normal_uids = lm.filter_uids(body.uids)

    results: list[PredictResponse] = (
        [_whitelisted_response(u) for u in wl_uids]
        + [_blacklisted_response(u) for u in bl_uids]
    )

    if normal_uids:
        features_df = await feature_engineer.extract_features(uids=normal_uids)
        if not features_df.empty:
            result_df = detector.predict(features_df)
            results.extend(
                _build_predict_response(row, features_df, explainer, detector)
                for _, row in result_df.iterrows()
            )

    high = sum(1 for r in results if r.risk_label == "high")
    medium = sum(1 for r in results if r.risk_label == "medium")
    low = sum(1 for r in results if r.risk_label == "low")
    return BatchPredictResponse(
        results=results, total=len(results),
        high_risk_count=high, medium_risk_count=medium, low_risk_count=low,
    )


# ── POST /predict-device ────────────────────────────────────────────────

@router.post(
    "/predict-device",
    response_model=DevicePredictResponse,
    responses={503: {"model": ErrorResponse}},
)
async def predict_device(body: DevicePredictRequest, request: Request):
    """Predict risk for all UIDs associated with a device fingerprint."""
    detector, feature_engineer, _, _ = _get_state(request)
    lm = request.app.state.listmanager

    device_fp = body.compute_device_fp()

    is_wl = lm.is_whitelisted(device_fp, "device")
    is_bl = lm.is_blacklisted(device_fp, "device")

    if is_bl:
        return DevicePredictResponse(
            device_fp=device_fp, is_blacklisted=True,
            associated_uids=0,
            high_risk_count=0, medium_risk_count=0, low_risk_count=0,
        )
    if is_wl:
        return DevicePredictResponse(
            device_fp=device_fp, is_whitelisted=True,
            associated_uids=0,
            high_risk_count=0, medium_risk_count=0, low_risk_count=0,
        )

    if not detector.is_fitted:
        raise HTTPException(503, detail="模型未训练，请先调用 POST /api/v1/train")

    since = datetime.now(TZ) - timedelta(days=settings.FEATURE_WINDOW_DAYS)
    sql, params = queries.query_uids_by_device(since, device_fp)
    rows = await execute_query(sql, params)
    uids = [str(r["uid"]) for r in rows if r.get("uid") is not None]

    high, medium, low = 0, 0, 0
    if uids:
        features_df = await feature_engineer.extract_features(uids=uids)
        if not features_df.empty:
            result_df = detector.predict(features_df)
            labels = result_df["risk_label"].value_counts()
            high = int(labels.get("high", 0))
            medium = int(labels.get("medium", 0))
            low = int(labels.get("low", 0))

    return DevicePredictResponse(
        device_fp=device_fp, associated_uids=len(uids),
        high_risk_count=high, medium_risk_count=medium, low_risk_count=low,
    )


# ── POST /predict-ip ────────────────────────────────────────────────────

@router.post(
    "/predict-ip",
    response_model=IpPredictResponse,
    responses={503: {"model": ErrorResponse}},
)
async def predict_ip(body: IpPredictRequest, request: Request):
    """Predict risk for all UIDs associated with an IP address."""
    detector, feature_engineer, _, _ = _get_state(request)
    lm = request.app.state.listmanager

    is_wl = lm.is_whitelisted(body.ip, "ip")
    is_bl = lm.is_blacklisted(body.ip, "ip")

    if is_bl:
        return IpPredictResponse(
            ip=body.ip, is_blacklisted=True,
            associated_uids=0,
            high_risk_count=0, medium_risk_count=0, low_risk_count=0,
        )
    if is_wl:
        return IpPredictResponse(
            ip=body.ip, is_whitelisted=True,
            associated_uids=0,
            high_risk_count=0, medium_risk_count=0, low_risk_count=0,
        )

    if not detector.is_fitted:
        raise HTTPException(503, detail="模型未训练，请先调用 POST /api/v1/train")

    since = datetime.now(TZ) - timedelta(days=settings.FEATURE_WINDOW_DAYS)
    sql, params = queries.query_uids_by_ip(since, body.ip)
    rows = await execute_query(sql, params)
    uids = [str(r["uid"]) for r in rows if r.get("uid") is not None]

    high, medium, low = 0, 0, 0
    if uids:
        features_df = await feature_engineer.extract_features(uids=uids)
        if not features_df.empty:
            result_df = detector.predict(features_df)
            labels = result_df["risk_label"].value_counts()
            high = int(labels.get("high", 0))
            medium = int(labels.get("medium", 0))
            low = int(labels.get("low", 0))

    return IpPredictResponse(
        ip=body.ip, associated_uids=len(uids),
        high_risk_count=high, medium_risk_count=medium, low_risk_count=low,
    )


# ── POST /train ──────────────────────────────────────────────────────────

@router.post("/train", response_model=TrainResponse)
async def train(body: TrainRequest, request: Request):
    """Trigger model training (synchronous)."""
    detector, feature_engineer, _, storage = _get_state(request)
    lm = request.app.state.listmanager

    if body.full_retrain or not detector.is_fitted:
        since = datetime.now(TZ) - timedelta(days=settings.FEATURE_WINDOW_DAYS)
    else:
        since = detector.last_trained_at or (
            datetime.now(TZ) - timedelta(days=settings.FEATURE_WINDOW_DAYS)
        )

    logger.info(f"Training triggered (full_retrain={body.full_retrain}), since={since}")

    exclude = lm.list_all("uid", "whitelist") or None
    features_df = await feature_engineer.extract_features(since=since, exclude_uids=exclude)
    if features_df.empty:
        raise HTTPException(400, detail="指定时间窗口内无数据，无法训练")

    new_detector = FraudDetector()
    result = await asyncio.to_thread(new_detector.train, features_df)
    await asyncio.to_thread(storage.save, new_detector)

    request.app.state.detector = new_detector
    request.app.state.explainer = RiskExplainer(new_detector)

    logger.info(f"Training complete: version={result['version']}, samples={result['sample_count']}")

    return TrainResponse(
        status="success",
        trained_at=result["trained_at"],
        sample_count=result["sample_count"],
        version=result["version"],
        feature_count=result["feature_count"],
    )


# ── GET /trace/{uid} ────────────────────────────────────────────────────

@router.get(
    "/trace/{uid}",
    response_model=TraceResponse,
    responses={503: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def trace(uid: str, request: Request):
    """Get detailed risk explanation for a user."""
    detector, feature_engineer, explainer, _ = _get_state(request)
    lm = request.app.state.listmanager

    if not detector.is_fitted:
        raise HTTPException(503, detail="模型未训练，请先调用 POST /api/v1/train")

    features_df = await feature_engineer.extract_features(uids=[uid])
    if features_df.empty:
        raise HTTPException(404, detail=f"未找到用户 {uid} 的行为日志")

    result_df = detector.predict(features_df)
    row = result_df.iloc[0]

    user_features = features_df.iloc[0]
    explanation = explainer.explain(
        uid=uid,
        user_features=user_features,
        risk_score=float(row["risk_score"]),
        risk_label=row["risk_label"],
    )

    return TraceResponse(
        **explanation,
        is_whitelisted=lm.is_whitelisted(uid, "uid"),
        is_blacklisted=lm.is_blacklisted(uid, "uid"),
    )


# ── GET /model/status ────────────────────────────────────────────────────

@router.get("/model/status", response_model=ModelStatusResponse)
async def model_status(request: Request):
    """Return current model metadata."""
    detector, _, _, _ = _get_state(request)
    return ModelStatusResponse(**detector.get_status())


# ── Blocklist management ─────────────────────────────────────────────────

@router.get("/blocklist", response_model=BlocklistResponse, tags=["blocklist"])
async def get_blocklist(
    request: Request,
    type: EntryType | None = Query(None),
    list_type: ListType | None = Query(None),
):
    """List blocklist entries, optionally filtered by type and list_type."""
    import json as _json
    from listmanager import _compute_key

    sql = f"SELECT `value`, `type`, list_type, platform FROM {settings.TABLE_BLOCKLIST}"
    conditions, params = [], []
    if type is not None:
        conditions.append("`type` = %s")
        params.append(type.value)
    if list_type is not None:
        conditions.append("list_type = %s")
        params.append(list_type.value)
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    rows = await execute_query(sql, tuple(params) if params else None)

    items = []
    for r in rows:
        try:
            data = _json.loads(r["value"])
        except (_json.JSONDecodeError, TypeError):
            continue
        items.append(BlocklistEntry(
            key=_compute_key(r["type"], data),
            type=r["type"],
            list_type=r["list_type"],
            platform=r.get("platform", ""),
            value=data,
        ))
    return BlocklistResponse(total=len(items), items=items)


@router.post("/blocklist", response_model=BlocklistResponse, tags=["blocklist"])
async def add_blocklist(body: BlocklistRequest, request: Request):
    """Add an entry to the blocklist."""
    lm = request.app.state.listmanager
    data = body.to_value_dict()
    from listmanager import _compute_key
    key = _compute_key(body.type.value, data)
    if not key:
        raise HTTPException(400, detail="值不能为空")
    await lm.add(body.type.value, body.list_type.value, data, body.platform)
    return await get_blocklist(request, type=body.type, list_type=body.list_type)


@router.delete("/blocklist", response_model=BlocklistResponse, tags=["blocklist"])
async def remove_blocklist(
    request: Request,
    key: str = Query(..., description="查找键 (uid/设备指纹/IP)"),
    type: EntryType = Query(...),
    list_type: ListType = Query(...),
):
    """Remove an entry from the blocklist."""
    lm = request.app.state.listmanager
    removed = await lm.remove(key, type.value, list_type.value)
    if not removed:
        raise HTTPException(404, detail=f"{key} 不在 {type.value} {list_type.value} 中")
    return await get_blocklist(request, type=type, list_type=list_type)
