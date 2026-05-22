from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


# ── Enums ────────────────────────────────────────────────────────────────

class RiskLabel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EntryType(str, Enum):
    UID = "uid"
    DEVICE = "device"
    IP = "ip"


class ListType(str, Enum):
    WHITELIST = "whitelist"
    BLACKLIST = "blacklist"


# ── Requests ─────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    uid: str = Field(..., description="用户ID")

    @field_validator("uid", mode="before")
    @classmethod
    def _coerce_uid(cls, v):
        return str(v)


class BatchPredictRequest(BaseModel):
    uids: list[str] = Field(..., description="用户ID列表", max_length=500)

    @field_validator("uids", mode="before")
    @classmethod
    def _coerce_uids(cls, v):
        return [str(i) for i in v]


class DevicePredictRequest(BaseModel):
    """设备预测：按维度填写设备信息，后端组装指纹。"""
    imei: str = Field(default="", description="IMEI")
    oaid: str = Field(default="", description="OAID")
    idfv: str = Field(default="", description="IDFV")
    device_brand: str = Field(default="", description="设备品牌")
    device_model: str = Field(default="", description="设备型号")
    os: str = Field(default="", description="操作系统")
    system_version: str = Field(default="", description="系统版本")
    split_ua: str = Field(default="", description="User Agent")

    def compute_device_fp(self) -> str:
        """与数据库 DEVICE_FINGERPRINT_EXPR 保持一致的指纹计算逻辑。"""
        import hashlib
        # COALESCE(NULLIF(imei,''), NULLIF(oaid,''), NULLIF(idfv,''),
        #   MD5(CONCAT(device_brand, device_model, os, system_version, split_ua)))
        for v in (self.imei, self.oaid, self.idfv):
            if v.strip():
                return v.strip()
        raw = (self.device_brand or "") + (self.device_model or "") + (self.os or "") + (self.system_version or "") + (self.split_ua or "")
        return hashlib.md5(raw.encode("utf-8")).hexdigest()


class IpPredictRequest(BaseModel):
    ip: str = Field(..., description="IPv4 地址")


class TrainRequest(BaseModel):
    full_retrain: bool = Field(
        default=False,
        description="True=使用完整窗口重训; False=增量训练(默认)",
    )


# ── Responses ────────────────────────────────────────────────────────────

class AnomalousFeature(BaseModel):
    feature: str
    description: str
    user_value: float
    population_median: float
    population_mean: float
    z_score: float
    direction: str
    suspected_fraud_types: list[str]


class PredictResponse(BaseModel):
    uid: str
    risk_score: float = Field(..., ge=0, le=1)
    risk_label: RiskLabel
    if_score: float
    zscore_component: float
    is_whitelisted: bool = False
    is_blacklisted: bool = False
    primary_risk_types: list[str] = Field(default_factory=list)
    top_anomalous_features: list[AnomalousFeature] = Field(default_factory=list)
    summary: str = ""


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    total: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int


class DevicePredictResponse(BaseModel):
    device_fp: str
    is_whitelisted: bool = False
    is_blacklisted: bool = False
    associated_uids: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int


class IpPredictResponse(BaseModel):
    ip: str
    is_whitelisted: bool = False
    is_blacklisted: bool = False
    associated_uids: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int


class TraceResponse(BaseModel):
    uid: str
    risk_score: float
    risk_label: RiskLabel
    primary_risk_types: list[str]
    top_anomalous_features: list[AnomalousFeature]
    summary: str
    is_whitelisted: bool = False
    is_blacklisted: bool = False


class ModelStatusResponse(BaseModel):
    is_fitted: bool
    last_trained_at: str | None
    training_sample_count: int
    feature_count: int
    feature_names: list[str]
    version: str | None
    config: dict


class TrainResponse(BaseModel):
    status: str
    trained_at: str
    sample_count: int
    version: str
    feature_count: int


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None


# ── Blocklist management ─────────────────────────────────────────────────

class BlocklistRequest(BaseModel):
    type: EntryType = Field(..., description="uid | device | ip")
    list_type: ListType = Field(..., description="whitelist | blacklist")
    platform: str = Field(default="", description="平台标识")
    # uid / ip 直接填值
    uid: str = Field(default="", description="用户ID (type=uid 时)")
    ip: str = Field(default="", description="IP地址 (type=ip 时)")
    # 设备维度字段 (type=device 时)
    imei: str = Field(default="", description="IMEI")
    oaid: str = Field(default="", description="OAID")
    idfv: str = Field(default="", description="IDFV")
    device_brand: str = Field(default="", description="设备品牌")
    device_model: str = Field(default="", description="设备型号")
    os: str = Field(default="", description="操作系统")
    system_version: str = Field(default="", description="系统版本")
    split_ua: str = Field(default="", description="User Agent")

    def to_value_dict(self) -> dict:
        """构建存入 DB value 列的 JSON dict。"""
        if self.type == EntryType.UID:
            return {"uid": str(self.uid).strip()}
        elif self.type == EntryType.IP:
            return {"ip": self.ip.strip()}
        else:
            return {
                "imei": self.imei, "oaid": self.oaid, "idfv": self.idfv,
                "device_brand": self.device_brand, "device_model": self.device_model,
                "os": self.os, "system_version": self.system_version,
                "split_ua": self.split_ua,
            }


class BlocklistEntry(BaseModel):
    key: str = Field(description="查找键 (uid/设备指纹/IP)")
    type: EntryType
    list_type: ListType
    platform: str = ""
    value: dict = Field(default_factory=dict, description="完整 JSON 数据")


class BlocklistResponse(BaseModel):
    total: int
    items: list[BlocklistEntry]
