"""
特征注册表 — 所有特征名称常量、描述、欺诈类型映射的唯一定义源。

新增/修改/删除特征只需要改这一个文件。
"""
from __future__ import annotations

from dataclasses import dataclass


# ── 特征名称常量 ──────────────────────────────────────────────────────────

class F:
    """Feature name constants — 消除硬编码字符串。"""

    # 设备 / 环境
    DISTINCT_DEVICE_COUNT = "distinct_device_count"
    DISTINCT_IP_COUNT = "distinct_ip_count"
    DISTINCT_IP_COUNT_DAILY = "distinct_ip_count_daily"
    IS_SIMULATOR_RATIO = "is_simulator_ratio"
    DEVICE_BRAND_COUNT = "device_brand_count"
    DEVICE_MODEL_COUNT = "device_model_count"

    # 行为模式
    REGISTER_TO_FIRST_LOGIN_SECONDS = "register_to_first_login_seconds"
    LOGIN_FREQUENCY_DAILY = "login_frequency_daily"
    LOGIN_HOUR_ENTROPY = "login_hour_entropy"
    ROLE_COUNT = "role_count"
    SERVER_COUNT = "server_count"
    NIGHT_ACTIVITY_RATIO = "night_activity_ratio"

    # 付费 (base names, 可带 prefix 复用)
    TOTAL_PAYMENT = "total_payment"
    TOTAL_PAYMENT_DAILY = "total_payment_daily"
    PAYMENT_COUNT = "payment_count"
    PAYMENT_COUNT_DAILY = "payment_count_daily"
    AVG_PAYMENT = "avg_payment"
    PAYMENT_FREQUENCY = "payment_frequency"
    FIRST_PAYMENT_TIME_SINCE_REGISTER = "first_payment_time_since_register"
    PAYMENT_PER_ACTIVITY = "payment_per_activity"
    PAYMENT_COUNT_PER_ACTIVITY = "payment_count_per_activity"

    # 跨账号
    ACCOUNTS_PER_DEVICE = "accounts_per_device"
    ACCOUNTS_PER_IP = "accounts_per_ip"

    # 付费成功 (prefix 派生)
    SUCCESS_PREFIX = "success_"


# ── 数据库列名常量 ────────────────────────────────────────────────────────

class Col:
    """DB column name constants — 查询返回和 DataFrame 中的列名。"""
    UID = "uid"
    ACTION_TIME = "action_time"
    DEVICE_FP = "device_fp"
    IPV4 = "ipv4"
    IS_SIMULATOR = "is_simulator"
    DEVICE_BRAND = "device_brand"
    DEVICE_MODEL = "device_model"
    ROLE_ID = "role_id"
    SERVER_ID = "server_id"
    EVENT = "event"
    MONEY = "money"


# ── 特征定义 ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FeatureDef:
    """单个特征的完整定义。"""
    name: str
    description: str
    fraud_types: tuple[str, ...] = ()
    category: str = ""
    weight: float = 1.0  # Z-score 权重，<1 降低影响，>1 提升影响


FEATURES: tuple[FeatureDef, ...] = (
    # ── 设备 / 环境 (5) ──────────────────────────────────────────────
    FeatureDef(F.DISTINCT_DEVICE_COUNT,  "使用的不同设备数量",                category="device", fraud_types=("账号交易",)),
    FeatureDef(F.DISTINCT_IP_COUNT_DAILY, "日均使用的不同IP数量",             category="device", fraud_types=("账号交易", "代练"), weight=0.3),
    FeatureDef(F.IS_SIMULATOR_RATIO,     "模拟器登录比例",                   category="device", fraud_types=("脚本/外挂", "工作室")),
    FeatureDef(F.DEVICE_BRAND_COUNT,     "设备品牌多样性",                   category="device", fraud_types=("账号交易",)),
    FeatureDef(F.DEVICE_MODEL_COUNT,     "设备型号多样性",                   category="device", fraud_types=("账号交易",)),

    # ── 行为模式 (6) ─────────────────────────────────────────────────
    FeatureDef(F.REGISTER_TO_FIRST_LOGIN_SECONDS, "注册到首次登录时间(秒)",   category="behavior", fraud_types=("批量注册", "机器人")),
    FeatureDef(F.LOGIN_FREQUENCY_DAILY,  "日均登录频次",                     category="behavior", fraud_types=("脚本/外挂", "机器人"), weight=0.3),
    FeatureDef(F.LOGIN_HOUR_ENTROPY,     "登录时段分布熵(低=机器人特征)",     category="behavior", fraud_types=("脚本/外挂", "机器人")),
    FeatureDef(F.ROLE_COUNT,             "创建角色数量",                     category="behavior", fraud_types=("多开", "工作室")),
    FeatureDef(F.SERVER_COUNT,           "进入区服数量",                     category="behavior", fraud_types=("多开", "工作室")),
    FeatureDef(F.NIGHT_ACTIVITY_RATIO,   "凌晨活跃比例(0:00-6:00)",         category="behavior", fraud_types=("脚本/外挂", "机器人")),

    # ── 下单 (6) ─────────────────────────────────────────────────────
    FeatureDef(F.TOTAL_PAYMENT_DAILY,    "日均下单总金额",                   category="payment_order", fraud_types=("洗钱", "盗刷"), weight=0.1),
    FeatureDef(F.AVG_PAYMENT,            "平均单笔下单金额",                  category="payment_order", fraud_types=("盗刷",), weight=0.1),
    FeatureDef(F.PAYMENT_FREQUENCY,      "日均下单频次",                     category="payment_order", fraud_types=("盗刷", "洗钱"), weight=0.3),
    FeatureDef(F.FIRST_PAYMENT_TIME_SINCE_REGISTER, "注册到首次下单时间(秒)",  category="payment_order", fraud_types=("盗刷", "洗钱"), weight=0.1),
    FeatureDef(F.PAYMENT_PER_ACTIVITY,   "单次活跃付费金额",                  category="payment_order", fraud_types=("盗刷",)),
    FeatureDef(F.PAYMENT_COUNT_PER_ACTIVITY, "单次活跃付费次数",              category="payment_order", fraud_types=("盗刷",)),

    # ── 付费成功 (6) ─────────────────────────────────────────────────
    FeatureDef(f"{F.SUCCESS_PREFIX}{F.TOTAL_PAYMENT_DAILY}",    "日均付费成功总金额",               category="payment_success", fraud_types=("洗钱", "盗刷"), weight=0.3),
    FeatureDef(f"{F.SUCCESS_PREFIX}{F.AVG_PAYMENT}",            "付费成功平均单笔",                  category="payment_success", fraud_types=("盗刷",)),
    FeatureDef(f"{F.SUCCESS_PREFIX}{F.PAYMENT_FREQUENCY}",      "付费成功日均频次",                 category="payment_success", fraud_types=("盗刷", "洗钱"), weight=0.3),
    FeatureDef(f"{F.SUCCESS_PREFIX}{F.FIRST_PAYMENT_TIME_SINCE_REGISTER}", "注册到首次付费成功时间(秒)", category="payment_success", fraud_types=("盗刷", "洗钱"), weight=0.1),
    FeatureDef(f"{F.SUCCESS_PREFIX}{F.PAYMENT_PER_ACTIVITY}",   "付费成功单次活跃金额",              category="payment_success", fraud_types=("盗刷",)),
    FeatureDef(f"{F.SUCCESS_PREFIX}{F.PAYMENT_COUNT_PER_ACTIVITY}", "付费成功单次活跃次数",          category="payment_success", fraud_types=("盗刷",)),

    # ── 跨账号 (2) ───────────────────────────────────────────────────
    FeatureDef(F.ACCOUNTS_PER_DEVICE,    "同设备关联账号数",                  category="cross_account", fraud_types=("多开", "批量注册", "工作室")),
    FeatureDef(F.ACCOUNTS_PER_IP,        "同IP关联账号数",                   category="cross_account", fraud_types=("多开", "批量注册", "工作室")),
)

# ── 便捷访问 ──────────────────────────────────────────────────────────────

FEATURE_NAMES: list[str] = [f.name for f in FEATURES]

FEATURE_DESCRIPTIONS: dict[str, str] = {f.name: f.description for f in FEATURES}

FEATURE_FRAUD_TYPE_MAPPING: dict[str, list[str]] = {
    f.name: list(f.fraud_types) for f in FEATURES
}

FEATURE_WEIGHTS: list[float] = [f.weight for f in FEATURES]
