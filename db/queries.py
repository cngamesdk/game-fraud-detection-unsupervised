from __future__ import annotations

from datetime import datetime

from config import settings

# ---------------------------------------------------------------------------
# Helper: build WHERE clause fragments
# ---------------------------------------------------------------------------


def _time_filter(since: datetime | None, time_column: str = "action_time") -> tuple[str, list]:
    """Return (sql_fragment, params) for time column filter."""
    if since is None:
        return "", []
    return f"{time_column} >= %s", [since]


def _uid_filter(uids: list[str] | None) -> tuple[str, list]:
    """Return (sql_fragment, params) for uid IN (...) filter."""
    if not uids:
        return "", []
    placeholders = ",".join(["%s"] * len(uids))
    return f"uid IN ({placeholders})", list(uids)


def _platform_filter(platforms: list[str] | None) -> tuple[str, list]:
    """Return (sql_fragment, params) for platform column filter."""
    if platforms is None:
        return "", []
    placeholders = ",".join(["%s"] * len(platforms))
    return f"platform IN ({placeholders})", list(platforms)


def _in_game_action_filter(action: list[int] | None) -> tuple[str, list]:
    """Return (sql_fragment, params) for action_id column filter."""
    if action is None:
        return "", []
    placeholders = ",".join(["%s"] * len(action))
    return f"action_id IN ({placeholders})", list(action)


def _pay_filter(status: list[int] | None) -> tuple[str, list]:
    """Return (sql_fragment, params) for pay column filter."""
    if status is None:
        return "", []
    placeholders = ",".join(["%s"] * len(status))
    return f"test_order = 0 AND order_status_id IN ({placeholders})", list(status)


def _build_where(*fragments: tuple[str, list]) -> tuple[str, tuple]:
    """Combine multiple WHERE fragments with AND."""
    clauses = []
    params: list = []
    for sql, p in fragments:
        if sql:
            clauses.append(sql)
            params.extend(p)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, tuple(params)


# ---------------------------------------------------------------------------
# Device fingerprint expression (reused across queries)
# ---------------------------------------------------------------------------

DEVICE_FINGERPRINT_EXPR = (
    "COALESCE(NULLIF(imei,''), NULLIF(oaid,''), NULLIF(idfv,''), "
    "MD5(CONCAT(IFNULL(device_brand,''), IFNULL(device_model,''), "
    "IFNULL(os,''), IFNULL(system_version,''), IFNULL(split_ua,''))))"
)

COMMON_FIELDS = (
    "imei,oaid, imei AS idfa, idfv, device_brand, device_model, "
    "os, system_version, network,SUBSTRING_INDEX(ip,'-',1) AS ipv4, SUBSTRING_INDEX(ip,'-',-1) AS ipv6"
)

BASE_FIELDS = (
    COMMON_FIELDS
    + ", split_ua AS ua, is_simulator"
)

BASE_FIELDS_IN_GAME = (
    COMMON_FIELDS
    + ", '' AS ua, 0 AS is_simulator"
)

BASE_FIELDS_PAY = (
    COMMON_FIELDS
    + ", split_ua AS ua, 0 AS is_simulator"
)

PLATFORMS = settings.PLATFORMS

# Per-table time column mapping
TIME_COLUMNS: dict[str, str] = {
    "activation": "action_time",
    "registration": "game_reg_time",
    "login": "login_time",
    "role_creation": "action_time",
    "ingame_event": "action_time",
    "payment": "pay_time",
}

# ---------------------------------------------------------------------------
# Per-table query builders
# ---------------------------------------------------------------------------


def query_activations(since: datetime | None = None) -> tuple[str, tuple]:
    tc = TIME_COLUMNS["activation"]
    where, params = _build_where(_platform_filter(PLATFORMS), _time_filter(since, tc))
    sql = (
        f"SELECT {tc} AS action_time, {BASE_FIELDS}, {DEVICE_FINGERPRINT_EXPR} AS device_fp "
        f"FROM {settings.TABLE_ACTIVATION}{where}"
    )
    return sql, params


def query_registrations(
    since: datetime | None = None,
    uids: list[str] | None = None,
) -> tuple[str, tuple]:
    tc = TIME_COLUMNS["registration"]
    where, params = _build_where(_platform_filter(PLATFORMS), _time_filter(since, tc), _uid_filter(uids))
    sql = (
        f"SELECT uid, {tc} AS action_time, {BASE_FIELDS}, {DEVICE_FINGERPRINT_EXPR} AS device_fp "
        f"FROM {settings.TABLE_REGISTRATION}{where}"
    )
    return sql, params


def query_logins(
    since: datetime | None = None,
    uids: list[str] | None = None,
) -> tuple[str, tuple]:
    tc = TIME_COLUMNS["login"]
    where, params = _build_where(_platform_filter(PLATFORMS), _time_filter(since, tc), _uid_filter(uids))
    sql = (
        f"SELECT uid, {tc} AS action_time, {BASE_FIELDS}, {DEVICE_FINGERPRINT_EXPR} AS device_fp "
        f"FROM {settings.TABLE_LOGIN}{where} ORDER BY {tc}"
    )
    return sql, params


def query_role_creations(
    since: datetime | None = None,
    uids: list[str] | None = None,
) -> tuple[str, tuple]:
    tc = TIME_COLUMNS["role_creation"]
    where, params = _build_where(_platform_filter(PLATFORMS), _in_game_action_filter([2]), _time_filter(since, tc), _uid_filter(uids))
    sql = (
        f"SELECT uid, role_id, server_id, {tc} AS action_time "
        f"FROM {settings.TABLE_ROLE_CREATION}{where}"
    )
    return sql, params


def query_ingame_events(
    since: datetime | None = None,
    uids: list[str] | None = None,
) -> tuple[str, tuple]:
    tc = TIME_COLUMNS["ingame_event"]
    where, params = _build_where(_platform_filter(PLATFORMS), _in_game_action_filter([4]), _time_filter(since, tc), _uid_filter(uids))
    sql = (
        f"SELECT uid, action_id AS event, {tc} AS action_time "
        f"FROM {settings.TABLE_INGAME_EVENT}{where}"
    )
    return sql, params


def query_payments(
    since: datetime | None = None,
    uids: list[str] | None = None,
) -> tuple[str, tuple]:
    """下单日志 (order_status_id=11)。"""
    tc = TIME_COLUMNS["payment"]
    where, params = _build_where(_platform_filter(PLATFORMS), _pay_filter([11]), _time_filter(since, tc), _uid_filter(uids))
    sql = (
        f"SELECT uid, pay_money AS money, {tc} AS action_time "
        f"FROM {settings.TABLE_PAYMENT}{where}"
    )
    return sql, params


def query_payments_success(
    since: datetime | None = None,
    uids: list[str] | None = None,
) -> tuple[str, tuple]:
    """付费成功日志 (order_status_id=1)。"""
    tc = TIME_COLUMNS["payment"]
    where, params = _build_where(_platform_filter(PLATFORMS), _pay_filter([1]), _time_filter(since, tc), _uid_filter(uids))
    sql = (
        f"SELECT uid, pay_money AS money, {tc} AS action_time "
        f"FROM {settings.TABLE_PAYMENT}{where}"
    )
    return sql, params


# ---------------------------------------------------------------------------
# Cross-account queries (aggregated at SQL level for performance)
# ---------------------------------------------------------------------------

def query_accounts_per_device(
    since: datetime | None = None,
    device_fps: list[str] | None = None,
) -> tuple[str, tuple]:
    """Count distinct UIDs per device fingerprint.

    When *device_fps* is provided the aggregation is limited to those
    fingerprints only (much cheaper for single-user prediction).
    """
    table_tc_pairs = [
        (settings.TABLE_REGISTRATION, TIME_COLUMNS["registration"]),
        (settings.TABLE_LOGIN, TIME_COLUMNS["login"]),
    ]
    parts = []
    for tbl, tc in table_tc_pairs:
        time_frag = _time_filter(since, tc)
        w, p = _build_where(_platform_filter(PLATFORMS), time_frag)
        parts.append(
            (f"SELECT uid, {DEVICE_FINGERPRINT_EXPR} AS device_fp FROM {tbl}{w}", p)
        )
    sqls = " UNION ALL ".join(s for s, _ in parts)
    params: list = []
    for _, p in parts:
        params.extend(p)

    fp_filter = ""
    if device_fps:
        placeholders = ",".join(["%s"] * len(device_fps))
        fp_filter = f" WHERE device_fp IN ({placeholders})"
        params.extend(device_fps)

    sql = (
        f"SELECT device_fp, COUNT(DISTINCT uid) AS account_count "
        f"FROM ({sqls}) AS t{fp_filter} GROUP BY device_fp"
    )
    return sql, tuple(params)


def query_accounts_per_ip(
    since: datetime | None = None,
    ips: list[str] | None = None,
) -> tuple[str, tuple]:
    """Count distinct UIDs per IPv4 within window.

    When *ips* is provided the aggregation is limited to those addresses
    only (much cheaper for single-user prediction).
    """
    table_tc_pairs = [
        (settings.TABLE_REGISTRATION, TIME_COLUMNS["registration"]),
        (settings.TABLE_LOGIN, TIME_COLUMNS["login"]),
    ]
    IP_EXPR = "SUBSTRING_INDEX(ip,'-',1)"
    parts = []
    for tbl, tc in table_tc_pairs:
        time_frag = _time_filter(since, tc)
        ip_frag = (f"{IP_EXPR} IS NOT NULL AND {IP_EXPR} != ''", [])
        w, p = _build_where(_platform_filter(PLATFORMS), time_frag, ip_frag)
        parts.append(
            (f"SELECT uid, {IP_EXPR} AS ipv4 FROM {tbl}{w}", p)
        )
    sqls = " UNION ALL ".join(s for s, _ in parts)
    params: list = []
    for _, p in parts:
        params.extend(p)

    ip_filter = ""
    if ips:
        placeholders = ",".join(["%s"] * len(ips))
        ip_filter = f" WHERE ipv4 IN ({placeholders})"
        params.extend(ips)

    sql = (
        f"SELECT ipv4, COUNT(DISTINCT uid) AS account_count "
        f"FROM ({sqls}) AS t{ip_filter} GROUP BY ipv4"
    )
    return sql, tuple(params)


def query_uid_device_mapping(
    since: datetime | None = None,
    uids: list[str] | None = None,
) -> tuple[str, tuple]:
    """Get (uid, device_fp) pairs for mapping uid -> device -> account_count."""
    uid_frag = _uid_filter(uids)
    table_tc_pairs = [
        (settings.TABLE_REGISTRATION, TIME_COLUMNS["registration"]),
        (settings.TABLE_LOGIN, TIME_COLUMNS["login"]),
    ]
    parts = []
    for tbl, tc in table_tc_pairs:
        time_frag = _time_filter(since, tc)
        w, p = _build_where(time_frag, uid_frag)
        parts.append(
            (f"SELECT DISTINCT uid, {DEVICE_FINGERPRINT_EXPR} AS device_fp FROM {tbl}{w}", p)
        )
    sqls = " UNION ".join(s for s, _ in parts)
    params = []
    for _, p in parts:
        params.extend(p)
    return sqls, tuple(params)


def query_uid_ip_mapping(
    since: datetime | None = None,
    uids: list[str] | None = None,
) -> tuple[str, tuple]:
    """Get (uid, ipv4) pairs for mapping uid -> ip -> account_count."""
    uid_frag = _uid_filter(uids)
    table_tc_pairs = [
        (settings.TABLE_REGISTRATION, TIME_COLUMNS["registration"]),
        (settings.TABLE_LOGIN, TIME_COLUMNS["login"]),
    ]
    IP_EXPR = "SUBSTRING_INDEX(ip,'-',1)"
    parts = []
    for tbl, tc in table_tc_pairs:
        time_frag = _time_filter(since, tc)
        w, p = _build_where(time_frag, uid_frag)
        parts.append(
            (f"SELECT DISTINCT uid, {IP_EXPR} AS ipv4 FROM {tbl}{w}", p)
        )
    sqls = " UNION ".join(s for s, _ in parts)
    params = []
    for _, p in parts:
        params.extend(p)
    return sqls, tuple(params)


# ---------------------------------------------------------------------------
# Blocklist queries (unused by ListManager which does direct SQL, kept for reference)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Reverse-lookup: find UIDs by device fingerprint or IP
# ---------------------------------------------------------------------------

def query_uids_by_device(
    since: datetime | None = None,
    device_fp: str = "",
) -> tuple[str, tuple]:
    """Find distinct UIDs associated with a device fingerprint."""
    table_tc_pairs = [
        (settings.TABLE_REGISTRATION, TIME_COLUMNS["registration"]),
        (settings.TABLE_LOGIN, TIME_COLUMNS["login"]),
    ]
    parts = []
    for tbl, tc in table_tc_pairs:
        time_frag = _time_filter(since, tc)
        fp_frag = (f"{DEVICE_FINGERPRINT_EXPR} = %s", [device_fp])
        w, p = _build_where(_platform_filter(PLATFORMS), time_frag, fp_frag)
        parts.append((f"SELECT DISTINCT uid FROM {tbl}{w}", p))
    sqls = " UNION ".join(s for s, _ in parts)
    params: list = []
    for _, p in parts:
        params.extend(p)
    return sqls, tuple(params)


def query_uids_by_ip(
    since: datetime | None = None,
    ip: str = "",
) -> tuple[str, tuple]:
    """Find distinct UIDs associated with an IPv4 address."""
    IP_EXPR = "SUBSTRING_INDEX(ip,'-',1)"
    table_tc_pairs = [
        (settings.TABLE_REGISTRATION, TIME_COLUMNS["registration"]),
        (settings.TABLE_LOGIN, TIME_COLUMNS["login"]),
    ]
    parts = []
    for tbl, tc in table_tc_pairs:
        time_frag = _time_filter(since, tc)
        ip_frag = (f"{IP_EXPR} = %s", [ip])
        w, p = _build_where(_platform_filter(PLATFORMS), time_frag, ip_frag)
        parts.append((f"SELECT DISTINCT uid FROM {tbl}{w}", p))
    sqls = " UNION ".join(s for s, _ in parts)
    params: list = []
    for _, p in parts:
        params.extend(p)
    return sqls, tuple(params)
