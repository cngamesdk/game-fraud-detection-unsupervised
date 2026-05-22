from __future__ import annotations

import hashlib
import json

import aiomysql
from loguru import logger

from config import settings
from db.connection import execute_query, get_pool

ENTRY_TYPES = ("uid", "device", "ip")
LIST_TYPES = ("whitelist", "blacklist")


def _compute_key(entry_type: str, data: dict) -> str:
    """从 JSON data 中提取/计算内存查找用的 key。"""
    if entry_type == "uid":
        return str(data.get("uid", ""))
    elif entry_type == "ip":
        return str(data.get("ip", ""))
    elif entry_type == "device":
        for field in ("imei", "oaid", "idfv"):
            v = data.get(field, "").strip()
            if v:
                return v
        raw = "".join(data.get(k, "") for k in (
            "device_brand", "device_model", "os", "system_version", "split_ua",
        ))
        return hashlib.md5(raw.encode("utf-8")).hexdigest()
    return ""


class ListManager:
    """Unified in-memory blacklist/whitelist for uid, device, ip.

    DB table ``fraud_blocklist`` columns: `value` (JSON text), `type`,
    list_type, platform.
    """

    def __init__(self) -> None:
        self._sets: dict[str, set[str]] = {
            f"{t}_{l}": set() for t in ENTRY_TYPES for l in LIST_TYPES
        }

    @staticmethod
    def _key(entry_type: str, list_type: str) -> str:
        return f"{entry_type}_{list_type}"

    # ── read ──────────────────────────────────────────────────────────

    def is_whitelisted(self, value: str, entry_type: str = "uid") -> bool:
        return str(value) in self._sets[self._key(entry_type, "whitelist")]

    def is_blacklisted(self, value: str, entry_type: str = "uid") -> bool:
        return str(value) in self._sets[self._key(entry_type, "blacklist")]

    def list_all(self, entry_type: str = "uid", list_type: str = "whitelist") -> list[str]:
        return sorted(self._sets[self._key(entry_type, list_type)])

    def filter_uids(self, uids: list[str]) -> tuple[list[str], list[str], list[str]]:
        wl = self._sets[self._key("uid", "whitelist")]
        bl = self._sets[self._key("uid", "blacklist")]
        whitelisted, blacklisted, normal = [], [], []
        for u in uids:
            s = str(u)
            if s in bl:
                blacklisted.append(u)
            elif s in wl:
                whitelisted.append(u)
            else:
                normal.append(u)
        return whitelisted, blacklisted, normal

    # ── load / refresh ────────────────────────────────────────────────

    async def load(self) -> None:
        tbl = settings.TABLE_BLOCKLIST
        rows = await execute_query(
            f"SELECT `value`, `type`, list_type FROM {tbl}"
        )

        for s in self._sets.values():
            s.clear()

        for row in rows:
            try:
                data = json.loads(row["value"])
            except (json.JSONDecodeError, TypeError):
                continue
            key = _compute_key(row["type"], data)
            if not key:
                continue
            set_key = self._key(row["type"], row["list_type"])
            if set_key in self._sets:
                self._sets[set_key].add(key)

        total = sum(len(s) for s in self._sets.values())
        logger.info(f"ListManager loaded: {total} entries")

    # ── write-through mutations ───────────────────────────────────────

    async def add(self, entry_type: str, list_type: str,
                  data: dict, platform: str = "") -> str:
        key = _compute_key(entry_type, data)
        value_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        tbl = settings.TABLE_BLOCKLIST
        await execute_query(
            f"INSERT IGNORE INTO {tbl} "
            f"(`value`, `type`, list_type, platform)"
            f" VALUES (%s, %s, %s, %s)",
            (value_json, entry_type, list_type, platform),
        )
        self._sets[self._key(entry_type, list_type)].add(key)
        return key

    async def remove(self, key: str, entry_type: str, list_type: str) -> bool:
        tbl = settings.TABLE_BLOCKLIST

        rows = await execute_query(
            f"SELECT `value` FROM {tbl} WHERE `type` = %s AND list_type = %s",
            (entry_type, list_type),
        )

        target_json = None
        for row in rows:
            try:
                data = json.loads(row["value"])
            except (json.JSONDecodeError, TypeError):
                continue
            if _compute_key(entry_type, data) == key:
                target_json = row["value"]
                break

        if target_json is None:
            self._sets[self._key(entry_type, list_type)].discard(key)
            return False

        await execute_query(
            f"DELETE FROM {tbl} "
            f"WHERE `type` = %s AND list_type = %s AND `value` = %s",
            (entry_type, list_type, target_json),
        )

        self._sets[self._key(entry_type, list_type)].discard(key)
        return True
