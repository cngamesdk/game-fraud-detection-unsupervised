from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import entropy

from db.connection import execute_query, execute_query_batched
from db import queries
from config import settings, TZ
from features.registry import FEATURE_NAMES, F, Col
from loguru import logger


class _FeatureCache:
    """Simple TTL cache for per-uid feature DataFrames."""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 2000) -> None:
        self._ttl = ttl_seconds
        self._max = max_size
        self._store: dict[str, tuple[float, pd.DataFrame]] = {}

    def get(self, uid: str) -> pd.DataFrame | None:
        entry = self._store.get(uid)
        if entry is None:
            return None
        ts, df = entry
        if time.monotonic() - ts > self._ttl:
            del self._store[uid]
            return None
        return df

    def put(self, uid: str, df: pd.DataFrame) -> None:
        if len(self._store) >= self._max:
            # evict oldest
            oldest = min(self._store, key=lambda k: self._store[k][0])
            del self._store[oldest]
        self._store[uid] = (time.monotonic(), df)


class FeatureEngineer:
    """Extract feature vectors from behavioral logs for each uid."""

    FEATURE_NAMES = FEATURE_NAMES  # re-export from registry

    def __init__(self) -> None:
        self._cache = _FeatureCache(ttl_seconds=300, max_size=2000)

    # ── Public API ───────────────────────────────────────────────────────

    async def extract_features(
        self,
        since: datetime | None = None,
        uids: list[str] | None = None,
        exclude_uids: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Main entry point.  Returns a DataFrame indexed by uid with columns = FEATURE_NAMES.

        Args:
            since: Only include data after this timestamp.
            uids: Only include these specific UIDs (prediction mode).
            exclude_uids: Remove these UIDs from the result (training mode,
                          used to filter out whitelisted test accounts).
        """
        # ── Default time window ────────────────────────────────────
        if since is None:
            since = datetime.now(TZ) - timedelta(days=settings.FEATURE_PREDICT_WINDOW_DAYS)

        # ── Cache hit (prediction for single uid) ────────────────────
        if uids and len(uids) == 1 and not exclude_uids:
            cached = self._cache.get(uids[0])
            if cached is not None:
                logger.debug(f"Cache hit for uid={uids[0]}")
                return cached

        # ── Prediction path: 2-round optimised pipeline ─────────────
        if uids:
            combined = await self._extract_for_uids(since, uids)
        else:
            combined = await self._extract_full(since)

        # Drop whitelisted UIDs from training data
        if exclude_uids and not combined.empty:
            exclude_set = {str(u) for u in exclude_uids}
            combined = combined[~combined.index.isin(exclude_set)]

        # Populate cache for single uid prediction
        if uids and len(uids) == 1 and not combined.empty:
            self._cache.put(uids[0], combined)

        return combined

    async def _extract_for_uids(
        self,
        since: datetime,
        uids: list[str],
    ) -> pd.DataFrame:
        """Optimised path for prediction: use buffered cursor, merge query rounds."""
        # ── Round 1: 7 concurrent queries (5 logs + 2 mappings) ─────
        uid_dev_sql, uid_dev_params = queries.query_uid_device_mapping(since, uids)
        uid_ip_sql, uid_ip_params = queries.query_uid_ip_mapping(since, uids)

        (reg, login, role, pay, pay_success, ingame,
         uid_dev_rows, uid_ip_rows) = await asyncio.gather(
            self._fetch_small(queries.query_registrations, since=since, uids=uids),
            self._fetch_small(queries.query_logins, since=since, uids=uids),
            self._fetch_small(queries.query_role_creations, since=since, uids=uids),
            self._fetch_small(queries.query_payments, since=since, uids=uids),
            self._fetch_small(queries.query_payments_success, since=since, uids=uids),
            self._fetch_small(queries.query_ingame_events, since=since, uids=uids),
            execute_query(uid_dev_sql, uid_dev_params),
            execute_query(uid_ip_sql, uid_ip_params),
        )

        # ── Round 2: 2 count queries (depend on mapping results) ────
        cross_feats = await self._build_cross_from_mappings(
            since, uids, uid_dev_rows, uid_ip_rows,
        )

        device_feats = self._build_device_features(login, reg)
        behavior_feats = self._build_behavior_features(reg, login, role)
        payment_feats = self._build_payment_features(pay, reg, ingame_df=ingame)
        payment_success_feats = self._build_payment_features(pay_success, reg, prefix=F.SUCCESS_PREFIX, ingame_df=ingame)

        return self._combine_and_fill(device_feats, behavior_feats, payment_feats, payment_success_feats, cross_feats)

    async def _extract_full(
        self,
        since: datetime,
    ) -> pd.DataFrame:
        """Training path: use streaming cursor for large result sets."""
        log_data = await self._fetch_all_logs(since, uids=None)

        reg_df = log_data["registrations"]
        login_df = log_data["logins"]
        role_df = log_data["roles"]
        pay_df = log_data["payments"]
        pay_success_df = log_data["payments_success"]
        ingame_df = log_data["ingame_events"]

        device_feats = self._build_device_features(login_df, reg_df)
        behavior_feats = self._build_behavior_features(reg_df, login_df, role_df)
        payment_feats = self._build_payment_features(pay_df, reg_df, ingame_df=ingame_df)
        payment_success_feats = self._build_payment_features(pay_success_df, reg_df, prefix=F.SUCCESS_PREFIX, ingame_df=ingame_df)
        cross_feats = await self._build_cross_account_features(since, uids=None)

        return self._combine_and_fill(device_feats, behavior_feats, payment_feats, payment_success_feats, cross_feats)

    # ── Data fetching ────────────────────────────────────────────────────

    @staticmethod
    async def _fetch_small(query_fn, **kwargs) -> pd.DataFrame:
        """Fetch with buffered cursor — fast for small result sets (prediction)."""
        sql, params = query_fn(**kwargs)
        rows = await execute_query(sql, params)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        if Col.UID in df.columns:
            df[Col.UID] = df[Col.UID].astype(str)
        return df

    async def _fetch_all_logs(
        self,
        since: datetime | None,
        uids: list[str] | None,
    ) -> dict[str, pd.DataFrame]:
        """Fire 5 log queries concurrently with streaming cursor (training path)."""

        async def _fetch_batched(query_fn, label: str, **kwargs) -> pd.DataFrame:
            sql, params = query_fn(**kwargs)
            chunks: list[pd.DataFrame] = []
            total_rows = 0
            async for batch in execute_query_batched(sql, params):
                df = pd.DataFrame(batch)
                if Col.UID in df.columns:
                    df[Col.UID] = df[Col.UID].astype(str)
                chunks.append(df)
                total_rows += len(batch)
            if chunks:
                logger.debug(f"[{label}] fetched {total_rows} rows in {len(chunks)} batch(es)")
                return pd.concat(chunks, ignore_index=True)
            return pd.DataFrame()

        reg, login, role, pay, pay_success, ingame = await asyncio.gather(
            _fetch_batched(queries.query_registrations, "registration", since=since, uids=uids),
            _fetch_batched(queries.query_logins, "login", since=since, uids=uids),
            _fetch_batched(queries.query_role_creations, "role_creation", since=since, uids=uids),
            _fetch_batched(queries.query_payments, "payment", since=since, uids=uids),
            _fetch_batched(queries.query_payments_success, "payment_success", since=since, uids=uids),
            _fetch_batched(queries.query_ingame_events, "ingame_event", since=since, uids=uids),
        )

        return {
            "registrations": reg,
            "logins": login,
            "roles": role,
            "payments": pay,
            "payments_success": pay_success,
            "ingame_events": ingame,
        }

    async def _build_cross_from_mappings(
        self,
        since: datetime,
        uids: list[str],
        uid_dev_rows: list[dict],
        uid_ip_rows: list[dict],
    ) -> pd.DataFrame:
        """Build cross-account features from pre-fetched mapping rows (prediction path)."""
        target_fps: set[str] = set()
        target_ips: set[str] = set()
        for row in uid_dev_rows:
            if row.get(Col.DEVICE_FP):
                target_fps.add(row[Col.DEVICE_FP])
        for row in uid_ip_rows:
            if row.get(Col.IPV4):
                target_ips.add(row[Col.IPV4])

        dev_sql, dev_params = queries.query_accounts_per_device(
            since, list(target_fps) if target_fps else None,
        )
        ip_sql, ip_params = queries.query_accounts_per_ip(
            since, list(target_ips) if target_ips else None,
        )

        dev_counts_rows, ip_counts_rows = await asyncio.gather(
            execute_query(dev_sql, dev_params),
            execute_query(ip_sql, ip_params),
        )

        dev_count_map: dict[str, int] = {}
        for row in dev_counts_rows:
            fp = row.get(Col.DEVICE_FP)
            if fp:
                dev_count_map[fp] = row["account_count"]

        ip_count_map: dict[str, int] = {}
        for row in ip_counts_rows:
            ip = row.get(Col.IPV4)
            if ip:
                ip_count_map[ip] = row["account_count"]

        uid_dev_max: dict[str, int] = {}
        for row in uid_dev_rows:
            uid = str(row[Col.UID]) if row.get(Col.UID) is not None else None
            fp = row.get(Col.DEVICE_FP)
            if uid and fp:
                count = dev_count_map.get(fp, 1)
                uid_dev_max[uid] = max(uid_dev_max.get(uid, 1), count)

        uid_ip_max: dict[str, int] = {}
        for row in uid_ip_rows:
            uid = str(row[Col.UID]) if row.get(Col.UID) is not None else None
            ip = row.get(Col.IPV4)
            if uid and ip:
                count = ip_count_map.get(ip, 1)
                uid_ip_max[uid] = max(uid_ip_max.get(uid, 1), count)

        all_uids = set(uid_dev_max.keys()) | set(uid_ip_max.keys())
        if not all_uids:
            return pd.DataFrame(columns=[F.ACCOUNTS_PER_DEVICE, F.ACCOUNTS_PER_IP]).rename_axis(Col.UID)

        result = pd.DataFrame(index=sorted(all_uids))
        result.index.name = Col.UID
        result[F.ACCOUNTS_PER_DEVICE] = pd.Series(uid_dev_max).reindex(result.index, fill_value=1)
        result[F.ACCOUNTS_PER_IP] = pd.Series(uid_ip_max).reindex(result.index, fill_value=1)
        return result

    # ── Feature builders ─────────────────────────────────────────────────

    def _build_device_features(
        self,
        login_df: pd.DataFrame,
        reg_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Per uid:
          distinct_device_count, distinct_ip_count, is_simulator_ratio,
          device_brand_count, device_model_count
        """
        # Combine registration + login rows for device-level stats
        frames = []
        for df in (reg_df, login_df):
            if not df.empty and Col.UID in df.columns:
                frames.append(df)
        if not frames:
            return pd.DataFrame(columns=[Col.UID] + self.FEATURE_NAMES[:5]).set_index(Col.UID)

        combined = pd.concat(frames, ignore_index=True)

        grouped = combined.groupby(Col.UID)

        result = pd.DataFrame(index=grouped.groups.keys())
        result.index.name = Col.UID

        if Col.DEVICE_FP in combined.columns:
            result[F.DISTINCT_DEVICE_COUNT] = grouped[Col.DEVICE_FP].nunique()
        else:
            result[F.DISTINCT_DEVICE_COUNT] = 1

        if Col.IPV4 in combined.columns:
            result[F.DISTINCT_IP_COUNT] = grouped[Col.IPV4].nunique()
            # 日均不同IP数
            if Col.ACTION_TIME in combined.columns:
                ip_daily = {}
                for uid, grp in grouped:
                    ts = pd.to_datetime(grp[Col.ACTION_TIME])
                    days = max((ts.max() - ts.min()).days, 1)
                    ip_daily[uid] = grp[Col.IPV4].nunique() / days
                result[F.DISTINCT_IP_COUNT_DAILY] = pd.Series(ip_daily)
            else:
                result[F.DISTINCT_IP_COUNT_DAILY] = result[F.DISTINCT_IP_COUNT]
        else:
            result[F.DISTINCT_IP_COUNT] = 1
            result[F.DISTINCT_IP_COUNT_DAILY] = 1

        if Col.IS_SIMULATOR in combined.columns:
            result[F.IS_SIMULATOR_RATIO] = grouped[Col.IS_SIMULATOR].apply(
                lambda s: s.astype(float).mean()
            )
        else:
            result[F.IS_SIMULATOR_RATIO] = 0.0

        if Col.DEVICE_BRAND in combined.columns:
            result[F.DEVICE_BRAND_COUNT] = grouped[Col.DEVICE_BRAND].nunique()
        else:
            result[F.DEVICE_BRAND_COUNT] = 1

        if Col.DEVICE_MODEL in combined.columns:
            result[F.DEVICE_MODEL_COUNT] = grouped[Col.DEVICE_MODEL].nunique()
        else:
            result[F.DEVICE_MODEL_COUNT] = 1

        return result

    def _build_behavior_features(
        self,
        reg_df: pd.DataFrame,
        login_df: pd.DataFrame,
        role_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Per uid:
          register_to_first_login_seconds, login_frequency_daily,
          login_hour_entropy, role_count, server_count, night_activity_ratio
        """
        all_uids: set[str] = set()
        for df in (reg_df, login_df, role_df):
            if not df.empty and Col.UID in df.columns:
                all_uids.update(df[Col.UID].unique())

        if not all_uids:
            return pd.DataFrame(
                columns=[Col.UID] + self.FEATURE_NAMES[5:11],
            ).set_index(Col.UID)

        result = pd.DataFrame(index=sorted(all_uids))
        result.index.name = Col.UID

        # register_to_first_login_seconds
        reg_time = {}
        if not reg_df.empty and Col.UID in reg_df.columns:
            for uid, grp in reg_df.groupby(Col.UID):
                reg_time[uid] = pd.to_datetime(grp[Col.ACTION_TIME]).min()

        first_login = {}
        if not login_df.empty and Col.UID in login_df.columns:
            for uid, grp in login_df.groupby(Col.UID):
                first_login[uid] = pd.to_datetime(grp[Col.ACTION_TIME]).min()

        reg_to_login = {}
        for uid in all_uids:
            if uid in reg_time and uid in first_login:
                delta = (first_login[uid] - reg_time[uid]).total_seconds()
                reg_to_login[uid] = max(delta, 0.0)
            else:
                reg_to_login[uid] = 0.0
        result[F.REGISTER_TO_FIRST_LOGIN_SECONDS] = pd.Series(reg_to_login)

        # login_frequency_daily & login_hour_entropy & night_activity_ratio
        login_freq = {}
        login_entropy = {}
        night_ratio = {}
        if not login_df.empty and Col.UID in login_df.columns:
            login_df = login_df.copy()
            login_df["_ts"] = pd.to_datetime(login_df[Col.ACTION_TIME])
            login_df["_hour"] = login_df["_ts"].dt.hour
            for uid, grp in login_df.groupby(Col.UID):
                ts = grp["_ts"]
                active_days = max((ts.max() - ts.min()).days, 1)
                login_freq[uid] = len(grp) / active_days

                hours = grp["_hour"]
                hist = np.histogram(hours, bins=24, range=(0, 24))[0].astype(float)
                hist_sum = hist.sum()
                login_entropy[uid] = float(entropy(hist / hist_sum)) if hist_sum > 0 else 0.0

                night_count = ((hours >= 0) & (hours < 6)).sum()
                night_ratio[uid] = night_count / len(grp) if len(grp) > 0 else 0.0

        result[F.LOGIN_FREQUENCY_DAILY] = pd.Series(login_freq).reindex(result.index, fill_value=0.0)
        result[F.LOGIN_HOUR_ENTROPY] = pd.Series(login_entropy).reindex(result.index, fill_value=0.0)

        # role_count & server_count
        if not role_df.empty and Col.UID in role_df.columns:
            role_grouped = role_df.groupby(Col.UID)
            result[F.ROLE_COUNT] = role_grouped[Col.ROLE_ID].nunique().reindex(result.index, fill_value=0)
            result[F.SERVER_COUNT] = role_grouped[Col.SERVER_ID].nunique().reindex(result.index, fill_value=0)
        else:
            result[F.ROLE_COUNT] = 0
            result[F.SERVER_COUNT] = 0

        # night_activity_ratio (login only)
        result[F.NIGHT_ACTIVITY_RATIO] = pd.Series(night_ratio).reindex(result.index, fill_value=0.0)

        return result

    def _build_payment_features(
        self,
        pay_df: pd.DataFrame,
        reg_df: pd.DataFrame,
        prefix: str = "",
        ingame_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Per uid:
          {prefix}total_payment, {prefix}payment_count, {prefix}avg_payment,
          {prefix}payment_frequency,
          {prefix}first_payment_time_since_register,
          {prefix}payment_per_activity, {prefix}payment_count_per_activity
        """
        col_names = [
            f"{prefix}{F.TOTAL_PAYMENT}",
            f"{prefix}{F.TOTAL_PAYMENT_DAILY}",
            f"{prefix}{F.PAYMENT_COUNT}",
            f"{prefix}{F.PAYMENT_COUNT_DAILY}",
            f"{prefix}{F.AVG_PAYMENT}",
            f"{prefix}{F.PAYMENT_FREQUENCY}",
            f"{prefix}{F.FIRST_PAYMENT_TIME_SINCE_REGISTER}",
            f"{prefix}{F.PAYMENT_PER_ACTIVITY}",
            f"{prefix}{F.PAYMENT_COUNT_PER_ACTIVITY}",
        ]
        if pay_df.empty or Col.UID not in pay_df.columns:
            return pd.DataFrame(columns=col_names).rename_axis(Col.UID)

        pay_df = pay_df.copy()
        pay_df[Col.MONEY] = pd.to_numeric(pay_df[Col.MONEY], errors="coerce").fillna(0)
        pay_df["_ts"] = pd.to_datetime(pay_df[Col.ACTION_TIME])

        grouped = pay_df.groupby(Col.UID)
        result = pd.DataFrame(index=grouped.groups.keys())
        result.index.name = Col.UID
        result[f"{prefix}{F.TOTAL_PAYMENT}"] = grouped[Col.MONEY].sum()
        result[f"{prefix}{F.PAYMENT_COUNT}"] = grouped[Col.MONEY].count()
        result[f"{prefix}{F.AVG_PAYMENT}"] = grouped[Col.MONEY].mean()

        # daily metrics: total_payment_daily, payment_count_daily, payment_frequency
        pay_freq = {}
        pay_total_daily = {}
        pay_count_daily = {}
        for uid, grp in grouped:
            ts = grp["_ts"]
            active_days = max((ts.max() - ts.min()).days, 1)
            pay_freq[uid] = len(grp) / active_days
            pay_total_daily[uid] = grp[Col.MONEY].sum() / active_days
            pay_count_daily[uid] = len(grp) / active_days
        result[f"{prefix}{F.TOTAL_PAYMENT_DAILY}"] = pd.Series(pay_total_daily)
        result[f"{prefix}{F.PAYMENT_COUNT_DAILY}"] = pd.Series(pay_count_daily)
        result[f"{prefix}{F.PAYMENT_FREQUENCY}"] = pd.Series(pay_freq)

        # first_payment_time_since_register
        reg_time = {}
        if not reg_df.empty and Col.UID in reg_df.columns:
            for uid, grp in reg_df.groupby(Col.UID):
                reg_time[uid] = pd.to_datetime(grp[Col.ACTION_TIME]).min()

        first_pay = {}
        for uid, grp in grouped:
            first_pay[uid] = grp["_ts"].min()

        pay_since_reg = {}
        for uid in result.index:
            if uid in reg_time and uid in first_pay:
                delta = (first_pay[uid] - reg_time[uid]).total_seconds()
                pay_since_reg[uid] = max(delta, 0.0)
            else:
                pay_since_reg[uid] = 0.0
        result[f"{prefix}{F.FIRST_PAYMENT_TIME_SINCE_REGISTER}"] = pd.Series(pay_since_reg)

        # payment_per_activity & payment_count_per_activity
        ingame_count: pd.Series | None = None
        if ingame_df is not None and not ingame_df.empty and Col.UID in ingame_df.columns:
            ingame_count = ingame_df.groupby(Col.UID)[Col.ACTION_TIME].count().reindex(result.index, fill_value=0)
        if ingame_count is not None and (ingame_count > 0).any():
            safe = ingame_count.replace(0, np.nan)
            result[f"{prefix}{F.PAYMENT_PER_ACTIVITY}"] = (result[f"{prefix}{F.TOTAL_PAYMENT}"] / safe).fillna(0.0)
            result[f"{prefix}{F.PAYMENT_COUNT_PER_ACTIVITY}"] = (result[f"{prefix}{F.PAYMENT_COUNT}"] / safe).fillna(0.0)
        else:
            result[f"{prefix}{F.PAYMENT_PER_ACTIVITY}"] = 0.0
            result[f"{prefix}{F.PAYMENT_COUNT_PER_ACTIVITY}"] = 0.0

        return result

    async def _build_cross_account_features(
        self,
        since: datetime,
        uids: list[str] | None,
    ) -> pd.DataFrame:
        """
        Per uid:
          accounts_per_device, accounts_per_ip
        """

        # Step 1: Get uid->device and uid->ip mappings for target users
        uid_dev_sql, uid_dev_params = queries.query_uid_device_mapping(since, uids)
        uid_ip_sql, uid_ip_params = queries.query_uid_ip_mapping(since, uids)

        uid_dev_rows, uid_ip_rows = await asyncio.gather(
            execute_query(uid_dev_sql, uid_dev_params),
            execute_query(uid_ip_sql, uid_ip_params),
        )

        # Collect devices / IPs used by target users
        target_fps: set[str] = set()
        target_ips: set[str] = set()
        for row in uid_dev_rows:
            if row.get(Col.DEVICE_FP):
                target_fps.add(row[Col.DEVICE_FP])
        for row in uid_ip_rows:
            if row.get(Col.IPV4):
                target_ips.add(row[Col.IPV4])

        # Step 2: Count accounts per device / IP.
        # When uids is provided (prediction): scope to target user's
        # devices/IPs only — avoids expensive global aggregation.
        dev_sql, dev_params = queries.query_accounts_per_device(
            since, list(target_fps) if uids else None,
        )
        ip_sql, ip_params = queries.query_accounts_per_ip(
            since, list(target_ips) if uids else None,
        )

        dev_counts_rows, ip_counts_rows = await asyncio.gather(
            execute_query(dev_sql, dev_params),
            execute_query(ip_sql, ip_params),
        )

        # Build lookup: device_fp -> account_count
        dev_count_map: dict[str, int] = {}
        for row in dev_counts_rows:
            fp = row.get(Col.DEVICE_FP)
            if fp:
                dev_count_map[fp] = row["account_count"]

        # Build lookup: ipv4 -> account_count
        ip_count_map: dict[str, int] = {}
        for row in ip_counts_rows:
            ip = row.get(Col.IPV4)
            if ip:
                ip_count_map[ip] = row["account_count"]

        # Map uid -> max accounts_per_device
        uid_dev_max: dict[str, int] = {}
        for row in uid_dev_rows:
            uid = str(row[Col.UID]) if row.get(Col.UID) is not None else None
            fp = row.get(Col.DEVICE_FP)
            if uid and fp:
                count = dev_count_map.get(fp, 1)
                uid_dev_max[uid] = max(uid_dev_max.get(uid, 1), count)

        # Map uid -> max accounts_per_ip
        uid_ip_max: dict[str, int] = {}
        for row in uid_ip_rows:
            uid = str(row[Col.UID]) if row.get(Col.UID) is not None else None
            ip = row.get(Col.IPV4)
            if uid and ip:
                count = ip_count_map.get(ip, 1)
                uid_ip_max[uid] = max(uid_ip_max.get(uid, 1), count)

        all_uids = set(uid_dev_max.keys()) | set(uid_ip_max.keys())
        if not all_uids:
            return pd.DataFrame(columns=[F.ACCOUNTS_PER_DEVICE, F.ACCOUNTS_PER_IP]).rename_axis(Col.UID)

        result = pd.DataFrame(index=sorted(all_uids))
        result.index.name = Col.UID
        result[F.ACCOUNTS_PER_DEVICE] = pd.Series(uid_dev_max).reindex(result.index, fill_value=1)
        result[F.ACCOUNTS_PER_IP] = pd.Series(uid_ip_max).reindex(result.index, fill_value=1)
        return result

    # ── Combine ──────────────────────────────────────────────────────────

    def _combine_and_fill(self, *dfs: pd.DataFrame) -> pd.DataFrame:
        """Join all feature sub-DataFrames on uid, fill NaN with 0, ensure column order."""
        non_empty = [df for df in dfs if not df.empty]
        if not non_empty:
            return pd.DataFrame(columns=self.FEATURE_NAMES).rename_axis(Col.UID)

        combined = non_empty[0]
        for df in non_empty[1:]:
            combined = combined.join(df, how="outer")

        # Ensure all expected columns exist
        for col in self.FEATURE_NAMES:
            if col not in combined.columns:
                combined[col] = 0.0

        combined = combined[self.FEATURE_NAMES].fillna(0.0).astype(float)
        return combined
