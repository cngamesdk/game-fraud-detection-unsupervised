from __future__ import annotations

import time
from collections.abc import AsyncIterator

import aiomysql
from loguru import logger

from config import settings

_pool: aiomysql.Pool | None = None


async def get_pool() -> aiomysql.Pool:
    """Return the singleton connection pool, creating it on first call."""
    global _pool
    if _pool is None:
        _pool = await aiomysql.create_pool(
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            db=settings.MYSQL_DB,
            minsize=2,
            maxsize=settings.MYSQL_POOL_SIZE,
            autocommit=True,
            charset="utf8mb4",
        )
    return _pool


async def close_pool() -> None:
    """Close the pool on application shutdown."""
    global _pool
    if _pool is not None:
        _pool.close()
        await _pool.wait_closed()
        _pool = None


def _log_sql(sql: str, params: tuple | None, rows: int, elapsed_ms: float) -> None:
    """Log SQL execution with params, row count and timing."""
    # Truncate long SQL for readability
    sql_short = sql[:500] + "..." if len(sql) > 500 else sql
    params_short = str(params)[:200] + "..." if params and len(str(params)) > 200 else str(params)
    logger.debug(
        "SQL {ms:.0f}ms rows={rows} | {sql} | params={params}",
        ms=elapsed_ms,
        rows=rows,
        sql=sql_short,
        params=params_short,
    )


async def execute_query(sql: str, params: tuple | None = None) -> list[dict]:
    """Execute a read query and return a list of dicts (for small result sets)."""
    pool = await get_pool()
    t0 = time.perf_counter()
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(sql, params)
            result = await cur.fetchall()
    _log_sql(sql, params, len(result), (time.perf_counter() - t0) * 1000)
    return result


async def execute_query_batched(
    sql: str,
    params: tuple | None = None,
    batch_size: int | None = None,
) -> AsyncIterator[list[dict]]:
    """
    Execute a query and yield results in batches using server-side streaming cursor.

    Uses SSDictCursor (unbuffered) so MySQL sends rows on demand instead of
    loading the entire result set into memory at once.

    If the generator is not fully consumed (error, cancellation, early exit),
    the underlying connection is closed instead of being returned to the pool,
    because an SSDictCursor leaves the connection in an indeterminate protocol
    state when interrupted mid-stream.
    """
    size = batch_size or settings.QUERY_BATCH_SIZE
    pool = await get_pool()
    conn = await pool.acquire()
    consumed = False
    total_rows = 0
    t0 = time.perf_counter()
    try:
        async with conn.cursor(aiomysql.SSDictCursor) as cur:
            await cur.execute(sql, params)
            while True:
                rows = await cur.fetchmany(size)
                if not rows:
                    consumed = True
                    break
                total_rows += len(rows)
                yield rows
    finally:
        _log_sql(sql, params, total_rows, (time.perf_counter() - t0) * 1000)
        if not consumed and not conn.closed:
            conn.close()
        pool.release(conn)
