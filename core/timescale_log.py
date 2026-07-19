# core/timescale_log.py
"""
TimescaleDB-backed event log.

Implements the same interface as FileEventLog (core/event_log.py) so it is a
drop-in replacement. Hypertable on `ts` gives O(log n) range queries and cheap
retention policies (drop_chunks older than N months).

Schema (auto-created on first use):

    CREATE EXTENSION IF NOT EXISTS timescaledb;

    CREATE TABLE IF NOT EXISTS events (
        id           BIGSERIAL,
        ts           TIMESTAMPTZ NOT NULL,
        type         TEXT        NOT NULL,
        version      INT         NOT NULL DEFAULT 1,
        symbol       TEXT,
        payload      JSONB       NOT NULL,
        ingested_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (id, ts)
    );

    SELECT create_hypertable('events', 'ts', if_not_exists => TRUE);

    CREATE INDEX IF NOT EXISTS events_type_ts_idx ON events (type, ts DESC);
    CREATE INDEX IF NOT EXISTS events_symbol_ts_idx ON events (symbol, ts DESC);
    CREATE INDEX IF NOT EXISTS events_payload_gin ON events USING GIN (payload);

Configure via:

    core:
      event_log:
        backend: timescale
        dsn: postgresql://cryptonaute:...@localhost:5432/cryptonaute
        retention_months: 24

Or set `TIMESCALE_DSN` in the environment.

Only requires `psycopg[binary]` (or `psycopg2-binary`) — declared as an optional
dependency. The class raises ImportError with a clear message if the driver is
missing, and the bus falls back to JSONL.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, Iterator, Optional

from core.event_log import EventLog

log = logging.getLogger("core.timescale")

_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS events (
    id           BIGSERIAL,
    ts           TIMESTAMPTZ NOT NULL,
    type         TEXT        NOT NULL,
    version      INT         NOT NULL DEFAULT 1,
    symbol       TEXT,
    payload      JSONB       NOT NULL,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, ts)
);

SELECT create_hypertable('events', 'ts', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS events_type_ts_idx   ON events (type, ts DESC);
CREATE INDEX IF NOT EXISTS events_symbol_ts_idx ON events (symbol, ts DESC);
CREATE INDEX IF NOT EXISTS events_payload_gin   ON events USING GIN (payload);
"""


def _import_driver():
    try:
        import psycopg  # type: ignore
        return psycopg, 3
    except ImportError:
        pass
    try:
        import psycopg2  # type: ignore
        return psycopg2, 2
    except ImportError as exc:
        raise ImportError(
            "TimescaleEventLog requires psycopg[binary] (or psycopg2-binary). "
            "Install with: pip install 'psycopg[binary]>=3.1'"
        ) from exc


class TimescaleEventLog(EventLog):
    def __init__(
        self,
        dsn: Optional[str] = None,
        *,
        retention_months: Optional[int] = None,
        create_schema: bool = True,
        batch_size: int = 1,
    ) -> None:
        self.dsn = dsn or os.environ.get("TIMESCALE_DSN")
        if not self.dsn:
            raise ValueError("TimescaleEventLog: no DSN (set TIMESCALE_DSN or pass dsn=)")
        self._driver, self._ver = _import_driver()
        self._lock = threading.Lock()
        self._batch: list[tuple] = []
        self._batch_size = max(1, int(batch_size))

        self._conn = self._driver.connect(self.dsn)
        self._conn.autocommit = True
        if create_schema:
            self._init_schema()
        if retention_months:
            self._apply_retention(retention_months)

    def _init_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(_SCHEMA_SQL)
        log.info("Timescale schema ready")

    def _apply_retention(self, months: int) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT add_retention_policy('events', INTERVAL %s, if_not_exists => TRUE);",
                (f"{int(months)} months",),
            )
        log.info("Timescale retention: %d months", months)

    def _normalize(self, event: Dict[str, Any]) -> tuple:
        ts = float(event.get("ts") or event.get("timestamp") or 0.0)
        etype = str(event.get("type", "unknown"))
        version = int(event.get("version", 1))
        payload = event.get("payload")
        if payload is None:
            payload = {k: v for k, v in event.items()
                       if k not in ("type", "ts", "timestamp", "version")}
        symbol = None
        if isinstance(payload, dict):
            symbol = payload.get("symbol") or payload.get("symbols")
            if isinstance(symbol, list):
                symbol = symbol[0] if symbol else None
        return (ts, etype, version, symbol, json.dumps(payload))

    def append(self, event: Dict[str, Any]) -> None:
        row = self._normalize(event)
        with self._lock:
            self._batch.append(row)
            if len(self._batch) >= self._batch_size:
                self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._batch:
            return
        rows = self._batch
        self._batch = []
        with self._conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO events (ts, type, version, symbol, payload) "
                "VALUES (to_timestamp(%s), %s, %s, %s, %s::jsonb)",
                rows,
            )

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def replay(self) -> Iterator[Dict[str, Any]]:
        return self.replay_from(0)

    def replay_from(self, offset: int = 0) -> Iterator[Dict[str, Any]]:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT id, ts, type, version, payload FROM events "
                "WHERE id > %s ORDER BY id ASC",
                (int(offset),),
            )
            for row in cur:
                rid, ts, etype, version, payload = row
                if isinstance(payload, str):
                    payload = json.loads(payload)
                yield {
                    "id": int(rid),
                    "ts": ts.timestamp() if hasattr(ts, "timestamp") else float(ts),
                    "type": etype,
                    "version": int(version or 1),
                    "payload": payload,
                }

    def count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM events")
            (n,) = cur.fetchone()
        return int(n)

    def query_since(self, ts: float, event_type: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        sql = "SELECT id, ts, type, version, payload FROM events WHERE ts >= to_timestamp(%s)"
        params = [float(ts)]
        if event_type:
            sql += " AND type = %s"
            params.append(event_type)
        sql += " ORDER BY id ASC"
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            for rid, ts_, etype, version, payload in cur:
                if isinstance(payload, str):
                    payload = json.loads(payload)
                yield {
                    "id": int(rid), "ts": ts_.timestamp(),
                    "type": etype, "version": int(version or 1),
                    "payload": payload,
                }

    def last_id(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute("SELECT COALESCE(MAX(id), 0) FROM events")
            (n,) = cur.fetchone()
        return int(n)

    def close(self) -> None:
        try:
            self.flush()
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass


def build_timescale_log(cfg: Dict[str, Any] | None) -> Optional[TimescaleEventLog]:
    cfg = cfg or {}
    if (cfg.get("backend") or "").lower() != "timescale":
        return None
    try:
        return TimescaleEventLog(
            dsn=cfg.get("dsn") or os.environ.get("TIMESCALE_DSN"),
            retention_months=cfg.get("retention_months"),
            batch_size=int(cfg.get("batch_size", 1)),
        )
    except ImportError as exc:
        log.warning("Timescale backend unavailable (%s) — falling back", exc)
        return None
    except Exception as exc:
        log.warning("Timescale connect failed (%s) — falling back", exc)
        return None
