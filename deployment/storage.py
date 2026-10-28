"""
deployment/storage.py
---------------------
Lightweight PostgreSQL helper using SQLAlchemy for saving scored predictions
and reading them back for the dashboard.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.engine import Engine

# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------


def _build_postgres_url() -> str:
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return env_url

    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "frauddb")
    user = os.getenv("POSTGRES_USER", "frauduser")
    password = os.getenv("POSTGRES_PASSWORD", "fraudpass")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


def get_engine(echo: bool = False) -> Engine:
    return create_engine(_build_postgres_url(), echo=echo, future=True)


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

metadata = MetaData()

predictions = Table(
    "predictions",
    metadata,
    Column("id", PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
    Column("transaction_id", String(64), nullable=False),
    Column("model_version", String(64), nullable=False),
    Column("fraud_probability", Float, nullable=False),
    Column("fraud_label", Integer, nullable=False),
    Column("action", String(16), nullable=False),
    Column("feed_name", String(64), nullable=True),
    Column("latency_ms", Float, nullable=True),
    Column("transaction_dt", Float, nullable=True),
    Column("received_at", DateTime(timezone=True), server_default=func.now()),
    Column("payload", JSON, nullable=True),
)


def init_db(engine: Optional[Engine] = None) -> None:
    engine = engine or get_engine()
    metadata.create_all(engine)


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def save_prediction(
    record: Dict[str, Any],
    engine: Optional[Engine] = None,
) -> None:
    engine = engine or get_engine()
    with engine.begin() as conn:
        conn.execute(predictions.insert().values(**record))


def fetch_recent_predictions(
    limit: int = 100,
    engine: Optional[Engine] = None,
) -> List[Dict[str, Any]]:
    engine = engine or get_engine()
    stmt = (
        select(predictions)
        .order_by(predictions.c.received_at.desc())
        .limit(limit)
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    return [dict(row) for row in rows]
