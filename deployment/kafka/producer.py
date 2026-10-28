"""
deployment/kafka/producer.py
----------------------------
Simple Kafka producer that replays cleaned IEEE-CIS transactions to a topic
for real-time scoring.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

# Load environment variables from project .env if present
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DEFAULT_SOURCE = Path("/app/data/interim/cleaned_holdout_replay.parquet")
LEGACY_SOURCE = Path("/app/data/interim/cleaned_train.parquet")
DEFAULT_FEED = os.getenv("FEED_NAME", "default_feed")


def _to_json_safe(value: object) -> object:
    """Convert pandas/numpy scalars into JSON-serialisable primitives."""
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def iter_payloads(df: pd.DataFrame, limit: int, feed_name: str) -> Iterable[dict]:
    """Yield JSON-serialisable dicts, dropping the label column if present."""
    if "isFraud" in df.columns:
        df = df.drop(columns=["isFraud"])
    for _, row in df.head(limit).iterrows():
        payload = {
            key: _to_json_safe(value)
            for key, value in row.items()
            if not pd.isna(value)
        }
        payload["feed_name"] = feed_name
        yield payload


def wait_for_producer(bootstrap: str, timeout_seconds: float) -> KafkaProducer:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            return KafkaProducer(
                bootstrap_servers=bootstrap.split(","),
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
        except NoBrokersAvailable as exc:
            last_error = exc
            print(f"Waiting for Kafka producer connection to {bootstrap}...")
            time.sleep(2)

    raise RuntimeError(
        f"Kafka producer could not connect to {bootstrap} within {timeout_seconds:.0f}s"
    ) from last_error


def resolve_source_path(source: str | Path) -> Path:
    """Resolve the replay file, falling back to the legacy training feed if needed."""
    source_path = Path(source)
    if source_path.exists():
        return source_path

    if source_path == DEFAULT_SOURCE and LEGACY_SOURCE.exists():
        print(
            "Held-out replay source not found; falling back to legacy cleaned_train.parquet"
        )
        return LEGACY_SOURCE

    raise FileNotFoundError(f"Source data not found: {source_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay transactions to Kafka for scoring.")
    parser.add_argument(
        "--source",
        type=str,
        default=str(DEFAULT_SOURCE),
        help="Parquet/CSV file with cleaned transactions.",
    )
    parser.add_argument("--limit", type=int, default=200, help="Number of messages to send.")
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.1,
        help="Delay between messages (seconds).",
    )
    args = parser.parse_args()

    source_path = resolve_source_path(args.source)

    if source_path.suffix == ".parquet":
        df = pd.read_parquet(source_path)
    else:
        df = pd.read_csv(source_path)

    topic = os.getenv("KAFKA_TX_TOPIC", "fraud_transactions")
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    kafka_startup_timeout = float(os.getenv("KAFKA_STARTUP_TIMEOUT_SECONDS", "60"))

    producer = wait_for_producer(bootstrap, kafka_startup_timeout)

    sent = 0
    feed_name = os.getenv("FEED_NAME", DEFAULT_FEED)

    for payload in iter_payloads(df, args.limit, feed_name):
        producer.send(topic, value=payload)
        sent += 1
        if args.sleep > 0:
            time.sleep(args.sleep)

    producer.flush()
    print(f"Sent {sent} messages to topic '{topic}' via {bootstrap}")


if __name__ == "__main__":
    main()
