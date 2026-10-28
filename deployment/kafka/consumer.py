"""Kafka consumer that scores incoming transactions through the FastAPI service."""

from __future__ import annotations

import json
import os
import time
import uuid

import httpx
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import NoBrokersAvailable

from deployment.storage import get_engine, init_db, save_prediction


def wait_for_consumer(topic: str, bootstrap: str, timeout_seconds: float) -> KafkaConsumer:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            return KafkaConsumer(
                topic,
                bootstrap_servers=bootstrap.split(","),
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            )
        except NoBrokersAvailable as exc:
            last_error = exc
            print(f"Waiting for Kafka consumer connection to {bootstrap}...")
            time.sleep(2)

    raise RuntimeError(f"Kafka consumer could not connect to {bootstrap} within {timeout_seconds:.0f}s") from last_error


def wait_for_producer(
    topic: str | None,
    bootstrap: str,
    timeout_seconds: float,
) -> KafkaProducer | None:
    if not topic:
        return None

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

    raise RuntimeError(f"Kafka producer could not connect to {bootstrap} within {timeout_seconds:.0f}s") from last_error


def score_via_api(
    payload: dict,
    api_url: str,
    retries: int = 3,
    timeout_seconds: float = 2.0,
) -> dict:
    """Call the FastAPI scoring endpoint with exponential backoff."""
    url = f"{api_url.rstrip('/')}/predict"
    last_error: Exception | None = None

    with httpx.Client(timeout=timeout_seconds) as client:
        for attempt in range(1, retries + 1):
            try:
                response = client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_error = exc
                if attempt == retries:
                    break
                time.sleep(0.25 * (2 ** (attempt - 1)))

    assert last_error is not None
    raise last_error


def wait_for_api(api_url: str, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    health_url = f"{api_url.rstrip('/')}/health"
    last_error: Exception | None = None

    with httpx.Client(timeout=2.0) as client:
        while time.time() < deadline:
            try:
                response = client.get(health_url)
                response.raise_for_status()
                return
            except Exception as exc:
                last_error = exc
                print(f"Waiting for Fraud API at {health_url}...")
                time.sleep(2)

    raise RuntimeError(f"Fraud API could not become healthy within {timeout_seconds:.0f}s") from last_error


def main() -> None:
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic_tx = os.getenv("KAFKA_TX_TOPIC", "fraud_transactions")
    topic_pred = os.getenv("KAFKA_PRED_TOPIC", "fraud_predictions")
    topic_dlq = os.getenv("KAFKA_DLQ_TOPIC", "fraud_dlq")
    api_url = os.getenv("FRAUD_API_URL", "http://localhost:8000")
    retries = int(os.getenv("FRAUD_API_RETRIES", "3"))
    timeout_seconds = float(os.getenv("FRAUD_API_TIMEOUT_SECONDS", "2.0"))
    kafka_startup_timeout = float(os.getenv("KAFKA_STARTUP_TIMEOUT_SECONDS", "60"))

    consumer = wait_for_consumer(topic_tx, bootstrap, kafka_startup_timeout)
    producer_pred = wait_for_producer(topic_pred, bootstrap, kafka_startup_timeout)
    producer_dlq = wait_for_producer(topic_dlq, bootstrap, kafka_startup_timeout)
    wait_for_api(api_url, kafka_startup_timeout)

    engine = get_engine()
    init_db(engine)

    print(f"Consumer listening on topic '{topic_tx}' (bootstrap: {bootstrap})")
    for msg in consumer:
        payload = msg.value
        txn_id = str(payload.get("TransactionID") or uuid.uuid4())
        feed_name = payload.get("feed_name") or os.getenv("FEED_NAME", "default_feed")

        start_t = time.perf_counter()
        try:
            prediction = score_via_api(
                payload=payload,
                api_url=api_url,
                retries=retries,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start_t) * 1000
            if producer_dlq:
                producer_dlq.send(
                    topic_dlq,
                    value={
                        "transaction_id": txn_id,
                        "feed_name": feed_name,
                        "latency_ms": latency_ms,
                        "error": str(exc),
                        "payload": payload,
                    },
                )
                producer_dlq.flush()
            print(f"[dlq] feed={feed_name} txn={txn_id} error={exc}")
            continue

        latency_ms = (time.perf_counter() - start_t) * 1000
        record = {
            "transaction_id": txn_id,
            "model_version": prediction["model_version"],
            "fraud_probability": prediction["fraud_probability"],
            "fraud_label": prediction["fraud_label"],
            "action": prediction["action"],
            "feed_name": feed_name,
            "latency_ms": latency_ms,
            "transaction_dt": (
                float(payload.get("TransactionDT"))
                if payload.get("TransactionDT") is not None
                else None
            ),
            "payload": payload,
        }

        save_prediction(record, engine)
        print(
            f"[scored] feed={feed_name} txn={txn_id} "
            f"prob={record['fraud_probability']:.4f} action={record['action']} "
            f"latency_ms={latency_ms:.1f}"
        )

        if producer_pred:
            producer_pred.send(
                topic_pred,
                value={
                    "transaction_id": txn_id,
                    "feed_name": feed_name,
                    "latency_ms": latency_ms,
                    **payload,
                    **prediction,
                },
            )
            producer_pred.flush()


if __name__ == "__main__":
    main()
