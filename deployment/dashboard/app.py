"""
deployment/dashboard/app.py
Modern dark dashboard for real-time fraud predictions.
Keeps the existing backend (PostgreSQL + Kafka + FastAPI) and improves UI/UX.
"""
# ruff: noqa: E501

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Tuple

import altair as alt
import httpx
import pandas as pd
import streamlit as st
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

# ---------------------------------------------------------------------------
# Compatibility shim: st.fragment was added in Streamlit 1.37.
# When running on an older version fall back to a no-op decorator so the
# dashboard still works (without partial-fragment reruns).
# ---------------------------------------------------------------------------
if not hasattr(st, "fragment"):
    def _fragment_shim(func=None, *, run_every=None):  # noqa: D401
        """No-op replacement for st.fragment on Streamlit < 1.37."""
        if func is None:
            return lambda fn: fn          # called as @st.fragment(...)
        return func                       # called as @st.fragment

    st.fragment = _fragment_shim          # type: ignore[attr-defined]

from deployment.storage import get_engine, init_db


def render_database_error(exc: OperationalError) -> None:
    """Render a concise database connection error without exposing a traceback."""
    configured_host = os.getenv("POSTGRES_HOST", "localhost")
    st.error(
        "Dashboard cannot connect to PostgreSQL. "
        f"It is currently configured with POSTGRES_HOST={configured_host!r}."
    )
    st.info(
        "Use POSTGRES_HOST=postgres only when Streamlit is started by docker compose. "
        "Use POSTGRES_HOST=localhost when Streamlit runs on your host machine."
    )
    with st.expander("Connection details"):
        st.code(str(exc.orig if getattr(exc, "orig", None) else exc))

# -----------------------------------------------------------------------------
# Page config and styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FraudGuard · Live Transaction Monitor",
    layout="wide",
    page_icon="🛡️",
)

# Design tokens (deep navy / teal, closer to the reference mock)
PALETTE = {
    "bg": "#0d1324",
    "panel": "#111a2e",
    "border": "#1f2a44",
    "text": "#e5ebff",
    "muted": "#9db2d4",
    "accent": "#00c2ff",     # bright cyan accents
    "accent2": "#fbbf24",    # warm secondary accent
    "danger": "#ef4444",
    "warn": "#f59e0b",
    "success": "#22c55e",
}

# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    :root {{
      --bg: {PALETTE["bg"]}; --panel:{PALETTE["panel"]}; --border:{PALETTE["border"]};
      --text:{PALETTE["text"]}; --muted:{PALETTE["muted"]}; --accent:{PALETTE["accent"]};
      --accent2:{PALETTE["accent2"]}; --danger:{PALETTE["danger"]}; --warn:{PALETTE["warn"]}; --success:{PALETTE["success"]};
    }}
    body, .stApp {{background-color: var(--bg) !important; color: var(--text) !important;}}
    .block-container {{padding-top: 0.3rem !important; padding-bottom: 0.3rem !important;}}
    div[data-testid="stSidebar"] {{background: #0e1a30 !important; color: var(--text) !important; border-right: 1px solid var(--border);}}
    div[data-testid="stSidebar"] .stSelectbox, div[data-testid="stSidebar"] .stSlider, div[data-testid="stSidebar"] .stMultiSelect {{color: var(--text) !important;}}
    .side-title {{font-weight: 900; font-size: 1.3rem; color: var(--text); line-height: 1.3;}}
    .nav-item {{padding: 0.45rem 0.65rem; border-radius: 10px; color: var(--muted);}}
    .nav-active {{background: #13213a; color: var(--text); border: 1px solid var(--border);}}
    .card {{background: var(--panel); border: 1px solid var(--border); border-radius: 16px; padding: 16px 18px; box-shadow: 0 10px 30px rgba(0,0,0,0.25);}}
    .kpi-label {{color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: 0.6px;}}
    .kpi-value {{font-size: 32px; font-weight: 900; color: var(--text);}}
    .kpi-sub {{color: var(--muted); font-size: 12px;}}
    .badge {{border:1px solid var(--border); padding:6px 12px; border-radius:999px; color:var(--muted); font-size:12px; background: #0d162b;}}
    .badge-live {{color:#fef3c7; background: #3b2d0a; border-color:#a16207;}}
    .accent-bar {{height:4px; width:100%; border-radius: 999px; background: linear-gradient(90deg, var(--accent), var(--accent2)); margin-top:8px;}}
    input[type=range] {{accent-color: {PALETTE["accent"]};}}
    div[data-baseweb="select"] > div {{background: #0d162b !important; border-color: var(--border) !important; color: var(--text) !important;}}
    .stButton>button {{background: #13213a; border:1px solid var(--border); color: var(--text);}}
    .stMetricLabel, .stMetricDelta {{color: var(--muted) !important;}}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Data access
# -----------------------------------------------------------------------------
@st.cache_data(ttl=5)
def load_predictions() -> pd.DataFrame:
    """Load all predictions from Postgres, newest first."""
    engine = get_engine()
    init_db(engine)  # idempotent
    query = text(
        """
        SELECT transaction_id,
               model_version,
               fraud_probability,
               fraud_label,
               action,
               transaction_dt,
               received_at,
               payload
        FROM predictions
        ORDER BY received_at DESC
        """
    )
    df = pd.read_sql_query(query, engine)
    if not df.empty:
        df["received_at"] = pd.to_datetime(df["received_at"], utc=True).dt.tz_convert(None)
        payload_df = pd.json_normalize(df["payload"]).add_prefix("payload.")
        df = pd.concat([df.reset_index(drop=True), payload_df], axis=1)
    return df


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def apply_filters(
    df: pd.DataFrame,
    actions: Tuple[str, ...],
    fraud_label: str,
    min_prob: float,
    hours_window: int,
) -> pd.DataFrame:
    if df.empty:
        return df
    cutoff = datetime.now() - timedelta(hours=hours_window)
    df = df[df["received_at"] >= cutoff]
    if actions:
        df = df[df["action"].isin(actions)]
    if fraud_label != "All":
        label_val = 1 if fraud_label == "Fraud only" else 0
        df = df[df["fraud_label"] == label_val]
    if min_prob > 0:
        df = df[df["fraud_probability"] >= min_prob]
    return df


def compute_kpis(df: pd.DataFrame, threshold: float) -> Dict[str, float]:
    if df.empty:
        return dict(
            total=0, tx_per_min=0, avg=0, max=0, alerts=0, block=0, review=0, fraud_rate=0, threshold=threshold
        )
    total = len(df)
    minutes = max((df["received_at"].max() - df["received_at"].min()).total_seconds() / 60, 1)
    tx_per_min = total / minutes
    alerts = int(df["fraud_label"].sum())
    block = int((df["action"] == "BLOCK").sum())
    review = int((df["action"] == "REVIEW").sum())
    return {
        "total": total,
        "tx_per_min": tx_per_min,
        "avg": float(df["fraud_probability"].mean()),
        "max": float(df["fraud_probability"].max()),
        "alerts": alerts,
        "block": block,
        "review": review,
        "fraud_rate": alerts / total if total else 0,
        "threshold": threshold,
    }


def render_live_stream(
    actions: Tuple[str, ...],
    fraud_label: str,
    min_prob: float,
    hours_window: int,
    refresh_seconds: int,
    throughput_window_minutes: int,
    threshold: float,
) -> None:
    """Refresh only the live section instead of reloading the whole page."""

    @st.fragment(run_every=refresh_seconds)
    def live_stream_fragment() -> None:
        header = st.container()
        with header:
            left, right = st.columns([4, 2])
            with left:
                st.markdown("## Live Transaction Monitor")
                st.caption("Kafka -> FastAPI -> PostgreSQL -> Streamlit")
            with right:
                st.markdown(
                    "<div style='display:flex; gap:8px; justify-content:flex-end;'>"
                    "<span class='badge'>IEEE-CIS · Vesta Corp</span>"
                    "<span class='badge badge-live'>● LIVE</span>"
                    "</div>",
                    unsafe_allow_html=True,
                )

        try:
            raw_df = load_predictions()
        except OperationalError as exc:
            render_database_error(exc)
            return
        df = apply_filters(raw_df, actions, fraud_label, min_prob, hours_window)
        if df.empty:
            st.info("No predictions yet for the selected filters. Run the producer/consumer or widen the window.")
            st.markdown(
                f"*Last refresh:* {datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')} - "
                "0 rows in view."
            )
            return

        amount_col = "payload.TransactionAmt"
        tx_count = len(df)
        fraud_count = int(df["fraud_label"].sum())
        fraud_rate = fraud_count / tx_count if tx_count else 0
        avg_amount = float(
            pd.to_numeric(df.get(amount_col, pd.Series([])), errors="coerce").mean() or 0
        )
        high_risk = int((df["fraud_probability"] >= threshold).sum())

        kpi_cols = st.columns(5)
        kpi_items = [
            ("Total Transactions", f"{tx_count:,.0f}", ""),
            ("Fraud Count", f"{fraud_count:,}", f"{fraud_rate:.1%} rate"),
            ("Fraud Rate", f"{fraud_rate*100:.1f}%", ""),
            ("Average Transaction Amount", f"${avg_amount:,.0f}", ""),
            ("High-Risk Alerts", f"{high_risk:,}", "Active cases"),
        ]
        for col, (label, value, sub) in zip(kpi_cols, kpi_items):
            with col:
                st.markdown(
                    f"""
                    <div class="card">
                      <div class="kpi-label">{label}</div>
                      <div class="kpi-value">{value}</div>
                      <div class="kpi-sub">{sub}</div>
                      <div class="accent-bar"></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        throughput_chart, throughput_df = build_throughput_chart(
            raw_df,
            window_minutes=throughput_window_minutes,
        )

        hist = (
            alt.Chart(df)
            .mark_bar(color=PALETTE["accent2"])
            .encode(
                x=alt.X("fraud_probability:Q", bin=alt.Bin(maxbins=25), title="Fraud score"),
                y=alt.Y("count()", title="Frequency"),
                tooltip=[alt.Tooltip("count()", title="Count")],
            )
            .properties(height=220)
            .configure_axis(labelColor=PALETTE["muted"], titleColor=PALETTE["muted"])
        )

        st.markdown("### Real-time throughput")
        latest_tpm = float(throughput_df["tpm"].iloc[-1]) if not throughput_df.empty else 0.0
        peak_tpm = float(throughput_df["tpm"].max()) if not throughput_df.empty else 0.0
        st.caption(
            f"Rolling last {throughput_window_minutes} minutes. "
            f"Updates every {refresh_seconds}s without reloading the full page. "
            f"Current throughput: {latest_tpm:.1f} TPM, peak: {peak_tpm:.1f} TPM."
        )
        st.altair_chart(throughput_chart, use_container_width=True)

        st.markdown("### Transaction investigation table")
        st.caption(f"{len(df)} rows shown")
        render_table(df)

        st.markdown("### Fraud score distribution")
        st.altair_chart(hist, use_container_width=True)

        st.markdown(
            f"*Last refresh:* {datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')} - "
            f"{len(df)} rows in view."
        )

    live_stream_fragment()
def build_throughput_chart(
    df: pd.DataFrame,
    window_minutes: int,
    bucket_seconds: int = 10,
):
    """Build a rolling throughput chart from recent prediction timestamps."""
    freq = f"{bucket_seconds}s"
    end_time = pd.Timestamp.now(tz="UTC").tz_localize(None)
    start_time = end_time - pd.Timedelta(minutes=window_minutes)
    bucket_index = pd.date_range(
        start=start_time.floor(freq),
        end=end_time.ceil(freq),
        freq=freq,
    )

    throughput_df = pd.DataFrame({"bucket": bucket_index, "events": 0})
    if not df.empty and "received_at" in df.columns:
        recent = (
            df[["received_at"]]
            .dropna()
            .loc[lambda frame: frame["received_at"].between(start_time, end_time)]
            .set_index("received_at")
            .sort_index()
            .resample(freq)
            .size()
            .reindex(bucket_index, fill_value=0)
            .rename("events")
            .reset_index()
            .rename(columns={"index": "bucket"})
        )
        throughput_df = recent

    throughput_df["tps"] = throughput_df["events"] / bucket_seconds
    throughput_df["tpm"] = throughput_df["tps"] * 60.0

    chart = (
        alt.Chart(throughput_df)
        .mark_line(point=True, color=PALETTE["accent"])
        .encode(
            x=alt.X("bucket:T", title="Time"),
            y=alt.Y("tpm:Q", title="Transactions / min"),
            tooltip=[
                alt.Tooltip("bucket:T", title="Bucket"),
                alt.Tooltip("events:Q", title="Transactions"),
                alt.Tooltip("tps:Q", title="TPS", format=".2f"),
                alt.Tooltip("tpm:Q", title="TPM", format=".1f"),
            ],
        )
        .properties(height=260)
        .configure_axis(labelColor=PALETTE["muted"], titleColor=PALETTE["muted"])
    )
    return chart, throughput_df


def render_kpi_cards(kpis: Dict[str, float]):
    cards = st.columns(5)
    items = [
        ("Transactions / min", f"{kpis['tx_per_min']:.0f}", "current window"),
        ("Fraud detected", kpis["alerts"], f"{kpis['fraud_rate']:.1%} rate"),
        ("Max probability", f"{kpis['max']:.2f}", f"avg {kpis['avg']:.2f}"),
        ("Block actions", kpis["block"], f"review: {kpis['review']}"),
        ("Threshold", f"{kpis['threshold']:.2f}", "applied in scoring"),
    ]
    for col, (label, value, sub) in zip(cards, items):
        with col:
            st.markdown(
                f"""
                <div class="card">
                  <div class="kpi-label">{label}</div>
                  <div class="kpi-value">{value}</div>
                  <div class="kpi-sub">{sub}</div>
                  <div class="accent-bar"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_rate_chart(df: pd.DataFrame):
    if df.empty:
        st.info("No data to plot.")
        return
    chart_df = df.copy()
    chart_df["hour"] = chart_df["received_at"].dt.floor("H")
    agg = (
        chart_df.groupby("hour")
        .agg(fraud_rate=("fraud_label", "mean"), count=("fraud_label", "size"))
        .reset_index()
    )
    chart = alt.Chart(agg).mark_line(point=True, color=PALETTE["accent"]).encode(
        x=alt.X("hour:T", title=None),
        y=alt.Y("fraud_rate:Q", title="Fraud rate"),
        tooltip=["hour:T", alt.Tooltip("fraud_rate:Q", format=".2%"), "count:Q"],
    )
    st.altair_chart(
        chart.properties(height=260).configure_axis(labelColor=PALETTE["muted"], titleColor=PALETTE["muted"]),
        use_container_width=True,
    )


def render_action_bar(df: pd.DataFrame):
    if df.empty:
        return
    agg = (
        df.groupby("action")
        .agg(avg_prob=("fraud_probability", "mean"), n=("action", "size"))
        .reset_index()
    )
    base = alt.Chart(agg).encode(
        x=alt.X("avg_prob:Q", title="Avg probability"),
        y=alt.Y("action:N", title=None, sort="-x"),
        color=alt.Color("action:N", scale=alt.Scale(range=[PALETTE["success"], PALETTE["warn"], PALETTE["danger"]])),
        tooltip=["action", alt.Tooltip("avg_prob:Q", format=".3f"), "n:Q"],
    )
    st.altair_chart(
        base.mark_bar().properties(height=180).configure_axis(labelColor=PALETTE["text"], titleColor=PALETTE["muted"]),
        use_container_width=True,
    )


def render_table(df: pd.DataFrame):
    if df.empty:
        st.info("No rows match the current filters.")
        return
    table_df = df.sort_values("received_at", ascending=False).copy()
    table_df["Score"] = table_df["fraud_probability"].map(lambda x: f"{x:.3f}")
    table_df["Label"] = table_df["fraud_label"].map({0: "Legit", 1: "Fraud"})
    table_df["Timestamp"] = table_df["received_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    table_df["Amount"] = table_df.get("payload.TransactionAmt", pd.Series(["—"] * len(table_df)))
    # prettify amount display if numeric
    try:
        table_df["Amount"] = pd.to_numeric(table_df["Amount"], errors="coerce").map(lambda x: f"${x:,.2f}" if pd.notnull(x) else "—")
    except Exception:
        pass
    table_df["Card type"] = (
        table_df.get("payload.card4", pd.Series(["?"] * len(table_df))).fillna("?")
        + "/"
        + table_df.get("payload.card6", pd.Series(["?"] * len(table_df))).fillna("?")
    )
    show_cols = ["Timestamp", "transaction_id", "Amount", "Card type", "Score", "Label", "action"]
    table_df = table_df[show_cols].rename(columns={"transaction_id": "Tx ID", "action": "Action"})

    def style_actions(val):
        colors = {"BLOCK": PALETTE["danger"], "REVIEW": PALETTE["warn"], "ALLOW": PALETTE["success"]}
        return f"color:{colors.get(val, 'white')}; font-weight:700"

    st.dataframe(
        table_df.style.applymap(style_actions, subset=["Action"]),
        use_container_width=True,
        hide_index=True,
    )


def status_dot(color: str, text: str):
    st.markdown(
        f"<span style='color:{color}; font-size:18px;'>●</span> <span style='color:#e5e7eb;'>{text}</span>",
        unsafe_allow_html=True,
    )


def _clean_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if pd.isna(value):
            cleaned[key] = None
        elif hasattr(value, "item"):
            cleaned[key] = value.item()
        else:
            cleaned[key] = value
    return cleaned


def call_explain_api(payload: dict[str, Any]) -> dict[str, Any] | None:
    api_url = os.getenv("FRAUD_API_URL", "http://localhost:8000").rstrip("/")
    try:
        response = httpx.post(f"{api_url}/predict/explain", json=payload, timeout=10.0)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.error(f"Explain API call failed: {exc}")
        return None


def load_saved_metrics() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    valid_path = "reports/metrics/metrics_valid.json"
    test_path = "reports/metrics/metrics_test.json"
    valid_metrics = None
    test_metrics = None
    if os.path.exists(valid_path):
        with open(valid_path, "r", encoding="utf-8") as fh:
            valid_metrics = json.load(fh)
    if os.path.exists(test_path):
        with open(test_path, "r", encoding="utf-8") as fh:
            test_metrics = json.load(fh)
    return valid_metrics, test_metrics


# -----------------------------------------------------------------------------
# Main UI
# -----------------------------------------------------------------------------
def main():
    df = pd.DataFrame()

    with st.sidebar:
        st.markdown(
            "<div class='side-title'>🛡️ FRAUDGUARD<br><span style='color:var(--muted); font-size:12px;'>IEEE-CIS · Transaction Fraud</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown("#### MONITOR")
        section = st.radio(
            "Navigation",
            (
                "Live stream",
                "Model performance",
                "SHAP explainer",
            ),
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("#### FILTERS")
        hours_window = st.selectbox(
            "Time window",
            [1, 6, 12, 24],
            index=3,
            format_func=lambda x: f"Last {x}h",
        )
        actions = st.multiselect(
            "Action",
            ["ALLOW", "REVIEW", "BLOCK"],
            default=["ALLOW", "REVIEW", "BLOCK"],
        )
        fraud_label = st.selectbox("Fraud label", ["All", "Fraud only", "Legit only"])
        min_prob = st.slider("Min probability", 0.0, 1.0, 0.0, step=0.01)

        st.markdown("#### LIVE REFRESH")
        refresh_seconds = st.selectbox(
            "Refresh every",
            [5, 10, 15, 30],
            index=1,
            format_func=lambda value: f"{value}s",
            disabled=section != "Live stream",
        )
        throughput_window_minutes = st.selectbox(
            "Throughput window",
            [5, 10],
            index=1,
            format_func=lambda value: f"Last {value} min",
            disabled=section != "Live stream",
        )

        st.markdown("#### THRESHOLD")
        threshold = float(os.getenv("FRAUD_THRESHOLD", "0.50"))
        st.slider(
            "Score",
            0.0,
            1.0,
            threshold,
            step=0.01,
            disabled=True,
            label_visibility="collapsed",
        )
        st.caption(f"Score ≥ {threshold:.2f}")

        st.markdown("#### STATUS")
        status_dot(PALETTE["success"], "Kafka connected (configured)")
        status_dot(PALETTE["success"], f"Model live (threshold {threshold:.2f})")

    if section == "Live stream":
        render_live_stream(
            actions=tuple(actions),
            fraud_label=fraud_label,
            min_prob=min_prob,
            hours_window=hours_window,
            refresh_seconds=refresh_seconds,
            throughput_window_minutes=throughput_window_minutes,
            threshold=threshold,
        )
        return

    if section == "Model performance":
        st.markdown("## Model performance")
        valid_metrics, test_metrics = load_saved_metrics()
        if not valid_metrics and not test_metrics:
            st.info("No saved evaluation metrics found yet. Run the training pipeline first.")
        else:
            cols = st.columns(5)
            metric_source = valid_metrics or {}
            items = [
                ("ROC-AUC", metric_source.get("roc_auc")),
                ("PR-AUC", metric_source.get("pr_auc")),
                ("Precision", metric_source.get("precision")),
                ("Recall", metric_source.get("recall")),
                ("F1", metric_source.get("f1")),
            ]
            for col, (label, value) in zip(cols, items):
                with col:
                    st.metric(label, f"{value:.4f}" if isinstance(value, (int, float)) else "n/a")

            shap_path = "reports/metrics/shap_global_importance_valid.csv"
            if os.path.exists(shap_path):
                shap_df = pd.read_csv(shap_path).head(15)
                shap_chart = (
                    alt.Chart(shap_df)
                    .mark_bar(color=PALETTE["accent"])
                    .encode(
                        x=alt.X("mean_abs_shap:Q", title="Mean |SHAP|"),
                        y=alt.Y("feature:N", sort="-x", title=None),
                        tooltip=["feature", alt.Tooltip("mean_abs_shap:Q", format=".4f")],
                    )
                    .properties(height=360)
                    .configure_axis(labelColor=PALETTE["muted"], titleColor=PALETTE["muted"])
                )
                st.markdown("### Global SHAP importance")
                st.altair_chart(shap_chart, use_container_width=True)

    elif section == "SHAP explainer":
        st.markdown("## SHAP explainer")
        try:
            df = load_predictions()
        except OperationalError as exc:
            render_database_error(exc)
            return
        if df.empty:
            st.info("No predictions available yet.")
            return
        options_df = df[["transaction_id", "payload"]].dropna(subset=["payload"]).head(100).copy()
        if options_df.empty:
            st.info("No payloads available for explanation.")
            return

        tx_id = st.selectbox("Transaction", options_df["transaction_id"].tolist())
        selected = options_df.loc[options_df["transaction_id"] == tx_id].iloc[0]
        explain_result = call_explain_api(_clean_payload(selected["payload"]))
        if not explain_result:
            return

        score_cols = st.columns(4)
        score_cols[0].metric("Fraud score", f"{explain_result['fraud_probability']:.3f}")
        score_cols[1].metric("Action", explain_result["action"])
        score_cols[2].metric("Threshold", f"{explain_result['threshold']:.2f}")
        score_cols[3].metric("Review threshold", f"{explain_result['review_threshold']:.2f}")

        contrib_df = pd.DataFrame(explain_result["top_features"])
        if contrib_df.empty:
            st.info("No SHAP contributions available.")
            return

        contrib_df["direction"] = contrib_df["shap_value"].apply(
            lambda value: "Increase risk" if value >= 0 else "Reduce risk"
        )
        shap_chart = (
            alt.Chart(contrib_df)
            .mark_bar()
            .encode(
                x=alt.X("shap_value:Q", title="SHAP value"),
                y=alt.Y("feature:N", sort="-x", title=None),
                color=alt.Color(
                    "direction:N",
                    scale=alt.Scale(
                        domain=["Increase risk", "Reduce risk"],
                        range=[PALETTE["danger"], PALETTE["success"]],
                    ),
                ),
                tooltip=[
                    "feature",
                    alt.Tooltip("feature_value:N", title="Feature value"),
                    alt.Tooltip("shap_value:Q", format=".4f"),
                ],
            )
            .properties(height=360)
            .configure_axis(labelColor=PALETTE["muted"], titleColor=PALETTE["muted"])
        )
        st.altair_chart(shap_chart, use_container_width=True)
        st.dataframe(contrib_df, use_container_width=True, hide_index=True)

    st.markdown(
        f"*Last refresh:* {datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')} - "
        f"{len(df)} rows in view."
    )
if __name__ == "__main__":
    main()
