"""Metrics Dashboard — Streamlit page."""
import streamlit as st
import pandas as pd
from datetime import timedelta
from metrics_logger import load_metrics

st.set_page_config(page_title="CloudDesk Metrics", page_icon="📊", layout="wide")

st.title("📊 CloudDesk AI — Usage Metrics")

records = load_metrics()

if not records:
    st.info("No queries logged yet. Ask a question on the main page first.")
    st.stop()

df = pd.DataFrame(records)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

# ---- Top KPIs ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Queries", len(df))
c2.metric("Avg Confidence", f"{df['confidence'].mean()*100:.1f}%")
c3.metric("Escalation Rate", f"{df['escalated'].mean()*100:.1f}%")
c4.metric("Avg Latency", f"{df['latency_s'].mean():.2f}s")

st.divider()

# ---- Drift detection ----
st.subheader("📉 Drift Detection")

def compute_drift(df_in):
    if len(df_in) < 20:
        return None

    df_d = df_in.copy()
    df_d["timestamp"] = pd.to_datetime(df_d["timestamp"])
    now = pd.Timestamp.utcnow()

    this_week = df_d[df_d["timestamp"] >= now - timedelta(days=7)]
    last_week = df_d[
        (df_d["timestamp"] >= now - timedelta(days=14)) &
        (df_d["timestamp"] < now - timedelta(days=7))
    ]

    if len(this_week) == 0 or len(last_week) == 0:
        return None

    return {
        "this_week": round(this_week["confidence"].mean(), 4),
        "last_week": round(last_week["confidence"].mean(), 4),
        "delta": round(this_week["confidence"].mean() - last_week["confidence"].mean(), 4),
    }

drift = compute_drift(df)

if drift is None:
    st.info(
        f"⏳ Not enough data for drift analysis "
        f"(need 20+ queries across 2 weeks — currently {len(df)})."
    )
else:
    delta_pct = drift["delta"] * 100
    d1, d2, d3 = st.columns(3)
    d1.metric("This Week Avg", f"{drift['this_week']*100:.1f}%")
    d2.metric("Last Week Avg", f"{drift['last_week']*100:.1f}%")
    d3.metric("Change", f"{delta_pct:+.1f}%")

    if delta_pct < -5:
        st.error(f"🚨 Confidence dropped {abs(delta_pct):.1f}% — possible drift.")
    elif delta_pct < -2:
        st.warning(f"⚠️ Slight drop of {abs(delta_pct):.1f}% — monitor.")
    else:
        st.success(f"✅ Stable ({delta_pct:+.1f}% week-over-week).")

st.divider()

# ---- Queries over time ----
st.subheader("Queries Over Time")
df["date"] = df["timestamp"].dt.date
daily = df.groupby("date").size().reset_index(name="queries")
st.line_chart(daily.set_index("date"))

# ---- Confidence distribution ----
st.subheader("Confidence Distribution")
st.bar_chart(df["confidence"].value_counts(bins=10).sort_index())

# ---- Vector store split ----
st.subheader("Vector Store Used")
st.bar_chart(df["vector_store"].value_counts())

# ---- Recent queries ----
st.subheader("Recent Queries")
st.dataframe(
    df[["timestamp", "question", "confidence", "escalated", "latency_s"]].tail(20),
    use_container_width=True,
)