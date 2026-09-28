"""Metrics Dashboard — Streamlit page."""
import streamlit as st
import pandas as pd
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