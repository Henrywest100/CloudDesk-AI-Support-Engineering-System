"""Error Log — Streamlit page."""
import streamlit as st
import pandas as pd
from metrics_logger import load_errors

st.set_page_config(page_title="Error Log", page_icon="🐛", layout="wide")

st.title("🐛 Error Log")
st.caption("Failed queries captured by the app — for debugging.")

errors = load_errors()

if not errors:
    st.success("✅ No errors logged — everything's working.")
    st.stop()

df = pd.DataFrame(errors)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp", ascending=False)

# ---- KPIs ----
c1, c2, c3 = st.columns(3)
c1.metric("Total Errors", len(df))
c2.metric("Unique Error Types", df["error_type"].nunique())
c3.metric("Last Error", df["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M"))

st.divider()

# ---- Error type breakdown ----
st.subheader("Error Type Distribution")
st.bar_chart(df["error_type"].value_counts())

# ---- Recent errors ----
st.subheader("Recent Errors")
for _, row in df.head(20).iterrows():
    with st.expander(
        f"❌ **{row['error_type']}** — {row['timestamp'].strftime('%Y-%m-%d %H:%M')} — {row['question'][:80]}"
    ):
        st.markdown(f"**Question:** {row['question']}")
        st.markdown(f"**Error type:** `{row['error_type']}`")
        st.markdown(f"**Message:** {row['error_message']}")
        if row.get("traceback"):
            st.code(row["traceback"], language="python")