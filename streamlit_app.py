"""
CloudDesk AI Support Engineer — Streamlit UI
"""

import os
import time
import streamlit as st
from huggingface_hub import InferenceClient
from retrieval_pipeline import load_config, get_vector_store, run_rag_pipeline
from metrics_logger import log_query, load_metrics, log_error, load_errors

# -------------------------------------------------------------
# Page config — MUST be the first Streamlit command
# -------------------------------------------------------------
st.set_page_config(
    page_title="CloudDesk AI Support Engineer",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------
# Custom CSS
# -------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* Page background */
.stApp {
    background: #f6f8fc;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #0b1220 0%, #1e3a8a 45%, #3b82f6 100%);
    padding: 40px 44px;
    border-radius: 20px;
    color: #ffffff;
    margin-bottom: 26px;
    box-shadow: 0 20px 50px rgba(15,23,42,0.25);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    top: -60%; right: -15%;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(255,255,255,0.18) 0%, transparent 65%);
}
.hero h1 {
    margin: 0;
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -0.8px;
    color: #ffffff;
}
.hero p {
    margin: 12px 0 0;
    font-size: 16px;
    color: rgba(255,255,255,0.92);
    line-height: 1.55;
    max-width: 720px;
}
.badges { margin-top: 20px; display: flex; flex-wrap: wrap; gap: 8px; }
.badge {
    display: inline-block;
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.22);
    padding: 6px 14px;
    border-radius: 24px;
    font-size: 12.5px;
    color: #ffffff;
    font-weight: 500;
    backdrop-filter: blur(8px);
}

/* Chat messages */
.user-msg {
    background: linear-gradient(135deg, #2563eb, #3b82f6);
    color: white;
    padding: 14px 18px;
    border-radius: 16px 16px 4px 16px;
    margin: 10px 0 10px auto;
    max-width: 80%;
    font-size: 14.5px;
    line-height: 1.6;
    box-shadow: 0 4px 12px rgba(37,99,235,0.25);
    width: fit-content;
}
.assistant-msg {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    padding: 18px 22px;
    border-radius: 4px 16px 16px 16px;
    margin: 10px auto 20px 0;
    max-width: 92%;
    font-size: 14.5px;
    line-height: 1.7;
    box-shadow: 0 4px 16px rgba(15,23,42,0.06);
    width: fit-content;
}

/* Status banner */
.banner-ok {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border-left: 5px solid #10b981;
    padding: 14px 20px;
    border-radius: 10px;
    margin-bottom: 12px;
    color: #065f46;
    font-weight: 600;
    font-size: 14.5px;
}
.banner-esc {
    background: linear-gradient(135deg, #fef2f2, #fee2e2);
    border-left: 5px solid #ef4444;
    padding: 14px 20px;
    border-radius: 10px;
    margin-bottom: 12px;
    color: #991b1b;
    font-weight: 600;
    font-size: 14.5px;
}

/* Confidence pill */
.conf-pill {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 12px;
}

/* Sources */
.source-item {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-size: 13.5px;
    color: #334155;
    line-height: 1.5;
}

/* Hide Streamlit chrome */
#MainMenu {visibility: visible;}
footer {visibility: visible;}
header {visibility: visible;}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Cache expensive resources
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Booting CloudDesk engine...")
def load_engine():
    cfg = load_config()
    vstore = get_vector_store(cfg)
    client = InferenceClient(api_key=cfg["hf_token"]) if cfg.get("hf_token") else None
    return cfg, vstore, client

CFG, VSTORE, CLIENT = load_engine()

# -------------------------------------------------------------
# Session state
# -------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources" not in st.session_state:
    st.session_state.sources = ""
if "status" not in st.session_state:
    st.session_state.status = ""

# -------------------------------------------------------------
# Hero header
# -------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>☁️ CloudDesk AI Support Engineer</h1>
    <p>Ask any CloudDesk support question — receive instant, grounded answers with verified source citations in seconds.</p>
    <div class="badges">
        <span class="badge">⚡ RAG-powered</span>
        <span class="badge">📎 Citations included</span>
        <span class="badge">🛡️ Auto-escalation when unsure</span>
        <span class="badge">🎯 Confidence scored</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Layout: chat (left, wide) + sidebar content (right, narrow)
# -------------------------------------------------------------
chat_col, info_col = st.columns([2.2, 1], gap="large")

# -------- Left column: chat --------
with chat_col:
    st.markdown("### 💬 Conversation")

    chat_container = st.container()

    with chat_container:
        if not st.session_state.messages:
            st.markdown(
                "<div style='color:#94a3b8;font-size:14px;padding:24px 0;text-align:center;'>"
                "Start by asking a question below, or pick an example from the right panel 👉"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="user-msg">{msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="assistant-msg">{msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )

    # -------- Input row --------
    with st.form("query_form", clear_on_submit=True):
        c1, c2 = st.columns([5, 1])
        with c1:
            question = st.text_input(
                "Ask a question",
                placeholder="💬 Ask a CloudDesk support question…",
                label_visibility="collapsed",
            )
        with c2:
            submitted = st.form_submit_button("Ask ➤", use_container_width=True)

    # -------- Handle submission --------
    if submitted and question.strip():
        # 1. Append user message
        st.session_state.messages.append({"role": "user", "content": question})

        # 2. Run pipeline with spinner
        with st.spinner("🔎 Searching knowledge base and generating answer…"):
            t0 = time.time()
            try:
                res = run_rag_pipeline(CFG, VSTORE, CLIENT, question.strip())
                elapsed = f"{time.time() - t0:.2f}s"
            except Exception as e:
                from metrics_logger import log_error
                log_error(question, e, context={"stage": "pipeline", "store": VSTORE[0]})
                st.error("⚠️ Something went wrong. This query has been logged for review.")
                st.stop()
            log_query(
            question=question,
            confidence=res["confidence"],
            escalated=res["requires_escalation"],
            sources_used=len(res["docs"]),
            vector_store=VSTORE[0],
            latency_s=float(elapsed.replace("s", "")),
        )    

        # 3. Build assistant message
        conf_pct = int(res["confidence"] * 100)
        escalated = res["requires_escalation"]

        if escalated:
            banner = '<div class="banner-esc">🚨 This question needs a human support agent — a Tier-2 engineer will follow up.</div>'
            conf_color = "#ef4444"
            conf_label = "🔴 Low confidence"
        elif conf_pct >= 75:
            banner = '<div class="banner-ok">✅ Answer found in CloudDesk documentation.</div>'
            conf_color = "#10b981"
            conf_label = "🟢 High confidence"
        else:
            banner = '<div class="banner-ok">✅ Answer found in CloudDesk documentation.</div>'
            conf_color = "#f59e0b"
            conf_label = "🟡 Good confidence"

        conf_html = (
            f'<div style="margin:8px 0 14px;">'
            f'<span class="conf-pill" style="background:{conf_color}1a;color:{conf_color};">'
            f'{conf_label} · {conf_pct}%</span></div>'
        )

        answer_body = f"{banner}{conf_html}\n\n{res['answer']}"

        # 4. Store assistant response + side state
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_body,
        })
        st.session_state.sources = res.get("citations", "") or "_No sources retrieved._"
        st.session_state.status = (
            f"⏱️ {elapsed} · 📚 {len(res['docs'])} chunks used · "
            f"🗄️ {VSTORE[0].upper()} vector store"
        )

        st.rerun()

# -------- Right column: sources + info --------
with info_col:
    st.markdown("### 📎 Sources")
    if st.session_state.sources:
        st.markdown(
            f'<div class="source-item">{st.session_state.sources}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='color:#94a3b8;font-size:13.5px;'>"
            "_Sources for your latest question will appear here._</div>",
            unsafe_allow_html=True,
        )

    if st.session_state.status:
        st.markdown(
            f"<div style='color:#64748b;font-size:13px;margin-top:10px;'>{st.session_state.status}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown("#### 💡 Try these examples")
    examples = [
        "My SAML login stopped working after adding a new domain",
        "How do I integrate Slack with CloudDesk?",
        "Users can't reset their password via email",
        "Webhook deliveries are failing with 401 errors",
        "How do I enable SSO for a new workspace?",
        "API rate limit exceeded error after upgrading plan",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": ex})
            with st.spinner("🔎 Searching…"):
                t0 = time.time()
                try:
                    res = run_rag_pipeline(CFG, VSTORE, CLIENT, ex)
                    elapsed = f"{time.time() - t0:.2f}s"
                except Exception as e:
                    from metrics_logger import log_error
                    log_error(ex, e, context={"stage": "example", "store": VSTORE[0]})
                    st.error("⚠️ Something went wrong. This query has been logged.")
                    st.stop()

            log_query(
                question=ex,
                confidence=res["confidence"],
                escalated=res["requires_escalation"],
                sources_used=len(res["docs"]),
                vector_store=VSTORE[0],
                latency_s=float(elapsed.replace("s", "")),
            )                

            conf_pct = int(res["confidence"] * 100)
            escalated = res["requires_escalation"]

            if escalated:
                banner = '<div class="banner-esc">🚨 This question needs a human support agent.</div>'
                conf_color = "#ef4444"
                conf_label = "🔴 Low confidence"
            elif conf_pct >= 75:
                banner = '<div class="banner-ok">✅ Answer found in CloudDesk documentation.</div>'
                conf_color = "#10b981"
                conf_label = "🟢 High confidence"
            else:
                banner = '<div class="banner-ok">✅ Answer found in CloudDesk documentation.</div>'
                conf_color = "#f59e0b"
                conf_label = "🟡 Good confidence"

            conf_html = (
                f'<div style="margin:8px 0 14px;">'
                f'<span class="conf-pill" style="background:{conf_color}1a;color:{conf_color};">'
                f'{conf_label} · {conf_pct}%</span></div>'
            )
            answer_body = f"{banner}{conf_html}\n\n{res['answer']}"
            st.session_state.messages.append({"role": "assistant", "content": answer_body})
            st.session_state.sources = res.get("citations", "") or "_No sources retrieved._"
            st.session_state.status = (
                f"⏱️ {elapsed} · 📚 {len(res['docs'])} chunks used · "
                f"🗄️ {VSTORE[0].upper()} vector store"
            )
            st.rerun()

    st.markdown("---")

    with st.expander("ℹ️ How this works"):
        st.markdown("""
1. 🔍 **Search** — your question is compared against all CloudDesk docs
2. 📚 **Retrieve** — the 3 most relevant chunks are pulled
3. 🤖 **Generate** — the AI writes an answer using only those chunks
4. 📊 **Score** — confidence is measured (0–100%)
5. ⚠️ **Escalate** — below 60% → human takes over
        """)

    with st.expander("🛟 Not sure what to ask?"):
        st.markdown("""
- *"How do I …?"* — setup questions
- *"I'm getting error X when …"* — troubleshooting
- *"Why did … stop working?"* — debugging
        """)

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.sources = ""
        st.session_state.status = ""
        st.rerun()