"""
CloudDesko AI Support Engineer
"""

import os
import time
import gradio as gr
from huggingface_hub import InferenceClient
from retrieval_pipeline import load_config, get_vector_store, run_rag_pipeline

# -------------------------------------------------------------
# 1. Startup
# -------------------------------------------------------------
print("[*] Booting CloudDesko UI...")
CFG = load_config()
VSTORE = get_vector_store(CFG)
CLIENT = InferenceClient(api_key=CFG["hf_token"]) if CFG.get("hf_token") else None
print(f"[*] Vector store: {VSTORE[0].upper()}")
print(f"[*] LLM: {'connected' if CLIENT else 'offline'}")

# -------------------------------------------------------------
# 2. Helpers
# -------------------------------------------------------------
def confidence_badge(pct: int) -> str:
    """Returns a Markdown-safe confidence indicator."""
    filled = int(pct / 5)
    empty = 20 - filled
    bar = "█" * filled + "░" * empty

    if pct >= 75:
        return f"🟢 **High confidence — {pct}%**\n\n`{bar}`"
    elif pct >= 60:
        return f"🟡 **Good confidence — {pct}%**\n\n`{bar}`"
    else:
        return f"🔴 **Low confidence — {pct}%**\n\n`{bar}`"

def clean_answer_markdown(answer: str) -> str:
    answer = answer.strip()
    for prefix in ["### ✅ AI-Generated Resolution", "### 🚨 Escalated to Human Support"]:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].lstrip()
    return answer

# -------------------------------------------------------------
# 3. Core handler — Markdown-only output
# -------------------------------------------------------------
def handle_query(question: str, history: list):
    if not question or not question.strip():
        return history, "", "", ""

    t0 = time.time()
    res = run_rag_pipeline(CFG, VSTORE, CLIENT, question.strip())
    elapsed = f"{time.time() - t0:.2f}s"

    answer_text = clean_answer_markdown(res["answer"])
    conf_pct = int(res["confidence"] * 100)
    escalated = res["requires_escalation"]

    # ---- Status banner (Markdown blockquote, colored via emoji) ----
    if escalated:
        banner = (
            "> 🚨 **This question needs a human support agent.**\n"
            "> Our AI wasn't confident enough to answer reliably.\n"
            "> A Tier-2 engineer will follow up.\n"
        )
    else:
        banner = "> ✅ **Answer found in CloudDesk documentation.**\n"

    # ---- Confidence meter (Markdown) ----
    conf_md = confidence_badge(conf_pct)

    # ---- Compose the assistant message ----
    body = f"{banner}\n---\n\n{conf_md}\n\n---\n\n{answer_text}"

    # ---- Sources for the sidebar ----
    cites_raw = res.get("citations", "") or ""
    if cites_raw.strip():
        sources_md = f"### 📎 Where this answer came from\n\n{cites_raw}"
    else:
        sources_md = "_No sources were retrieved for this question._"

    # ---- Status line ----
    status_md = (
        f"⏱️ **{elapsed}** &nbsp;·&nbsp; "
        f"📚 **{len(res['docs'])}** chunks used &nbsp;·&nbsp; "
        f"🗄️ **{VSTORE[0].upper()}** vector store"
    )

    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": body},
    ]
    return history, "", sources_md, status_md

def clear_all():
    return [], "", "", ""

# -------------------------------------------------------------
# 4. Examples
# -------------------------------------------------------------
EXAMPLES = [
    "My SAML login stopped working after adding a new domain",
    "How do I integrate Slack with CloudDesk?",
    "Users can't reset their password via email",
    "Webhook deliveries are failing with 401 errors",
    "How do I enable SSO for a new workspace?",
    "API rate limit exceeded error after upgrading plan",
]

# -------------------------------------------------------------
# 5. Custom CSS — polished, professional
# -------------------------------------------------------------
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

body, .gradio-container {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    background: #f6f8fc !important;
}
.gradio-container {
    max-width: 1280px !important;
    margin: auto !important;
    padding: 16px 24px 32px !important;
}

/* Hero */
#hero {
    background: linear-gradient(135deg, #0b1220 0%, #1e3a8a 45%, #3b82f6 100%);
    padding: 44px 48px;
    border-radius: 20px;
    color: #ffffff !important;
    margin-bottom: 28px;
    box-shadow: 0 20px 50px rgba(15,23,42,0.25);
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
}
#hero::before {
    content: "";
    position: absolute;
    top: -60%; right: -15%;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(255,255,255,0.18) 0%, transparent 65%);
    pointer-events: none;
}
#hero::after {
    content: "";
    position: absolute;
    bottom: -70%; left: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(96,165,250,0.30) 0%, transparent 70%);
    pointer-events: none;
}
#hero h1 {
    margin: 0; font-size: 36px; font-weight: 800;
    letter-spacing: -0.8px; color: #ffffff !important;
    position: relative; z-index: 1; line-height: 1.15;
}
#hero p {
    margin: 14px 0 0; font-size: 16px;
    color: rgba(255,255,255,0.92) !important;
    line-height: 1.6; font-weight: 400;
    max-width: 720px; position: relative; z-index: 1;
}
#hero .badges {
    margin-top: 22px; display: flex; flex-wrap: wrap;
    gap: 8px; position: relative; z-index: 1;
}
#hero .badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.22);
    padding: 6px 14px; border-radius: 24px;
    font-size: 12.5px; color: #ffffff !important;
    font-weight: 500; letter-spacing: 0.2px;
    backdrop-filter: blur(8px);
}

/* Chatbot card */
.chatbot, .gr-chatbot {
    border-radius: 16px !important;
    border: 1px solid #e5e7eb !important;
    background: #ffffff !important;
    box-shadow: 0 4px 16px rgba(15,23,42,0.05) !important;
    overflow: hidden !important;
}
.message, .message-wrap, .bubble {
    border-radius: 12px !important;
    font-size: 14.5px !important;
    line-height: 1.65 !important;
}

/* Inputs */
textarea, input[type="text"] {
    border-radius: 12px !important;
    border: 1.5px solid #e2e8f0 !important;
    padding: 14px 16px !important;
    font-size: 15px !important;
    background: #ffffff !important;
    transition: all 0.18s ease;
    font-family: 'Inter', sans-serif !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 4px rgba(59,130,246,0.12) !important;
    outline: none !important;
}

/* Buttons */
button.primary, .primary {
    background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 14.5px !important;
    padding: 14px 24px !important;
    box-shadow: 0 6px 16px rgba(37,99,235,0.28) !important;
    transition: all 0.15s ease !important;
    color: #ffffff !important;
}
button.primary:hover {
    transform: translateY(-1.5px);
    box-shadow: 0 10px 22px rgba(37,99,235,0.38) !important;
}
button.secondary {
    border-radius: 12px !important;
    border: 1.5px solid #e2e8f0 !important;
    font-weight: 500 !important;
    padding: 14px 22px !important;
    background: #ffffff !important;
    color: #475569 !important;
    transition: all 0.15s ease !important;
}
button.secondary:hover {
    border-color: #cbd5e1 !important;
    background: #f8fafc !important;
}

/* Accordion */
.gr-accordion, .gr-group {
    border-radius: 14px !important;
    border: 1px solid #e5e7eb !important;
    margin-bottom: 12px !important;
    background: #ffffff !important;
    overflow: hidden !important;
    box-shadow: 0 2px 8px rgba(15,23,42,0.03);
}
.gr-accordion > button, .gr-accordion > .label-wrap {
    font-weight: 600 !important;
    color: #1e293b !important;
    padding: 14px 18px !important;
    font-size: 14.5px !important;
    background: #ffffff !important;
}
.gr-accordion > button:hover {
    background: #f8fafc !important;
}

/* Sidebar heading */
.gr-markdown h3 {
    color: #0f172a !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    margin-top: 6px !important;
    letter-spacing: -0.2px;
}

/* Status bar */
#status-bar {
    font-size: 13px;
    color: #64748b !important;
    padding: 12px 4px 4px;
    border-top: 1px solid #eef2f7;
    margin-top: 14px;
    letter-spacing: 0.1px;
}

/* Examples */
.gr-samples button, .examples button {
    border-radius: 10px !important;
    border: 1px solid #e2e8f0 !important;
    background: #ffffff !important;
    padding: 10px 14px !important;
    font-size: 13.5px !important;
    color: #334155 !important;
    text-align: left !important;
    transition: all 0.15s ease !important;
    margin-bottom: 6px !important;
}
.gr-samples button:hover, .examples button:hover {
    border-color: #3b82f6 !important;
    background: #eff6ff !important;
    color: #1e40af !important;
    transform: translateX(2px);
}

footer { display: none !important; }

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #f1f5f9; border-radius: 4px; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
"""

# -------------------------------------------------------------
# 6. UI
# -------------------------------------------------------------
with gr.Blocks(title="CloudDesk AI Support Engineer") as demo:

    gr.HTML("""
        <div id="hero">
            <h1>☁️ CloudDesk AI Support Engineer</h1>
            <p>Ask any CloudDesk support question — receive instant, grounded answers with verified source citations in seconds.</p>
            <div class="badges">
                <span class="badge">⚡ RAG-powered</span>
                <span class="badge">📎 Citations included</span>
                <span class="badge">🛡️ Auto-escalation when unsure</span>
                <span class="badge">🎯 Confidence scored</span>
            </div>
        </div>
    """)

    with gr.Row(equal_height=False):
        # ===================== LEFT: Chat =====================
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=580,
                show_label=False,
            )

            with gr.Row():
                question_box = gr.Textbox(
                    placeholder="💬  Ask a CloudDesk support question and press Enter…",
                    label="",
                    scale=6,
                    container=False,
                    autofocus=True,
                    lines=1,
                )
                submit_btn = gr.Button("Ask ➤", variant="primary", scale=1)
                clear_btn = gr.Button("Clear", variant="secondary", scale=1)

            status_html = gr.Markdown("", elem_id="status-bar")

        # ===================== RIGHT: Sidebar =====================
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("### 📎 Sources")
                sources_box = gr.Markdown("_No sources yet — ask a question to begin._")

            with gr.Accordion("💡 Try these examples", open=True):
                gr.Examples(examples=EXAMPLES, inputs=question_box, label="")

            with gr.Accordion("ℹ️ How this works", open=False):
                gr.Markdown("""
**What happens when you ask a question:**

1. 🔍 **Search** — Your question is compared against all CloudDesk docs
2. 📚 **Retrieve** — The 3 most relevant chunks are pulled from our knowledge base
3. 🤖 **Generate** — The AI writes an answer using **only** those chunks
4. 📊 **Score** — Confidence is measured (0–100%)
5. ⚠️ **Escalate** — If confidence is below 60%, a human takes over

**Knowledge sources:**

| Source | Type |
|---|---|
| 📘 API Documentation | Markdown |
| 🛠 Engineering Runbooks | Markdown |
| 📚 Help Center Articles | PDF |
| 📄 Release Notes | PDF |
| 🎫 Support Tickets | CSV |
                """)

            with gr.Accordion("🛟 Not sure what to ask?", open=False):
                gr.Markdown("""
Try these formats:

- *"How do I …?"* — for setup questions
- *"I'm getting error X when …"* — for troubleshooting
- *"Why did … stop working?"* — for debugging help
- *"Can you explain how X works?"* — for conceptual questions
                """)

    submit_btn.click(
        handle_query,
        inputs=[question_box, chatbot],
        outputs=[chatbot, question_box, sources_box, status_html],
    )
    question_box.submit(
        handle_query,
        inputs=[question_box, chatbot],
        outputs=[chatbot, question_box, sources_box, status_html],
    )
    clear_btn.click(
        clear_all,
        outputs=[chatbot, question_box, sources_box, status_html],
    )

# -------------------------------------------------------------
# 7. Launch
# -------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        show_error=True,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )