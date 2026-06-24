"""
streamlit_app.py
----------------
Pharma Regulatory Doc QA — Streamlit UI.

Features:
- PDF upload + ingestion trigger
- Natural language query input
- Citation-backed answers with source attribution
- Confidence scoring per source chunk
- Session history with expand/collapse
- Reset session button

Run:
    streamlit run app/streamlit_app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
from pathlib import Path
import shutil
import json

from ingestion import run_ingestion
from agent import PharmaQAAgent
from config import DATA_RAW_DIR, VECTOR_STORE_DIR, FAISS_INDEX_NAME


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Pharma Regulatory QA",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────

def init_state():
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []   # [{role, content, sources, scores}]
    if "index_ready" not in st.session_state:
        st.session_state.index_ready = False
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

init_state()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def save_uploaded_pdfs(uploaded_files) -> list[str]:
    """Save Streamlit uploaded files to data/raw/."""
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    saved = []
    for uf in uploaded_files:
        dest = os.path.join(DATA_RAW_DIR, uf.name)
        with open(dest, "wb") as f:
            f.write(uf.read())
        saved.append(uf.name)
    return saved


def index_exists() -> bool:
    return os.path.exists(os.path.join(VECTOR_STORE_DIR, FAISS_INDEX_NAME))


def confidence_colour(score: float) -> str:
    if score >= 0.85:
        return "🟢"
    elif score >= 0.75:
        return "🟡"
    else:
        return "🔴"


def get_agent() -> PharmaQAAgent:
    if st.session_state.agent is None:
        st.session_state.agent = PharmaQAAgent()
    return st.session_state.agent


# ─────────────────────────────────────────────
# SIDEBAR — DOCUMENT UPLOAD + INGESTION
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("💊 Pharma Doc QA")
    st.markdown("---")

    st.subheader("📂 Upload Documents")
    uploaded = st.file_uploader(
        "Upload regulatory PDFs (FDA, ICH, EMA, clinical docs)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded:
        if st.button("🔄 Index Documents", type="primary"):
            with st.spinner("Saving PDFs..."):
                saved = save_uploaded_pdfs(uploaded)
                st.session_state.uploaded_files = saved

            with st.spinner(f"Chunking + embedding {len(saved)} file(s)... (takes ~30s)"):
                try:
                    run_ingestion(docs_dir=DATA_RAW_DIR, rebuild=True)
                    st.session_state.index_ready = True
                    st.session_state.agent = None  # reset agent to reload index
                    st.success(f"✅ Indexed {len(saved)} file(s)!")
                    for f in saved:
                        st.caption(f"• {f}")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    # Auto-detect existing index
    if not st.session_state.index_ready and index_exists():
        st.session_state.index_ready = True
        st.info("📦 Existing index detected and loaded.")

    st.markdown("---")

    # Indexed files list
    if st.session_state.uploaded_files:
        st.subheader("📄 Indexed Files")
        for f in st.session_state.uploaded_files:
            st.caption(f"• {f}")

    st.markdown("---")

    # Session controls
    st.subheader("⚙️ Session")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        if st.session_state.agent:
            st.session_state.agent.reset_history()
        st.rerun()

    if st.button("♻️ Reset Everything"):
        st.session_state.agent = None
        st.session_state.chat_history = []
        st.session_state.index_ready = False
        st.session_state.uploaded_files = []
        st.rerun()

    st.markdown("---")
    st.caption("Model: GPT-4o | Embeddings: text-embedding-3-small | VectorDB: FAISS")


# ─────────────────────────────────────────────
# MAIN — CHAT INTERFACE
# ─────────────────────────────────────────────

st.title("🔬 Pharmaceutical Regulatory Document QA")
st.markdown(
    "Ask natural language questions about uploaded regulatory documents "
    "(FDA guidelines, ICH Q-series, clinical protocols, compliance docs). "
    "Answers are citation-backed with source attribution."
)

if not st.session_state.index_ready:
    st.warning("⬅️ Upload and index your regulatory PDFs using the sidebar to get started.")
    st.stop()

# ── Chat history display ──────────────────────
for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

        # Show sources for assistant turns
        if turn["role"] == "assistant" and turn.get("sources"):
            with st.expander(f"📎 Sources ({len(turn['sources'])} cited)", expanded=False):
                for s in turn["sources"]:
                    colour = confidence_colour(s["score"])
                    st.markdown(
                        f"{colour} **{s['source']}** — Page {s['page_num']} "
                        f"| Confidence: `{s['score']:.0%}`"
                    )

# ── Query input ───────────────────────────────
query = st.chat_input(
    "Ask a regulatory question... (e.g. 'What are ICH Q1A stability requirements?')"
)

if query:
    if not query.strip():
        st.warning("Enter a question.")
        st.stop()

    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.chat_history.append({
        "role": "user",
        "content": query,
        "sources": [],
    })

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Searching documents + generating answer..."):
            try:
                agent = get_agent()
                response = agent.ask(query)
                answer = response.answer
                sources = response.sources

            except Exception as e:
                answer = f"⚠️ Error: {e}"
                sources = []

        st.markdown(answer)

        # Source attribution panel
        if sources:
            with st.expander(f"📎 Sources ({len(sources)} cited)", expanded=True):
                for s in sources:
                    colour = confidence_colour(s["score"])
                    st.markdown(
                        f"{colour} **{s['source']}** — Page {s['page_num']} "
                        f"| Confidence: `{s['score']:.0%}`"
                    )
        else:
            st.caption("⚠️ No sources retrieved above confidence threshold.")

    # Save assistant turn
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })


# ─────────────────────────────────────────────
# FOOTER — EVAL STATS (if report exists)
# ─────────────────────────────────────────────

report_path = "data/qa_pairs/eval_report.json"
if os.path.exists(report_path):
    st.markdown("---")
    with st.expander("📊 Validation Report", expanded=False):
        with open(report_path) as f:
            report = json.load(f)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Questions Evaluated", report["total"])
        col2.metric("Avg Token F1", f"{report['avg_token_f1']:.1%}")
        col3.metric("Avg ROUGE-L", f"{report['avg_rouge_l']:.1%}")
        col4.metric("Citation Hit Rate", f"{report['citation_hit_rate']:.1%}")
