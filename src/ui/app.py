"""
LocalMind – polished Streamlit UI
Modern chat interface for the Agentic RAG assistant.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

from src.config import settings
from src.retrieval.embeddings import Embedder
from src.retrieval.vectorstore import VectorStore
from src.retrieval.hybrid import HybridRetriever
from src.ingestion.chunker import Chunk
from src.agent.agent import LocalMindAgent
from src.llm import LLM


# ---------------------------------------------------------------------------
# Page config & custom CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LocalMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    /* Hide default Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main background */
    .stApp {
        background: linear-gradient(160deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0b1220;
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #94a3b8;
    }

    /* Cards */
    .lm-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.8rem;
    }
    .lm-card-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        margin-bottom: 0.35rem;
    }

    /* Tool badge */
    .tool-badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .tool-local   { background: #14532d; color: #86efac; }
    .tool-web     { background: #1e3a8a; color: #93c5fd; }
    .tool-both    { background: #4c1d95; color: #c4b5fd; }
    .tool-none    { background: #374151; color: #d1d5db; }

    /* Source chip */
    .source-chip {
        display: inline-block;
        background: #0f172a;
        border: 1px solid #334155;
        color: #94a3b8;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.78rem;
        margin: 0.15rem 0.25rem 0.15rem 0;
    }

    /* Chat bubbles */
    .user-bubble {
        background: #1e40af;
        color: #eff6ff;
        padding: 0.9rem 1.2rem;
        border-radius: 16px 16px 4px 16px;
        margin: 0.6rem 0 0.6rem auto;
        max-width: 85%;
        width: fit-content;
        margin-left: auto;
    }
    .assistant-bubble {
        background: #1e293b;
        border: 1px solid #334155;
        color: #e2e8f0;
        padding: 0.9rem 1.2rem;
        border-radius: 16px 16px 16px 4px;
        margin: 0.6rem auto 0.6rem 0;
        max-width: 90%;
    }

    /* Header */
    .lm-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .lm-header h1 {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .lm-header p {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 0;
    }

    /* Status pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: #0f172a;
        border: 1px solid #334155;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        font-size: 0.8rem;
        color: #94a3b8;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #22c55e;
        box-shadow: 0 0 6px #22c55e;
    }
    .status-dot.warn { background: #eab308; box-shadow: 0 0 6px #eab308; }
    .status-dot.err  { background: #ef4444; box-shadow: 0 0 6px #ef4444; }

    /* Input area */
    .stTextArea textarea {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
    }

    /* Metric cards in sidebar */
    [data-testid="stMetric"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 0.6rem 0.9rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Load components (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_components():
    embedder = Embedder()
    vs = VectorStore(embedder=embedder)

    chunks = []
    chunks_path = settings.processed_dir / "chunks.jsonl"
    if chunks_path.exists():
        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                chunks.append(
                    Chunk(
                        id=d["id"],
                        content=d["content"],
                        source=d["source"],
                        metadata=d["metadata"],
                    )
                )

    retriever = HybridRetriever(vectorstore=vs, chunks=chunks)
    llm = LLM()
    agent = LocalMindAgent(retriever=retriever, llm_generate=llm.generate)
    return agent, vs, retriever, llm


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, meta}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 LocalMind")
    st.caption("Privacy-first Agentic RAG")

    st.markdown("---")

    # Status
    try:
        agent, vs, retriever, llm = load_components()
        chunk_count = vs.count()
        model_ready = getattr(llm, "is_ready", False)

        st.markdown(
            f"""
            <div class="status-pill">
                <span class="status-dot {'warn' if not model_ready else ''}"></span>
                {'Model ready' if model_ready else 'Mock mode (set GGUF path)'}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")
        c1, c2 = st.columns(2)
        c1.metric("Chunks", chunk_count)
        c2.metric("Messages", len(st.session_state.messages))

    except Exception as e:
        st.error(f"Load error: {e}")
        st.info("Run `python scripts/ingest.py` after adding files to `data/raw/`")
        st.stop()

    st.markdown("---")
    st.markdown("#### Retrieval")
    top_k = st.slider("Top-K results", 1, 12, settings.top_k, help="How many chunks to retrieve")
    alpha = st.slider(
        "Hybrid α (dense weight)",
        0.0,
        1.0,
        settings.hybrid_alpha,
        0.05,
        help="1.0 = pure semantic, 0.0 = pure keyword (BM25)",
    )

    st.markdown("---")
    st.markdown("#### Quick tips")
    st.markdown(
        """
        - Put documents in `data/raw/`
        - Run `python scripts/ingest.py`
        - Ask questions that need local docs, web, or both
        """
    )

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="lm-header">
        <h1>LocalMind</h1>
        <p>Agentic RAG • Hybrid retrieval • Source citations • Fully local</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
TOOL_STYLES = {
    "local_rag": ("tool-local", "📄 Local documents"),
    "web_search": ("tool-web", "🌐 Web search"),
    "both": ("tool-both", "🔀 Local + Web"),
    "none": ("tool-none", "💬 General knowledge"),
}


def render_message(msg: dict):
    role = msg["role"]
    content = msg["content"]
    meta = msg.get("meta") or {}

    if role == "user":
        st.markdown(f'<div class="user-bubble">{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-bubble">{content}</div>', unsafe_allow_html=True)

        tool = meta.get("tool_used")
        if tool:
            css, label = TOOL_STYLES.get(tool, ("tool-none", tool))
            st.markdown(
                f'<span class="tool-badge {css}">{label}</span>',
                unsafe_allow_html=True,
            )

        sources = meta.get("sources") or []
        if sources:
            chips = " ".join(f'<span class="source-chip">{s}</span>' for s in sources[:8])
            st.markdown(
                f'<div style="margin-top:0.5rem"><span class="lm-card-title">Sources</span><br>{chips}</div>',
                unsafe_allow_html=True,
            )

        # Expandable context
        ctx = meta.get("context")
        if ctx:
            with st.expander("View retrieved context", expanded=False):
                st.code(ctx[:5000] + ("…" if len(ctx) > 5000 else ""), language=None)


for msg in st.session_state.messages:
    render_message(msg)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
st.markdown("---")

col_input, col_btn = st.columns([6, 1])
with col_input:
    question = st.text_area(
        "Message",
        placeholder="Ask anything about your documents or the web…",
        height=90,
        label_visibility="collapsed",
        key="user_input",
    )
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    send = st.button("Send", type="primary", use_container_width=True)

# Example prompts
ex1, ex2, ex3 = st.columns(3)
with ex1:
    if st.button("📄 What’s in my documents?", use_container_width=True):
        st.session_state._force_q = "Summarize the main topics covered in my documents."
with ex2:
    if st.button("🌐 Latest AI news", use_container_width=True):
        st.session_state._force_q = "What are the latest developments in generative AI this week?"
with ex3:
    if st.button("🔀 Compare local vs web", use_container_width=True):
        st.session_state._force_q = (
            "Based on my documents and current public information, "
            "what should I know about RAG systems?"
        )

# Handle forced example or normal send
force_q = st.session_state.pop("_force_q", None)
query = force_q or (question.strip() if send and question.strip() else None)

if query:
    # Add user message
    st.session_state.messages.append(
        {"role": "user", "content": query, "meta": None}
    )

    with st.spinner("Thinking…"):
        settings.top_k = top_k
        settings.hybrid_alpha = alpha
        try:
            result = agent.run(query)
            answer = result.answer
            meta = {
                "tool_used": result.tool_used,
                "sources": result.sources,
                "context": result.context,
            }
        except Exception as e:
            answer = f"Sorry, something went wrong: `{e}`"
            meta = {}

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "meta": meta}
    )
    st.rerun()


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; color:#475569; font-size:0.8rem; margin-top:2rem;">
        LocalMind · Privacy-first Agentic RAG · Built for demonstration of production GenAI skills
    </div>
    """,
    unsafe_allow_html=True,
)
