"""
LocalMind v0.2 – UI-driven Agentic RAG
Chat, document management, ingest, and model settings from the browser.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

from src.config import settings
from src.retrieval.embeddings import Embedder
from src.retrieval.vectorstore import VectorStore
from src.retrieval.hybrid import HybridRetriever
from src.agent.agent import LocalMindAgent
from src.llm import LLM
from src.services.ingest_service import (
    save_uploads,
    list_raw_documents,
    delete_raw_document,
    run_full_ingest,
    load_chunks_from_disk,
)
from src.services.config_service import (
    get_model_path, set_model_path, get_gpu_layers, set_gpu_layers,
    get_preferred_backend, set_preferred_backend,
)
from src.inference.hardware import detect_hardware, format_hardware_summary
from src.inference.backends import probe_backends, BACKEND_CHOICES
from src.services.acceleration_service import (
    build_plans,
    recommended_plan,
    install_backend,
    format_pip_command,
    current_llama_info,
)



# ---------------------------------------------------------------------------
# Page setup
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
    #MainMenu, footer, header {visibility: hidden;}
    .stApp {
        background: linear-gradient(160deg, #0f172a 0%, #1e293b 45%, #0f172a 100%);
    }
    section[data-testid="stSidebar"] {
        background: #0b1220;
        border-right: 1px solid #1e293b;
    }
    .tool-badge {
        display: inline-block; padding: 0.25rem 0.7rem; border-radius: 999px;
        font-size: 0.8rem; font-weight: 600;
    }
    .tool-local { background: #14532d; color: #86efac; }
    .tool-web   { background: #1e3a8a; color: #93c5fd; }
    .tool-both  { background: #4c1d95; color: #c4b5fd; }
    .tool-none  { background: #374151; color: #d1d5db; }
    .source-chip {
        display: inline-block; background: #0f172a; border: 1px solid #334155;
        color: #94a3b8; padding: 0.2rem 0.55rem; border-radius: 6px;
        font-size: 0.75rem; margin: 0.15rem 0.2rem 0.15rem 0;
    }
    .user-bubble {
        background: #1e40af; color: #eff6ff; padding: 0.85rem 1.1rem;
        border-radius: 16px 16px 4px 16px; margin: 0.5rem 0 0.5rem auto;
        max-width: 85%; width: fit-content; margin-left: auto;
    }
    .assistant-bubble {
        background: #1e293b; border: 1px solid #334155; color: #e2e8f0;
        padding: 0.85rem 1.1rem; border-radius: 16px 16px 16px 4px;
        margin: 0.5rem auto 0.5rem 0; max-width: 92%;
    }
    .lm-header { text-align: center; padding: 0.8rem 0 0.4rem 0; }
    .lm-header h1 {
        font-size: 1.9rem; font-weight: 700; margin-bottom: 0.15rem;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #34d399);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .lm-header p { color: #94a3b8; margin-top: 0; font-size: 0.95rem; }
    .status-pill {
        display: inline-flex; align-items: center; gap: 0.4rem;
        background: #0f172a; border: 1px solid #334155;
        padding: 0.3rem 0.75rem; border-radius: 999px;
        font-size: 0.78rem; color: #94a3b8;
    }
    .status-dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: #22c55e; box-shadow: 0 0 6px #22c55e;
    }
    .status-dot.warn { background: #eab308; box-shadow: 0 0 6px #eab308; }
    .status-dot.err  { background: #ef4444; box-shadow: 0 0 6px #ef4444; }
    [data-testid="stMetric"] {
        background: #1e293b; border: 1px solid #334155;
        border-radius: 10px; padding: 0.5rem 0.8rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session defaults
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "reload_token" not in st.session_state:
    st.session_state.reload_token = 0


def bump_reload():
    """Force @st.cache_resource to rebuild on next load_components call."""
    st.session_state.reload_token += 1
    load_components.clear()


# ---------------------------------------------------------------------------
# Cached core stack
# ---------------------------------------------------------------------------
@st.cache_resource
def load_components(_token: int = 0):
    embedder = Embedder()
    vs = VectorStore(embedder=embedder)
    chunks = load_chunks_from_disk()
    retriever = HybridRetriever(vectorstore=vs, chunks=chunks)
    model_path = get_model_path() or None
    llm = LLM(model_path=model_path if model_path else None)
    agent = LocalMindAgent(retriever=retriever, llm_generate=llm.generate)
    return agent, vs, retriever, llm


TOOL_STYLES = {
    "local_rag": ("tool-local", "Local documents"),
    "web_search": ("tool-web", "Web search"),
    "both": ("tool-both", "Local + Web"),
    "none": ("tool-none", "General knowledge"),
}


def render_message(msg: dict):
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        return
    st.markdown(f'<div class="assistant-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
    meta = msg.get("meta") or {}
    tool = meta.get("tool_used")
    if tool:
        css, label = TOOL_STYLES.get(tool, ("tool-none", tool))
        if tool == "refused":
            css, label = "tool-none", "Refused (weak evidence)"
        st.markdown(f'<span class="tool-badge {css}">{label}</span>', unsafe_allow_html=True)
    # Route validation details
    bits = []
    if meta.get("proposed_tool"):
        bits.append(f"proposed: `{meta['proposed_tool']}`")
    if meta.get("max_local_score") is not None:
        bits.append(f"max local score: `{meta['max_local_score']:.3f}`")
    if meta.get("route_overridden"):
        bits.append(f"overridden: {meta.get('override_reason') or 'yes'}")
    if bits:
        st.caption(" · ".join(bits))
    sources = meta.get("sources") or []
    if sources:
        chips = " ".join(f'<span class="source-chip">{s}</span>' for s in sources[:8])
        st.markdown(f"<div style='margin-top:0.4rem'>{chips}</div>", unsafe_allow_html=True)
    ctx = meta.get("context")
    if ctx:
        with st.expander("Retrieved context", expanded=False):
            st.code(ctx[:5000] + ("…" if len(ctx) > 5000 else ""), language=None)


# ---------------------------------------------------------------------------
# Sidebar – status + quick controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### LocalMind")
    st.caption("v0.2 · Agentic RAG")

    try:
        agent, vs, retriever, llm = load_components(st.session_state.reload_token)
        chunk_count = vs.count()
        model_ready = getattr(llm, "is_ready", False)
        model_path_display = get_model_path() or "(not set)"
    except Exception as e:
        st.error(f"Load error: {e}")
        agent = vs = retriever = llm = None
        chunk_count = 0
        model_ready = False
        model_path_display = ""

    dot = "" if model_ready else "warn"
    st.markdown(
        f'<div class="status-pill"><span class="status-dot {dot}"></span>'
        f'{"Model ready" if model_ready else "Mock mode"}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")
    c1, c2 = st.columns(2)
    c1.metric("Chunks", chunk_count)
    c2.metric("Docs", len(list_raw_documents()))

    st.markdown("---")
    st.markdown("#### Retrieval")
    top_k = st.slider("Top-K", 1, 12, int(settings.top_k))
    alpha = st.slider("Hybrid α (dense)", 0.0, 1.0, float(settings.hybrid_alpha), 0.05)
    min_score = st.slider(
        "Min retrieval score",
        0.0,
        1.0,
        float(settings.retrieval_min_score),
        0.05,
        help="Local docs below this score are rejected (fallback to web or refuse)",
    )
    weak_action = st.selectbox(
        "If local score too low",
        options=["web", "refuse"],
        index=0 if settings.weak_local_action != "refuse" else 1,
        help="web = try DuckDuckGo; refuse = fixed message, no generation from weak context",
    )

    st.markdown("---")
    if st.button("Reload engine", use_container_width=True, help="Reload model + index after changes"):
        bump_reload()
        st.rerun()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="lm-header">
        <h1>LocalMind</h1>
        <p>Upload docs · Build index · Chat — all from the UI</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_chat, tab_docs, tab_settings = st.tabs(["Chat", "Documents", "Settings"])


# ========================= CHAT =========================
with tab_chat:
    if agent is None:
        st.warning("Engine not loaded. Check Settings and click **Reload engine**.")
    else:
        if chunk_count == 0:
            st.info("No indexed documents yet. Go to **Documents**, upload files, then click **Build index**.")

        for msg in st.session_state.messages:
            render_message(msg)

        st.markdown("---")
        q = st.chat_input("Ask about your documents or the web…")
        if q and q.strip():
            st.session_state.messages.append({"role": "user", "content": q.strip(), "meta": None})
            settings.top_k = top_k
            settings.hybrid_alpha = alpha
            settings.retrieval_min_score = min_score
            settings.weak_local_action = weak_action
            with st.spinner("Thinking…"):
                try:
                    result = agent.run(q.strip())
                    meta = {
                        "tool_used": result.tool_used,
                        "sources": result.sources,
                        "context": result.context,
                        "proposed_tool": getattr(result, "proposed_tool", ""),
                        "max_local_score": getattr(result, "max_local_score", None),
                        "route_overridden": getattr(result, "route_overridden", False),
                        "override_reason": getattr(result, "override_reason", ""),
                    }
                    answer = result.answer
                except Exception as e:
                    answer = f"Error: `{e}`"
                    meta = {}
            st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})
            st.rerun()


# ========================= DOCUMENTS =========================
with tab_docs:
    st.subheader("Your documents")
    st.caption(f"Folder: `{settings.raw_dir}`")

    uploaded = st.file_uploader(
        "Upload PDF, DOCX, TXT, or Markdown",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded:
        if st.button("Save uploads", type="primary"):
            names = save_uploads(uploaded)
            st.success(f"Saved: {', '.join(names)}")
            st.rerun()

    docs = list_raw_documents()
    if not docs:
        st.write("No documents yet. Upload files above.")
    else:
        for d in docs:
            col_a, col_b, col_c = st.columns([4, 1, 1])
            col_a.write(f"**{d['name']}**")
            col_b.write(f"{d['size_kb']} KB")
            if col_c.button("Delete", key=f"del_{d['name']}"):
                delete_raw_document(d["name"])
                st.rerun()

    st.markdown("---")
    st.subheader("Index")
    st.write("Build or rebuild the vector index from all files in the documents folder.")

    if st.button("Build index", type="primary", use_container_width=True):
        progress = st.empty()
        try:
            def on_progress(msg: str):
                progress.info(msg)

            stats = run_full_ingest(progress=on_progress)
            if stats["ok"]:
                progress.success(stats["message"])
                bump_reload()
                st.balloons()
            else:
                progress.warning(stats["message"])
        except Exception as e:
            progress.error(f"Ingest failed: {e}")
        st.rerun()

    st.caption(f"Indexed chunks: **{chunk_count}**")


# ========================= SETTINGS =========================
with tab_settings:
    st.subheader("Model")
    st.write("Path to a local GGUF file. Use forward slashes on Windows.")

    current = get_model_path()
    new_path = st.text_input(
        "GGUF model path",
        value=current,
        placeholder="C:/Users/.../models/model.gguf",
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save model path", type="primary", use_container_width=True):
            if new_path.strip():
                p = Path(new_path.strip().replace("\\", "/"))
                if not p.exists():
                    st.error(f"File not found: {p}")
                else:
                    set_model_path(str(p))
                    bump_reload()
                    st.success("Model path saved. Engine will reload.")
                    st.rerun()
            else:
                st.warning("Enter a path first.")
    with c2:
        if st.button("Clear model path (mock mode)", use_container_width=True):
            set_model_path("")
            bump_reload()
            st.rerun()

    st.markdown("---")
    st.subheader("Hardware acceleration")
    st.caption(
        "Backend is chosen by your **llama-cpp-python** install + this preference. "
        "CUDA / Metal / ROCm require a matching GPU build of the library."
    )

    try:
        hw = detect_hardware()
        status = probe_backends(hw)
    except Exception as e:
        hw = None
        status = None
        st.warning(f"Hardware detection failed: {e}")

    if status is not None:
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("CUDA HW", "Yes" if "cuda" in status.available else "No")
        b2.metric("Metal HW", "Yes" if "metal" in status.available else "No")
        b3.metric("ROCm HW", "Yes" if "rocm" in status.available else "No")
        b4.metric(
            "llama.cpp GPU",
            "Yes" if status.llama_gpu_offload else "No",
        )

        st.write(
            f"**Recommended:** `{status.recommended}` · "
            f"**Available:** {', '.join(status.available)}"
        )

        if status.notes:
            for n in status.notes:
                st.info(n)

        if hw is not None:
            with st.expander("Hardware details", expanded=False):
                st.code(format_hardware_summary(hw), language=None)

        with st.expander("Install matching llama-cpp-python", expanded=True):
            st.caption(
                "Detect hardware, then install a llama-cpp-python build that supports it. "
                "After install: Preferred backend = Auto, Save, Reload engine."
            )
            li = current_llama_info()
            if li.get("installed"):
                st.write(
                    f"Current package: **v{li.get('version')}** · "
                    f"GPU offload: **{'supported' if li.get('gpu_offload_supported') else 'not detected'}**"
                )
            else:
                st.warning("llama-cpp-python not importable.")

            plans = build_plans(hw)
            available = [p for p in plans if p.available and p.pip_args]
            if available:
                rec = recommended_plan(plans)
                labels = [p.label for p in available]
                try:
                    di = next(i for i, p in enumerate(available) if p.backend == rec.backend or p.backend.startswith(rec.backend))
                except StopIteration:
                    di = 0
                sel = st.selectbox("Install backend", labels, index=di, key="accel_install_sel")
                plan = next(p for p in available if p.label == sel)
                st.write(plan.reason)
                if plan.notes:
                    st.caption(plan.notes)
                st.code(format_pip_command(plan), language="bash")
                if st.button(f"Install {plan.label}", type="primary", key="accel_install_btn"):
                    with st.spinner("Installing… can take several minutes"):
                        ok, log = install_backend(plan)
                    if ok:
                        st.success("Installed. Set Preferred backend to Auto and Reload engine.")
                        set_gpu_layers(-1)
                        try:
                            set_preferred_backend("auto")
                        except Exception:
                            pass
                        bump_reload()
                    else:
                        st.error("Install failed — see log or run the command in a terminal.")
                    with st.expander("Install log", expanded=not ok):
                        st.code((log or "")[-6000:], language=None)
            else:
                st.write("No automated install plans for this machine.")

    # Preference select
    current_backend = get_preferred_backend()
    labels = {
        "auto": "Auto — pick best available (CUDA / Metal / ROCm / CPU)",
        "cuda": "CUDA — NVIDIA",
        "metal": "Metal — Apple Silicon",
        "rocm": "ROCm — AMD",
        "cpu": "CPU only",
    }
    # Order options: auto first, then available GPU, always cpu
    opts = ["auto"]
    if status:
        for b in ("cuda", "metal", "rocm"):
            if b in status.available and b not in opts:
                opts.append(b)
    for b in ("cuda", "metal", "rocm", "cpu"):
        if b not in opts:
            opts.append(b)

    display_opts = [labels.get(o, o) for o in opts]
    try:
        idx = opts.index(current_backend) if current_backend in opts else 0
    except ValueError:
        idx = 0

    choice_label = st.selectbox(
        "Preferred backend",
        options=display_opts,
        index=idx,
        help="If the matching llama-cpp build is missing, LocalMind falls back to CPU and shows a warning.",
    )
    # Map label back to key
    label_to_key = {v: k for k, v in labels.items()}
    preferred = label_to_key.get(choice_label, "auto")

    current_layers = get_gpu_layers()
    custom_layers = st.number_input(
        "GPU layers (−1 = auto, 0 = CPU, 1+ = fixed)",
        min_value=-1,
        max_value=999,
        value=int(current_layers),
        help="Used when a GPU backend is selected. Auto uses hardware recommendations.",
    )

    if st.button("Save acceleration settings", type="primary"):
        set_preferred_backend(preferred)
        if preferred == "cpu":
            set_gpu_layers(0)
        else:
            set_gpu_layers(int(custom_layers))
        bump_reload()
        st.success("Acceleration settings saved. Engine reloading…")
        st.rerun()

    if llm is not None and hasattr(llm, "acceleration_info"):
        info = llm.acceleration_info()
        st.markdown("**Currently loaded**")
        st.code(
            f"Preferred      : {info.get('preferred_backend', 'n/a')}\n"
            f"Backend in use : {info.get('backend_in_use', 'n/a')}\n"
            f"GPU layers     : {info.get('gpu_layers_in_use', 'n/a')}\n"
            f"Context size   : {info.get('context_size_in_use', 'n/a')}\n"
            f"llama GPU OK   : {info.get('llama_gpu_offload', 'n/a')}\n"
            f"Model ready    : {info.get('model_ready', model_ready)}",
            language=None,
        )
        for w in info.get("warnings") or []:
            st.warning(w)
        if (
            info.get("backend_in_use") == "cpu"
            and info.get("preferred_backend") not in ("cpu", None)
            and (status and status.recommended != "cpu")
        ):
            st.error(
                "You preferred a GPU backend but the model is on CPU. "
                "Reinstall llama-cpp-python with the matching backend (see install commands above), "
                "then Save acceleration and Reload engine."
            )

    st.subheader("Status")
    st.code(
        f"Model path : {get_model_path() or '(empty → mock mode)'}\n"
        f"Model ready: {model_ready}\n"
        f"GPU layers : {get_gpu_layers()}  (-1=auto, 0=CPU)\n"
        f"Raw docs   : {len(list_raw_documents())}\n"
        f"Chunks     : {chunk_count}\n"
        f"Raw dir    : {settings.raw_dir}\n"
        f"Vector dir : {settings.vectorstore_dir}",
        language=None,
    )

    st.markdown("---")
    st.subheader("Quick start")
    st.markdown(
        """
1. **Settings** → set GGUF path → **Save model path**
2. Check **Hardware acceleration** (CUDA / Metal / ROCm) → Auto → **Save**
3. **Documents** → upload → **Build index**
4. **Chat** → ask questions

If GPU shows as available but backend is CPU, reinstall `llama-cpp-python` with the matching backend.
"""
    )

st.markdown(
    "<div style='text-align:center;color:#475569;font-size:0.8rem;margin-top:1.5rem'>"
    "LocalMind v0.2 · Privacy-first Agentic RAG</div>",
    unsafe_allow_html=True,
)
