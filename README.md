<img width="2532" height="1197" alt="Screenshot 2 - 2026-10-10 210247" src="https://github.com/user-attachments/assets/977d6d59-70a6-4fd2-b33d-97f68460db97" />

# LocalMind

**Privacy-first Agentic RAG Knowledge Assistant**

LocalMind is a local-first Agentic RAG system that answers questions over **your documents** and the **web**.  
An agent decides which tools to use, hybrid retrieval finds the right context, and answers include source citations.

> Built to demonstrate production-oriented Generative AI skills: RAG, embeddings, vector databases, AI agents, evaluation, and end-to-end project ownership.

---

## Features

| Feature | Description |
|---------|-------------|
| **Agentic routing** | Chooses `local_rag`, `web_search`, `both`, or `none` per query |
| **Hybrid retrieval** | Dense embeddings + BM25 keyword search with score fusion |
| **Citations** | Answers reference source files (and pages when available) |
| **Local inference** | GGUF models via `llama-cpp-python` with auto GPU detection |
| **Document ingestion** | PDF, DOCX, TXT, Markdown |
| **Modern UI** | Streamlit chat interface with tool badges and source chips |
| **Clean architecture** | Ingestion → Retrieval → Agent → UI, easy to extend |

---

## Architecture

```text
User Query
    |
    v
+------------------+
|  Agent (ReAct)   |  decides tools
+--------+---------+
         |
    +----+----+-----------+
    |         |           |
    v         v           v
Local RAG  Web Search  (future tools)
(hybrid)   (DuckDuckGo)
    |
    v
Retriever
  - Dense (sentence-transformers)
  - BM25
  - Score fusion
    |
    v
Vector Store (Chroma) + metadata
    |
    v
LLM (local GGUF)
    |
    v
Answer + Citations
```

**Inference** reuses hardware-aware loading logic adapted from [LocalBot](https://github.com/Ivan-Dambulov/LocalBot) (CUDA / Metal / ROCm / CPU).

---

## Project structure

```text
LocalMind/
├── src/
│   ├── ingestion/       # Load, chunk documents
│   ├── retrieval/       # Embeddings, Chroma, hybrid search
│   ├── inference/       # GGUF runtime + hardware detection
│   ├── agent/           # Tool routing + prompts
│   ├── evaluation/      # Golden set + metrics (in progress)
│   ├── ui/              # Streamlit app
│   ├── llm.py           # LLM adapter
│   └── config.py        # Settings (.env)
├── data/
│   ├── raw/             # Put your documents here
│   ├── processed/       # Chunk JSONL
│   └── vectorstore/     # Chroma DB
├── docs/                # Architecture, WBS
├── scripts/
│   ├── ingest.py
│   └── evaluate.py
├── requirements.txt
└── README.md
```

---

## Quick start

### 1. Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure the model

Create a `.env` file in the project root:

```env
LLM_MODEL_PATH=C:/path/to/your/model.gguf
```

Use **forward slashes** on Windows.  
Any GGUF model works (e.g. Qwen, Llama, Mistral quantized).

Without a model path the app runs in **mock mode** (pipeline still works; answers are placeholders).

### 3. Add documents

Copy PDF, DOCX, TXT, or Markdown files into:

```text
data/raw/
```

### 4. Ingest

```bash
python scripts/ingest.py
```

This loads, chunks, embeds, and stores documents in Chroma.

### 5. Run the UI

```bash
streamlit run src/ui/app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## How it works

### Ingestion

1. Load documents (`loader.py`)
2. Split into overlapping chunks (`chunker.py`)
3. Embed with SentenceTransformers
4. Upsert into Chroma + keep chunks for BM25

### Retrieval

For each query:

1. **Dense search** — semantic similarity in Chroma  
2. **BM25** — keyword match  
3. **Fusion** — `score = α × dense + (1 − α) × BM25`  
4. Return top-k chunks with metadata  

`α` is adjustable in the UI sidebar (1.0 = pure semantic, 0.0 = pure keyword).

### Agent

1. LLM chooses a tool: local docs, web, both, or none  
2. Tools run and build context  
3. Final answer is generated with citation instructions  

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL_PATH` | _(empty)_ | Path to GGUF model |
| `LLM_CONTEXT_SIZE` | `4096` | Context window |
| `LLM_TEMPERATURE` | `0.1` | Generation temperature |
| `HYBRID_ALPHA` | `0.6` | Dense vs BM25 weight |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |

All of these can also be changed in `src/config.py`.

---

## Tech stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.10+ |
| Embeddings | `sentence-transformers` |
| Vector DB | Chroma |
| Keyword search | `rank-bm25` |
| LLM | `llama-cpp-python` (GGUF) |
| Web search | DuckDuckGo (`ddgs`) — no API key |
| UI | Streamlit |
| Documents | `pypdf`, `python-docx` |
| Config | `pydantic-settings` |

---

## Roadmap

| Phase | Item | Status |
|-------|------|--------|
| 1 | Ingestion + chunking + Chroma | Done |
| 1 | Hybrid retrieval (dense + BM25) | Done |
| 1 | Agentic tool routing | Done |
| 1 | Streamlit UI | Done |
| 1 | Local GGUF inference | Done |
| 2 | Golden evaluation set + metrics | Planned |
| 3 | Document upload from UI | Planned |
| 3 | Streaming answers | Planned |
| 4 | FastAPI endpoint | Planned |
| 4 | Reranker + better chunking | Planned |

---

## Evaluation philosophy

A RAG system should not ship on “it seems to work.”

- Golden question set (`data/golden_set.json`)
- Metrics: faithfulness, answer relevance, context precision
- Track scores when changing chunk size, hybrid α, or prompts
- Document failure cases and fixes

(Evaluation harness is next on the roadmap.)

---

## Path to production

What would come next for a real deployment:

- API layer (FastAPI) + auth
- Incremental / scheduled ingestion
- Observability (latency, retrieval quality, feedback)
- Optional cloud LLM backend alongside local models
- Multi-user isolation and document ACL
- Stronger retrieval (reranker, hybrid with filters)

See `docs/architecture.md` and `docs/wbs.md` for design notes and work breakdown.

---

## Related project

**[LocalBot](https://github.com/Ivan-Dambulov/LocalBot)** — private desktop AI assistant (local GGUF, hardware detection, web search, document chat).  
LocalMind reuses and extends that inference approach inside a focused Agentic RAG architecture.

---

Built by [Ivan Dambulov](https://github.com/Ivan-Dambulov)
