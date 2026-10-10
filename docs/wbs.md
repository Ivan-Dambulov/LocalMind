# Work Breakdown Structure – LocalMind

## Project Goal
Build a privacy-first Agentic RAG system that demonstrates production-grade Generative AI skills suitable for a Generative AI Specialist role.

## Success Criteria
- Documents can be ingested and queried with hybrid retrieval
- Agent correctly routes between local RAG and web search
- Answers include citations
- Evaluation harness produces measurable metrics on a golden set
- Clean README + architecture documentation
- Demo UI that a recruiter can run in < 5 minutes

---

## Phase 1 – Foundation (Days 1-2)
| ID | Task | Est. | Status |
|----|------|------|--------|
| 1.1 | Project scaffolding + config | 2h | Done |
| 1.2 | Document loaders (PDF, DOCX, TXT, MD) | 3h | Done |
| 1.3 | Chunking strategy + metadata | 2h | Done |
| 1.4 | Embeddings + Chroma vector store | 3h | Done |
| 1.5 | Basic ingest script | 1h | Done |
| 1.6 | Streamlit skeleton | 2h | Done |

## Phase 2 – Hybrid Retrieval + Quality (Days 3-4)
| ID | Task | Est. | Status |
|----|------|------|--------|
| 2.1 | BM25 index | 2h | Done |
| 2.2 | Score fusion (hybrid) | 2h | Done |
| 2.3 | Golden question set (30+ Qs) | 3h | Planned |
| 2.4 | Evaluation metrics (faithfulness, relevance) | 4h | Planned |
| 2.5 | Improve chunking / metadata based on eval | 3h | Planned |

## Phase 3 – Agentic Layer (Days 5-6)
| ID | Task | Est. | Status |
|----|------|------|--------|
| 3.1 | Tool decision prompt + routing | 2h | Done |
| 3.2 | Local RAG tool | 1h | Done |
| 3.3 | Web search tool (DuckDuckGo) | 1h | Done |
| 3.4 | Final answer generation with citations | 2h | Done |
| 3.5 | Agent result packaging | 1h | Done |

## Phase 4 – Polish & Portfolio (Day 7+)
| ID | Task | Est. | Status |
|----|------|------|--------|
| 4.1 | Real LLM integration (GGUF) | 2h | Partial |
| 4.2 | Better UI (sources, tool visualization) | 3h | Planned |
| 4.3 | Architecture diagram + decision log | 2h | Planned |
| 4.4 | “Path to production” section | 2h | Planned |
| 4.5 | Demo video + README polish | 3h | Planned |
| 4.6 | Optional FastAPI endpoint | 3h | Planned |

---

## Risks & Mitigations
- Local LLM quality may be weak → support optional cloud LLM for demo
- Embedding model download time → document it clearly
- Golden set creation takes time → start with 15 high-quality questions
