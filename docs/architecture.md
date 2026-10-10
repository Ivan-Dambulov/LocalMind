# LocalMind Architecture

## High-level flow

1. **Ingestion**
   - Loaders extract text + metadata from PDF / DOCX / TXT / MD
   - Recursive character splitter creates overlapping chunks
   - Chunks are embedded with SentenceTransformers and stored in Chroma
   - BM25 index is built in parallel for keyword search

2. **Retrieval (Hybrid)**
   - Dense search via Chroma (cosine)
   - Keyword search via BM25
   - Scores are normalized and fused with weight α
   - Top-k results returned with metadata

3. **Agent**
   - LLM decides tool: `local_rag` | `web_search` | `both` | `none`
   - Selected tool(s) are executed
   - Context is assembled
   - Final answer is generated with explicit citation instructions

4. **Evaluation**
   - Golden set of (question, expected_answer, relevant_doc_ids)
   - Metrics: faithfulness, answer relevance, context precision
   - Results tracked over time to prove improvements

## Design principles
- Local-first (no required API keys)
- Clear module boundaries (ingestion / retrieval / agent / evaluation)
- Measurable quality (never “it seems to work”)
- Easy to extend with new tools
- Production path documented (vector DB swap, caching, monitoring)

## Future production considerations
- Replace in-memory BM25 with a proper sparse index (Elasticsearch / OpenSearch)
- Add reranker (cross-encoder)
- Add document versioning & incremental ingestion
- Observability (Langfuse / Phoenix)
- API gateway + auth
- Async ingestion queue
