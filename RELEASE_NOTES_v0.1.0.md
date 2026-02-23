# CiteRAG v0.1.0

Release date: 2026-02-23

## Highlights

- Local PDF RAG MVP is complete.
- Upload and index PDF documents with FAISS persistence.
- Ask grounded questions with citations (doc/page/chunk).
- Scope answers to selected documents.
- Delete indexed documents with hard delete + index rebuild.

## Included in this release

- Backend API endpoints: `POST /documents/upload`, `GET /documents`, `POST /documents/delete`, `POST /chat`.
- Streamlit UI for indexing, QA, source inspection, document scoping, and deletion.
- Retrieval + generation pipeline with SentenceTransformers + Ollama.
- Automated tests for ingestion-adjacent behavior, retrieval, vector store, API, and deletion flow.

## Runtime notes

- Recommended low-memory generation model: `CITERAG_OLLAMA_MODEL=llama3.2:1b`.
- Ollama server: `CITERAG_OLLAMA_BASE_URL=http://localhost:11434`.

## Known MVP limitations

- No auth/multi-user isolation.
- Deletion currently rebuilds the FAISS index (simple and safe, not optimized for very large corpora).
