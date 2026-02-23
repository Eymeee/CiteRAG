# Changelog

All notable changes to this project are documented in this file.

## [0.1.0] - 2026-02-23

### Added

- FastAPI endpoints: `POST /documents/upload`, `GET /documents`, `POST /documents/delete`, `POST /chat`.
- End-to-end PDF RAG pipeline: PDF loading/cleaning, chunking, embedding, FAISS indexing, metadata persistence, and generation with Ollama.
- Streamlit UI for upload, indexing, question answering, source display, and settings
- Document-scoped retrieval (choose which indexed docs to answer from)
- Hard delete of indexed documents with FAISS rebuild
- Test coverage for chunking, vector store, retrieval, API, and document deletion flows
- Utility scripts: `scripts/build_index.py`, `scripts/eval_rag.py`
- Docker support via `docker/Dockerfile`

### Changed

- Default Ollama model guidance to low-memory local model (`llama3.2:1b`) in env example/docs.

### Notes

- This is the first MVP release.
