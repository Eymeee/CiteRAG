# CiteRAG

Local Retrieval-Augmented Generation (RAG) app for PDF question answering with citations.

## What it does
- Ingests PDFs locally: extract text, clean, chunk, embed, and persist to FAISS.
- Answers questions grounded in retrieved chunks.
- Returns citation context (document, page, chunk id).
- Refuses when context is insufficient.

## Tech stack
- Python + FastAPI + Streamlit
- Ollama (`langchain-ollama`) for local LLM
- SentenceTransformers for embeddings
- FAISS for vector search

## Quick start
1. Create and activate a virtual environment.
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.
```bash
pip install -r requirements.txt
```

3. Configure environment variables.
```bash
cp .env.example .env
```

4. Ensure Ollama is running and pull a model.
```bash
ollama pull llama3.2
```

## Run the app

### FastAPI backend
```bash
uvicorn app.main:app --reload
```
Open: `http://127.0.0.1:8000/docs`

### Streamlit UI
```bash
streamlit run ui/streamlit_app.py
```

## API endpoints
- `POST /documents/upload`: upload and ingest one PDF.
- `GET /documents`: list indexed document ids.
- `POST /chat`: ask a question and get answer + citations.

## Notes
- Use the same embedding model for ingestion and retrieval (`CITERAG_EMBEDDING_MODEL`).
- Data persists under `data/` (`faiss.index`, `faiss_ids.json`, `metadata.json`).
- If upload endpoints fail at startup, ensure `python-multipart` is installed.

## Development
Run tests:
```bash
pytest -q
```
