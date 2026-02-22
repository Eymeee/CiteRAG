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

Note: API, Streamlit, and scripts auto-load `.env` from the project root.

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

### Optional CLI indexing

```bash
python scripts/build_index.py tests/sample.pdf
python scripts/build_index.py ./docs --recursive
```

## API endpoints

- `POST /documents/upload`: upload and ingest one PDF.
- `GET /documents`: list indexed document ids.
- `POST /chat`: ask a question and get answer + citations.

## Evaluation script

Run retrieval-only evaluation:

```bash
python scripts/eval_rag.py --dataset eval/examples.json --skip-generation
```

Run retrieval + generation evaluation:

```bash
python scripts/eval_rag.py --dataset eval/examples.json --output eval/report.json
```

Supported dataset formats:

- JSON list of examples
- JSON object with `examples` field
- JSONL (one JSON object per line)

Minimal example record:

```json
{
  "id": "q1",
  "question": "What is ...?",
  "doc_ids": ["mydoc.pdf-abc12345"],
  "expected_doc_ids": ["mydoc.pdf-abc12345"],
  "must_refuse": false
}
```

Fields:

- `question` required
- `doc_ids` optional filter for retrieval
- `expected_doc_ids` optional for retrieval metrics
- `must_refuse` optional boolean for refusal accuracy

## Docker

Build image:

```bash
docker build -f docker/Dockerfile -t citerag:latest .
```

Run FastAPI:

```bash
docker run --rm -it \
  -p 8000:8000 \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  --add-host=host.docker.internal:host-gateway \
  citerag:latest
```

Run Streamlit:

```bash
docker run --rm -it \
  -p 8501:8501 \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  --add-host=host.docker.internal:host-gateway \
  citerag:latest \
  streamlit run ui/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

## Notes

- Use the same embedding model for ingestion and retrieval (`CITERAG_EMBEDDING_MODEL`).
- Data persists under `data/` (`faiss.index`, `faiss_ids.json`, `metadata.json`).
- If upload endpoints fail at startup, ensure `python-multipart` is installed.
- If Ollama is outside Docker, keep `CITERAG_OLLAMA_BASE_URL` pointed to the host.

## Development

Run tests:

```bash
pytest -q
```
