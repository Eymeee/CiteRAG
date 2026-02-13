# Project Description: Local Document Intelligence (RAG) System — LangChain + Ollama

## 1) Project Overview

Build an end-to-end **Document Intelligence / Document Q&A** application that lets users:

1. **Upload documents** (PDF)  
2. Automatically **extract text**, **chunk**, **embed**, and **index** the content locally  
3. Ask questions in a chat interface and receive answers that are:
   - **grounded in the documents**
   - include **citations** (document + page + chunk id)
   - refuse when the answer is not supported by retrieved context

This is a classic **RAG (Retrieval-Augmented Generation)** system, implemented with a **free, fully local stack**.

---

## 2) Goals & Success Criteria

### Primary goal (MVP)

A working app where a user can upload a PDF and ask questions, getting:

- an answer
- citations referencing the specific retrieved chunks/pages

### Success criteria

- Upload → index → ask → answer flow works reliably
- Answers do **not hallucinate** when the info is missing
- Sources are displayed clearly and are traceable to pages/chunks
- Index persists between app runs (no re-embedding every time)

### Secondary goals (V1)

- Multi-document support 
- Better retrieval quality (tuning chunk size/top-k, optional reranking)
- Simple evaluation script to measure retrieval + groundedness

---

## 3) Constraints

- **100% free:** no paid APIs, no subscriptions, no credits
- Runs **locally** using open-source models
- Use **LangChain** as the orchestration framework

---

## 4) Recommended Tech Stack (Free / Local)

### Runtime & app

- **Python 3.11+**
- **Backend:** FastAPI
- **UI:** Streamlit

### LLM (local)

- **Ollama** to run local LLMs
- LangChain integration via `ChatOllama`

### Embeddings (local)

- `sentence-transformers`
- LangChain embedding wrapper (e.g., HuggingFace/SentenceTransformer embeddings)

### Vector store (local)

- **FAISS** (simplest, fast, local)

### Document parsing

- `pypdf` (extract text by page)

### Dev tools

- `pytest` (tests)
- `ruff` (linting)
- `python-dotenv` (configuration)
- `Docker`

---

## 5) High-Level Architecture

### A) Ingestion Pipeline

**Input:** PDF  
**Output:** stored chunks + vector index

Steps:

1. Parse PDF into page-level text
2. Clean text
3. Chunk text
4. Generate embeddings for each chunk
5. Store:
   - vectors in vector store (FAISS)
   - chunk metadata (doc name, page, chunk id) in local storage

### B) Query / RAG Pipeline

1. User question
2. Embed question
3. Similarity search (top-k chunks)
4. Build prompt with:
   - user question
   - retrieved chunks + their source ids
   - strict instructions: use context only, cite sources
5. LLM generates:
   - answer
   - citations (chunk ids like [1], [2], …)

---

## 6) Data Model & Metadata

### Chunk object

Each chunk should be stored with metadata:

```json
{
  "doc_id": "uuid-or-hash",
  "doc_name": "myfile.pdf",
  "chunk_id": "docid_0007",
  "text": "chunk text ...",
  "metadata": {
    "page": 4,
    "char_start": 1200,
    "char_end": 1800
  }
}
```

### Why metadata matters

- Enables citations in answers
- Lets you show “Source excerpts” in UI
- Helps debugging retrieval

---

## 7) Chunking Strategy (Baseline)

### Recommended settings

- Chunk size: **800–1200 characters** (or ~200–300 tokens)
- Overlap: **10–15%**
- Preserve page numbers

### Notes

- Start with a simple recursive splitter
- Later improvements:
  - split by headings first
  - better cleaning of headers/footers

---

## 8) Prompting & Grounding Rules

### System rules (must)

- Use **only** the provided context chunks
- If context is insufficient: say you **don’t know**
- Always include citations in the format `[1]`, `[2]` …

### Context formatting example

When you retrieve chunks, format them like:

- `[1] (source=myfile.pdf, page=3, chunk=doc_0001): ...text...`
- `[2] (source=myfile.pdf, page=5, chunk=doc_0007): ...text...`

### Output requirements

- Answer in short paragraphs or bullet points
- Add a “Sources” section:
  - `Sources: [1], [2]`

---

## 9) UI Requirements (Streamlit)

### Screens / components

- Document upload component
- Index/collection selector (V1)
- Chat interface:
  - user question input
  - model answer
  - expandable “Sources” area showing:
    - doc name
    - page number
    - chunk text snippet

### UX requirement

If retrieval score is low or no chunks are relevant:

- respond: “I don’t have enough information in the uploaded documents to answer that.”

---

## 10) API Design (FastAPI)

### Suggested endpoints

- `POST /documents/upload`
  - uploads PDF, triggers ingestion
- `GET /documents`
  - list indexed docs/collections
- `POST /chat`
  - request: `{ question, doc_ids/collection_id }`
  - response: `{ answer, citations: [{id, doc, page, chunk_id, text_preview}] }`

---

## 11) Persistence (Must for MVP)

### Requirement

Indexes and metadata must persist to disk.

- For FAISS: store index file + metadata JSON/SQLite
- For Chroma: persist directory

---

## 12) Project Phases

### Phase 0 — Setup

- Install Python deps
- Install Ollama + pull a model
- Run a “hello world” LangChain chat call

### Phase 1 — MVP (single PDF)

- PDF parsing + page text extraction
- Chunking + embeddings
- Vector store indexing
- Query retrieval + prompt + answer
- Citations displayed

### Phase 2 — V1 (multi-doc)

- Multiple documents
- Collections/workspaces
- Persistent storage
- Better UI sources panel

### Phase 3 — V1.5 (quality)

- Tune chunk size/top-k
- Add simple evaluation script
- Add tests for chunking/retrieval

### Phase 4 — V2 (optional)

- Hybrid search (keyword + vector)
- Reranker
- OCR for scanned PDFs
- Streaming responses

---

## 13) Evaluation

### Why evaluate

RAG systems must be assessed for:

- retrieval quality
- groundedness / faithfulness

### Minimal evaluation plan

Create 20–50 Q/A pairs for 2–3 PDFs:

- question
- expected page/source
- expected short answer

Metrics:

- **Retrieval hit rate**: did top-k contain expected page?
- **Groundedness**: does answer claim anything not in context?

Deliverable:

- `scripts/eval_rag.py` that outputs a simple report

---

## 14) Repository Structure (Recommended)

```bash
CiteRAG/
  app/
    main.py                 # FastAPI entry
    rag/
      ingest.py             # parse -> chunk -> embed -> index
      retrieve.py           # similarity search
      generate.py           # prompt + LLM call
      prompts.py            # prompt templates
      schemas.py            # Pydantic models
    storage/
      vector_store.py       # FAISS wrapper
      metadata_store.py     # JSON/SQLite persistence
    utils/
      pdf_loader.py
      text_cleaning.py
      chunking.py
  ui/
    streamlit_app.py
  scripts/
    build_index.py
    eval_rag.py
  tests/
    test_chunking.py
    test_retrieval.py
  docker/
    Dockerfile
  .env.example
  requirements.txt
  README.md
```

---

## 15) README Template Requirements (Portfolio-Ready)

Your README must include:

- What the project does (1 paragraph)
- Architecture diagram (simple is fine)
- Setup steps (including Ollama)
- Demo usage steps
- Design decisions:
  - chunking parameters
  - vector store choice
  - citation strategy
- Limitations & next steps
- Screenshots/GIFs

---

## 16) Deliverables Checklist

### MVP deliverables

- [ ] Upload PDF
- [ ] Parse by page
- [ ] Chunk with overlap
- [ ] Embed locally
- [ ] Vector search
- [ ] LLM answer via Ollama
- [ ] Citations with page + chunk id
- [ ] Persistence
- [ ] Basic tests

### V1 deliverables

- [ ] Multi-doc collections
- [ ] Better UI source viewer
- [ ] Evaluation script + sample dataset of Q/A

---

## 17) Non-Goals (to avoid scope creep)

- Training your own LLM
- Full enterprise auth/permissions
- Complex multi-agent automation
- Perfect OCR from day one

Ship a solid, grounded MVP first.

---

## 18) Suggested Local Model Guidance (Ollama)

Pick based on your machine:

- If you have limited RAM: choose a smaller 7B/8B model
- If you have more RAM: try bigger or higher-quality quantized models

The project should support swapping models via config.

---

## 20) Final Notes

This project is designed to demonstrate real AI engineering:

- building pipelines (ingestion + retrieval + generation)
- grounding & citations
- persistence
- evaluation

If done cleanly with a strong README + demo, it’s a standout GitHub portfolio project.
