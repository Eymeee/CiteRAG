from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from uuid import uuid4

import faiss
from fastapi import FastAPI, File, HTTPException, UploadFile
from dotenv import load_dotenv

from app.rag.generate import INSUFFICIENT_CONTEXT_MSG, generate
from app.rag.ingest import ingest_pdf
from app.rag.retrieve import RetrievedChunk, Retriever
from app.rag.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    DocumentsResponse,
    UploadDocumentResponse,
)
from app.storage.metadata_store import MetadataStore
from app.storage.vector_store import VectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = Path("data")
UPLOADS_DIR = DATA_DIR / "uploads"
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
FAISS_IDS_PATH = DATA_DIR / "faiss_ids.json"
METADATA_PATH = DATA_DIR / "metadata.json"

EMBEDDING_MODEL_NAME = os.getenv("CITERAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
if not os.getenv("CITERAG_EMBEDDING_MODEL"):
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

OLLAMA_MODEL = os.getenv("CITERAG_OLLAMA_MODEL", "llama3.2")
if not os.getenv("CITERAG_OLLAMA_MODEL"):
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

OLLAMA_BASE_URL = os.getenv("CITERAG_OLLAMA_BASE_URL", "http://localhost:11434")
if not os.getenv("CITERAG_OLLAMA_BASE_URL"):
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

app = FastAPI(title="CiteRAG API", version="0.1.0")


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("._")
    return cleaned or "document.pdf"


def _build_doc_id(filename: str, content: bytes) -> str:
    safe_name = _sanitize_filename(Path(filename).name)
    digest = hashlib.sha1(content).hexdigest()[:8]
    return f"{safe_name}-{digest}"


def _index_dim() -> int:
    if not FAISS_INDEX_PATH.exists():
        raise HTTPException(status_code=400, detail="No indexed documents found.")
    return int(faiss.read_index(str(FAISS_INDEX_PATH)).d)


def _build_retriever(metadata_store: MetadataStore) -> Retriever:
    dim = _index_dim()
    vector_store = VectorStore(
        index_path=FAISS_INDEX_PATH,
        ids_path=FAISS_IDS_PATH,
        dim=dim,
        normalize=True,
    )
    return Retriever(
        vector_store=vector_store,
        metadata_store=metadata_store,
        embed_model_name=EMBEDDING_MODEL_NAME,
    )


def _text_preview(text: str, max_chars: int = 280) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _to_citation(cid: int, chunk: RetrievedChunk) -> Citation:
    return Citation(
        id=cid,
        doc=chunk.doc_id,
        page=chunk.page_start,
        chunk_id=chunk.chunk_id,
        text_preview=_text_preview(chunk.text),
        score=chunk.score,
    )


@app.post("/documents/upload", response_model=UploadDocumentResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadDocumentResponse:
    _ensure_dirs()

    filename = (file.filename or "").strip()
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    payload = await file.read()
    await file.close()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    metadata_store = MetadataStore(METADATA_PATH)
    doc_id = _build_doc_id(filename, payload)
    if doc_id in set(metadata_store.list_docs()):
        doc_id = f"{doc_id}-{uuid4().hex[:6]}"

    temp_pdf_path = UPLOADS_DIR / f"{uuid4().hex}.pdf"
    temp_pdf_path.write_bytes(payload)

    try:
        result = ingest_pdf(
            temp_pdf_path,
            doc_id=doc_id,
            embedding_model_name=EMBEDDING_MODEL_NAME,
            faiss_index_path=FAISS_INDEX_PATH,
            faiss_ids_path=FAISS_IDS_PATH,
            metadata_path=METADATA_PATH,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc
    finally:
        if temp_pdf_path.exists():
            temp_pdf_path.unlink()

    return UploadDocumentResponse(
        doc_id=result.doc_id,
        pages=result.pages,
        chunks=result.chunks,
        index_size=result.index_size,
    )


@app.get("/documents", response_model=DocumentsResponse)
def list_documents() -> DocumentsResponse:
    _ensure_dirs()
    metadata_store = MetadataStore(METADATA_PATH)
    return DocumentsResponse(documents=metadata_store.list_docs())


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    _ensure_dirs()

    if body.collection_id:
        raise HTTPException(
            status_code=501,
            detail="collection_id filtering is not implemented yet in this MVP.",
        )

    metadata_store = MetadataStore(METADATA_PATH)
    if len(metadata_store) == 0:
        raise HTTPException(
            status_code=400,
            detail="No indexed documents found. Upload a PDF first.",
        )

    selected_doc_ids: list[str] | None = None
    if body.doc_ids is not None:
        selected_doc_ids = sorted({d.strip() for d in body.doc_ids if d and d.strip()})
        if not selected_doc_ids:
            raise HTTPException(status_code=400, detail="doc_ids cannot be empty.")

        available_doc_ids = set(metadata_store.list_docs())
        unknown_doc_ids = [d for d in selected_doc_ids if d not in available_doc_ids]
        if unknown_doc_ids:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown doc_ids: {', '.join(unknown_doc_ids)}",
            )

    retriever = _build_retriever(metadata_store)
    result = generate(
        body.question,
        retriever=retriever,
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        top_k=body.top_k,
        min_score=body.min_score,
        doc_ids=selected_doc_ids,
    )

    citations = [_to_citation(i, c) for i, c in enumerate(result.contexts, start=1)]
    refusal = result.answer == INSUFFICIENT_CONTEXT_MSG

    return ChatResponse(
        answer=result.answer,
        citations=citations,
        refusal=refusal,
        model=result.model,
    )
