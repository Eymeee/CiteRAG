from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
from pathlib import Path
from uuid import uuid4

import faiss
import streamlit as st
from dotenv import load_dotenv

# Allow running from either repo root or the ui/ directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from app.rag.delete import hard_delete_documents
from app.rag.generate import generate
from app.rag.ingest import ingest_pdf
from app.rag.retrieve import Retriever
from app.storage.metadata_store import MetadataStore
from app.storage.vector_store import VectorStore

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
LOG_LEVEL = os.getenv("CITERAG_LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


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


def _index_dim() -> int | None:
    if not FAISS_INDEX_PATH.exists():
        return None
    return int(faiss.read_index(str(FAISS_INDEX_PATH)).d)


@st.cache_resource(show_spinner=False)
def _get_retriever(index_dim: int, embedding_model_name: str) -> Retriever:
    vector_store = VectorStore(
        index_path=FAISS_INDEX_PATH,
        ids_path=FAISS_IDS_PATH,
        dim=index_dim,
        normalize=True,
    )
    metadata_store = MetadataStore(METADATA_PATH)
    return Retriever(
        vector_store=vector_store,
        metadata_store=metadata_store,
        embed_model_name=embedding_model_name,
    )


def _text_preview(text: str, max_chars: int = 320) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _page_label(page_start: int | None, page_end: int | None) -> str:
    if page_start is None:
        return "unknown"
    if page_end is None or page_end == page_start:
        return str(page_start)
    return f"{page_start}-{page_end}"


def _serialize_source(chunk, index: int) -> dict:
    return {
        "id": index,
        "doc_id": chunk.doc_id,
        "chunk_id": chunk.chunk_id,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "score": float(chunk.score),
        "text_preview": _text_preview(chunk.text),
    }


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("Sources", expanded=False):
        for src in sources:
            label = _page_label(src["page_start"], src["page_end"])
            st.markdown(
                f"**[{src['id']}] {src['doc_id']} | page {label} | `{src['chunk_id']}` | score={src['score']:.4f}**"
            )
            st.caption(src["text_preview"])


def _ingest_uploaded_pdf(uploaded_file) -> tuple[bool, str]:
    payload = uploaded_file.getvalue()
    if not payload:
        return False, "Uploaded file is empty."

    doc_id = _build_doc_id(uploaded_file.name, payload)
    metadata_store = MetadataStore(METADATA_PATH)
    if doc_id in set(metadata_store.list_docs()):
        doc_id = f"{doc_id}-{uuid4().hex[:6]}"

    temp_path = UPLOADS_DIR / f"{uuid4().hex}.pdf"
    temp_path.write_bytes(payload)
    try:
        result = ingest_pdf(
            temp_path,
            doc_id=doc_id,
            embedding_model_name=EMBEDDING_MODEL_NAME,
            faiss_index_path=FAISS_INDEX_PATH,
            faiss_ids_path=FAISS_IDS_PATH,
            metadata_path=METADATA_PATH,
        )
    finally:
        if temp_path.exists():
            temp_path.unlink()

    _get_retriever.clear()
    return (
        True,
        f"Indexed `{result.doc_id}` ({result.pages} pages, {result.chunks} chunks, index size {result.index_size}).",
    )


def _delete_documents(doc_ids: list[str]) -> tuple[bool, str]:
    result = hard_delete_documents(
        doc_ids=doc_ids,
        metadata_path=METADATA_PATH,
        faiss_index_path=FAISS_INDEX_PATH,
        faiss_ids_path=FAISS_IDS_PATH,
    )
    _get_retriever.clear()
    st.session_state.messages = []

    remaining_docs = MetadataStore(METADATA_PATH).list_docs()
    st.session_state["active_doc_ids"] = remaining_docs.copy()
    st.session_state["docs_to_delete"] = []
    st.session_state["confirm_delete"] = False

    return (
        True,
        f"Deleted {result.deleted_chunks} chunks from {len(result.deleted_doc_ids)} document(s).",
    )


def main() -> None:
    st.set_page_config(page_title="CiteRAG", layout="wide")
    _ensure_dirs()

    st.title("CiteRAG MVP")
    st.caption("Upload PDF documents, then ask grounded questions with citations.")
    flash_success = st.session_state.pop("flash_success", None)
    if flash_success:
        st.success(flash_success)
    flash_error = st.session_state.pop("flash_error", None)
    if flash_error:
        st.error(flash_error)

    metadata_store = MetadataStore(METADATA_PATH)
    docs = metadata_store.list_docs()
    active_doc_ids: list[str] = []

    with st.sidebar:
        st.subheader("Settings")
        top_k = st.slider("Top-k chunks", min_value=1, max_value=20, value=6)
        min_score_enabled = st.toggle("Use minimum similarity score", value=False)
        min_score = st.slider(
            "Minimum score",
            min_value=-1.0,
            max_value=1.0,
            value=0.25,
            step=0.01,
            disabled=not min_score_enabled,
        )
        if not min_score_enabled:
            min_score = None

        st.divider()
        st.write(f"Indexed documents: {len(docs)}")
        if docs:
            existing_active = st.session_state.get("active_doc_ids")
            if not isinstance(existing_active, list):
                st.session_state["active_doc_ids"] = docs.copy()
            else:
                kept = [d for d in existing_active if d in docs]
                st.session_state["active_doc_ids"] = kept or docs.copy()
            existing_delete = st.session_state.get("docs_to_delete")
            if not isinstance(existing_delete, list):
                st.session_state["docs_to_delete"] = []
            else:
                st.session_state["docs_to_delete"] = [d for d in existing_delete if d in docs]

            active_doc_ids = st.multiselect(
                "Answer from documents",
                options=docs,
                key="active_doc_ids",
            )
            st.caption("\n".join(docs))

            st.divider()
            st.subheader("Manage indexed docs")
            docs_to_delete = st.multiselect(
                "Delete indexed documents",
                options=docs,
                key="docs_to_delete",
            )
            confirm_delete = st.checkbox(
                "Confirm permanent deletion",
                key="confirm_delete",
            )
            if st.button(
                "Delete selected documents",
                disabled=not docs_to_delete or not confirm_delete,
            ):
                with st.spinner("Deleting selected documents..."):
                    try:
                        ok, message = _delete_documents(docs_to_delete)
                    except Exception as exc:
                        ok, message = False, f"Deletion failed: {exc}"
                if ok:
                    st.session_state["flash_success"] = message
                    st.rerun()
                else:
                    st.error(message)
        else:
            st.session_state["active_doc_ids"] = []
            st.session_state["docs_to_delete"] = []
            st.session_state["confirm_delete"] = False
            st.caption("No documents indexed yet.")

    st.subheader("1) Upload and Index")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if st.button("Index document", type="primary", disabled=uploaded_file is None):
        with st.spinner("Indexing document..."):
            try:
                ok, message = _ingest_uploaded_pdf(uploaded_file)
            except Exception as exc:
                ok, message = False, f"Ingestion failed: {exc}"
        if ok:
            st.success(message)
        else:
            st.error(message)

    st.subheader("2) Ask Questions")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _render_sources(message.get("sources", []))

    question = st.chat_input("Ask something about your indexed PDFs")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    metadata_store = MetadataStore(METADATA_PATH)
    if len(metadata_store) == 0:
        answer = "No documents are indexed yet. Upload and index at least one PDF first."
        assistant_message = {"role": "assistant", "content": answer, "sources": []}
        st.session_state.messages.append(assistant_message)
        with st.chat_message("assistant"):
            st.markdown(answer)
        return

    all_doc_ids = metadata_store.list_docs()
    selected_doc_ids = [d for d in active_doc_ids if d in all_doc_ids]
    if not selected_doc_ids:
        answer = "Select at least one document in the sidebar before asking a question."
        assistant_message = {"role": "assistant", "content": answer, "sources": []}
        st.session_state.messages.append(assistant_message)
        with st.chat_message("assistant"):
            st.markdown(answer)
        return
    query_doc_ids = None if len(selected_doc_ids) == len(all_doc_ids) else selected_doc_ids

    dim = _index_dim()
    if dim is None:
        answer = "Vector index is missing. Re-index a document."
        assistant_message = {"role": "assistant", "content": answer, "sources": []}
        st.session_state.messages.append(assistant_message)
        with st.chat_message("assistant"):
            st.markdown(answer)
        return

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                retriever = _get_retriever(dim, EMBEDDING_MODEL_NAME)
                result = generate(
                    question,
                    retriever=retriever,
                    model=OLLAMA_MODEL,
                    base_url=OLLAMA_BASE_URL,
                    top_k=top_k,
                    min_score=min_score,
                    doc_ids=query_doc_ids,
                )
                sources = [_serialize_source(c, i) for i, c in enumerate(result.contexts, start=1)]
                st.markdown(result.answer)
                _render_sources(sources)
            except Exception as exc:
                result = None
                sources = []
                st.error(f"Generation failed: {exc}")

    if result is not None:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.answer,
                "sources": sources,
            }
        )


if __name__ == "__main__":
    main()
