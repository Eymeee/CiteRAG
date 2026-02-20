from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sentence_transformers import SentenceTransformer

from app.storage.metadata_store import ChunkMeta, MetadataStore
from app.storage.vector_store import VectorStore
from app.utils.chunking import ChunkOptions, chunk_pages
from app.utils.pdf_loader import load_pdf_pages
from app.utils.text_cleaning import clean_pages


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    pages: int
    chunks: int
    index_size: int # total vectors in FAISS after ingest


def ingest_pdf(
    pdf_path: str | Path,
    *,
    doc_id: str | None = None,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    # Storage ---
    faiss_index_path: str | Path = "data/faiss.index",
    faiss_ids_path: str | Path = "data/faiss_ids.json",
    metadata_path: str | Path = "data/metadata.json",
) -> IngestResult:
    """
    Minimal ingestion pipeline:
    PDF -> pages -> clean -> chunk -> embed -> FAISS + JSON metadata

    Assumptions:
      - VectorStore uses: cosine via IP + normalization
      - MetadataStore stores ChunkMeta keyed by chunk_id
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    effective_doc_id = doc_id or pdf_path.name

    # 1 - Loading
    pages = load_pdf_pages(pdf_path)

    # 2 - Cleaning
    pages = clean_pages(pages)

    # 3 - Chunking
    options = ChunkOptions(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunk_pages(pages, options=options, doc_id=effective_doc_id)
    if not chunks:
        return IngestResult(doc_id=effective_doc_id, pages=len(pages), chunks=0, index_size=0)
    
    # 4 - Embedding
    model = SentenceTransformer(embedding_model_name)
    embeddings = model.encode(
        [c.text for c in chunks],
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False
    )
    dim = int(embeddings.shape[1])

    # 5 - Saving vectors
    vs = VectorStore(
        index_path=faiss_index_path,
        ids_path=faiss_ids_path,
        dim=dim,
        normalize=True
    )
    vs.add(
        embeddings=embeddings,
        chunk_ids=[c.id for c in chunks],
        save=True
    )

    # 6 - Saving metadata
    meta_store = MetadataStore(metadata_path)
    metas = [
        ChunkMeta(
            chunk_id=c.id,
            doc_id=effective_doc_id,
            text=c.text,
            page_start=c.page_start,
            page_end=c.page_end,
            chunk_index=c.chunk_index
        )
        for c in chunks
    ]
    meta_store.add_chunkmeta_list(metas=metas, save=True)

    return IngestResult(
        doc_id=effective_doc_id,
        pages=len(pages),
        chunks=len(chunks),
        index_size=len(vs),
    )