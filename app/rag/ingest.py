from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import threading
import time

from sentence_transformers import SentenceTransformer

from app.storage.metadata_store import ChunkMeta, MetadataStore
from app.storage.vector_store import VectorStore
from app.utils.chunking import ChunkOptions, chunk_pages
from app.utils.pdf_loader import load_pdf_pages
from app.utils.text_cleaning import clean_pages

logger = logging.getLogger(__name__)
_EMBEDDER_CACHE: dict[tuple[str, str | None], SentenceTransformer] = {}
_EMBEDDER_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    pages: int
    chunks: int
    index_size: int # total vectors in FAISS after ingest


def _get_embedder(model_name: str, *, device: str | None = None) -> tuple[SentenceTransformer, bool]:
    key = (model_name, device)

    with _EMBEDDER_CACHE_LOCK:
        cached = _EMBEDDER_CACHE.get(key)
        if cached is not None:
            return cached, True

    model = SentenceTransformer(model_name, device=device)
    with _EMBEDDER_CACHE_LOCK:
        _EMBEDDER_CACHE[key] = model
    return model, False


def ingest_pdf(
    pdf_path: str | Path,
    *,
    doc_id: str | None = None,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    embedding_device: str | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    # Storage ---
    faiss_index_path: str | Path = "data/faiss.index",
    faiss_ids_path: str | Path = "data/faiss_ids.json",
    metadata_path: str | Path = "data/metadata.json",
    log_timing: bool = True,
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
    t_total_start = time.perf_counter()

    # 1 - Loading
    t0 = time.perf_counter()
    pages = load_pdf_pages(pdf_path)
    t_load = time.perf_counter() - t0

    # 2 - Cleaning
    t0 = time.perf_counter()
    pages = clean_pages(pages)
    t_clean = time.perf_counter() - t0

    # 3 - Chunking
    t0 = time.perf_counter()
    options = ChunkOptions(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunk_pages(pages, options=options, doc_id=effective_doc_id)
    t_chunk = time.perf_counter() - t0
    if not chunks:
        if log_timing:
            logger.info(
                "ingest timings | doc_id=%s pages=%d chunks=0 load=%.3fs clean=%.3fs chunk=%.3fs total=%.3fs",
                effective_doc_id,
                len(pages),
                t_load,
                t_clean,
                t_chunk,
                time.perf_counter() - t_total_start,
            )
        return IngestResult(doc_id=effective_doc_id, pages=len(pages), chunks=0, index_size=0)
    
    # 4 - Embedding
    t0 = time.perf_counter()
    model, cache_hit = _get_embedder(embedding_model_name, device=embedding_device)
    t_model = time.perf_counter() - t0

    t0 = time.perf_counter()
    embeddings = model.encode(
        [c.text for c in chunks],
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False # VectorStore handles normalization
    )
    t_embed = time.perf_counter() - t0
    dim = int(embeddings.shape[1])

    # 5 - Saving vectors
    t0 = time.perf_counter()
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
    t_save_vectors = time.perf_counter() - t0

    # 6 - Saving metadata
    t0 = time.perf_counter()
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
    t_save_meta = time.perf_counter() - t0

    t_total = time.perf_counter() - t_total_start
    if log_timing:
        logger.info(
            "ingest timings | doc_id=%s pages=%d chunks=%d load=%.3fs clean=%.3fs chunk=%.3fs "
            "embedder=%.3fs(cache_hit=%s) embed=%.3fs save_vectors=%.3fs save_meta=%.3fs total=%.3fs",
            effective_doc_id,
            len(pages),
            len(chunks),
            t_load,
            t_clean,
            t_chunk,
            t_model,
            cache_hit,
            t_embed,
            t_save_vectors,
            t_save_meta,
            t_total,
        )

    return IngestResult(
        doc_id=effective_doc_id,
        pages=len(pages),
        chunks=len(chunks),
        index_size=len(vs),
    )
