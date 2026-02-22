from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.ingest import IngestResult, ingest_pdf

logger = logging.getLogger(__name__)


def _sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("._")
    return cleaned or "document.pdf"


def _build_doc_id(pdf_path: Path) -> str:
    safe_name = _sanitize_filename(pdf_path.name)
    digest = hashlib.sha1(pdf_path.read_bytes()).hexdigest()[:8]
    return f"{safe_name}-{digest}"


def _collect_pdf_paths(inputs: list[str], recursive: bool) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()

    for raw in inputs:
        p = Path(raw).expanduser()
        if not p.exists():
            logger.warning("Skipping missing path: %s", p)
            continue

        candidates: list[Path] = []
        if p.is_file():
            candidates = [p] if p.suffix.lower() == ".pdf" else []
        elif p.is_dir():
            iterator = p.rglob("*.pdf") if recursive else p.glob("*.pdf")
            candidates = [x for x in iterator if x.is_file()]

        for c in candidates:
            resolved = c.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(resolved)

    return sorted(out)


def _ingest_one(pdf_path: Path, args: argparse.Namespace) -> IngestResult:
    return ingest_pdf(
        pdf_path,
        doc_id=_build_doc_id(pdf_path),
        embedding_model_name=args.embedding_model,
        embedding_device=args.embedding_device,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        faiss_index_path=args.faiss_index_path,
        faiss_ids_path=args.faiss_ids_path,
        metadata_path=args.metadata_path,
        log_timing=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or extend the local FAISS index from PDF files."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="PDF file(s) or directory path(s) containing PDFs.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan directories for PDFs.",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("CITERAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        help="SentenceTransformers model name.",
    )
    parser.add_argument(
        "--embedding-device",
        default=None,
        help="Optional embedding device, e.g. cpu, cuda, mps.",
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--faiss-index-path", default="data/faiss.index")
    parser.add_argument("--faiss-ids-path", default="data/faiss_ids.json")
    parser.add_argument("--metadata-path", default="data/metadata.json")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    pdf_paths = _collect_pdf_paths(args.paths, recursive=args.recursive)
    if not pdf_paths:
        logger.error("No PDF files found from provided paths.")
        return 1

    success = 0
    failure = 0
    final_index_size = 0

    for pdf_path in pdf_paths:
        logger.info("Ingesting: %s", pdf_path)
        try:
            result = _ingest_one(pdf_path, args)
            success += 1
            final_index_size = result.index_size
            logger.info(
                "Done: doc_id=%s pages=%d chunks=%d index_size=%d",
                result.doc_id,
                result.pages,
                result.chunks,
                result.index_size,
            )
        except Exception as exc:
            failure += 1
            logger.exception("Failed to ingest %s: %s", pdf_path, exc)

    print(
        f"Summary: total={len(pdf_paths)} success={success} failure={failure} "
        f"final_index_size={final_index_size}"
    )
    return 0 if failure == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
