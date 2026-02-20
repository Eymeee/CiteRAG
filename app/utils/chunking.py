from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import re

from app.utils.pdf_loader import PDFPage


@dataclass(frozen=True)
class ChunkOptions:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 200


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str

    # for citations
    doc_id: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None

    # for traceability
    chunk_index: int = 0


def chunk_text(
    text: str,
    *,
    options: Optional[ChunkOptions] = None,
    doc_id: Optional[str] = None
) -> list[Chunk]:
    """
    Chunk a single text -> `text` into overlapping character windows
    """

    opt = options or ChunkOptions()
    _validate_options(opt)

    cleaned = (text or "").strip()
    if not cleaned: 
        return []

    spans = _split_into_spans(
        cleaned,
        opt.chunk_size,
        opt.chunk_overlap,
        opt.min_chunk_size
    )

    chunks: list[Chunk] = []
    for i, span in enumerate(spans):
        chunk_id = _make_chunk_id(doc_id = doc_id, page_start=None, chunk_index=i)
        chunks.append(
            Chunk(
                id=chunk_id,
                text=span,
                doc_id=doc_id,
                chunk_index=i
            )
        )
    return chunks


def chunk_pages(
    pages: list[PDFPage],
    *,
    options: Optional[ChunkOptions] = None,
    doc_id: Optional[str] = None
) -> list[Chunk]:
    """
    Chunk a list of PDFPage objects into overlapping chunks and attach page_start/page_end
    for citation-friendly provenance.

    how (you may ask hhh):
    - Concatenate pages with a clear separator ("\\n\\n") so page boundaries are preserved.
    - Track offsets for each page in the concatenated text.
    - Chunk the concatenated text (with overlap).
    - Map each chunk span back to intersecting page ranges.
    """
    opt = options or ChunkOptions()
    _validate_options(opt)

    if not pages: 
        return []

    page_infos: list[tuple[int, int, int]] = [] # list of (page_number, start_offset, end_offset)
    parts: list[str] = [] # the list where we accumulate: prefixes ("" or "\n\n") + page texts
    cursor = 0

    for p in pages:
        page_num = p.page_number
        page_text = p.text

        prefix = "" if not parts else "\n\n"
        parts.append(prefix + page_text)

        start = cursor + len(prefix)
        end = start + len(page_text)

        page_infos.append((page_num, start, end))

        cursor = end

    full_text = "".join(parts)

    spans_with_offsets = _split_with_offsets(
        full_text,
        opt.chunk_size,
        opt.chunk_overlap,
        opt.min_chunk_size
    )

    chunks: list[Chunk] = []
    for i, (span_text, span_start, span_end) in enumerate(spans_with_offsets):
        
        page_start, page_end = _map_span_to_pages(span_start, span_end, page_infos)
        chunk_id = _make_chunk_id(doc_id=doc_id, page_start=page_start, chunk_index=i)

        chunks.append(
            Chunk(
                id=chunk_id,
                text=span_text,
                doc_id=doc_id,
                page_start=page_start,
                page_end=page_end,
                chunk_index=i
            )
        )
    return chunks


# -------------------------------------------
#              Helper functions 
# -------------------------------------------

def _validate_options(opt: ChunkOptions) -> None:
    if opt.chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if opt.chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if opt.chunk_overlap >= opt.chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")
    if opt.min_chunk_size < 0:
        raise ValueError("min_chunk_size must be >= 0")


def _make_chunk_id(*, doc_id: Optional[str], page_start: Optional[int], chunk_index: int) -> str:
    base = doc_id or "doc"
    if page_start is None:
        return f"{base}::c{chunk_index:04d}"
    return f"{base}::p{page_start:04d}::c{chunk_index:04d}"

def _split_into_spans(text: str, chunk_size: int, overlap: int, min_chunk_size: int) -> List[str]:
    return [t for (t, _, _) in _split_with_offsets(text, chunk_size, overlap, min_chunk_size)]


def _split_with_offsets(
    text: str,
    chunk_size: int,
    overlap: int,
    min_chunk_size: int,
) -> List[Tuple[str, int, int]]:
    """
    Sliding-window chunking with boundary-aware end selection.
    Returns (chunk_text, start, end) offsets into the original text.
    """
    n = len(text)
    if n <= chunk_size:
        return [(text, 0, n)]

    out: List[Tuple[str, int, int]] = []
    start = 0

    while start < n:
        target_end = min(start + chunk_size, n)

        end = _choose_boundary(text, start, target_end)
        if end <= start:
            end = target_end

        chunk = text[start:end].strip()
        if chunk:
            out.append((chunk, start, end))

        if end >= n:
            break

        next_start = max(0, end - overlap)

        # Prevent stalling/infinite loops if boundary selection yields same region
        if out and next_start <= out[-1][1]:
            next_start = out[-1][2]

        start = next_start

    # Drop tiny last chunk if it's too small and likely redundant
    if len(out) >= 2:
        last_text, _, _ = out[-1]
        if len(last_text) < min_chunk_size:
            out.pop()

    return out


def _choose_boundary(text: str, start: int, target_end: int) -> int:
    """
    Prefer splitting near target_end at clean boundaries.

    Order (searching backwards near the end):
      1 - paragraph break (\\n\\n)
      2 - newline (\\n)
      3 - sentence boundary ([.?!] followed by whitespace/end)
      4 - space
      5 - target_end (fallback)
    """
    window = text[start:target_end]

    # Search backwards in the last ~250 chars to keep chunk lengths stable.
    tail_limit = max(0, len(window) - 250)
    tail = window[tail_limit:]

    # 1 - paragraph break
    idx = tail.rfind("\n\n")
    if idx != -1:
        return start + tail_limit + idx + 2

    # 2 - newline
    idx = tail.rfind("\n")
    if idx != -1:
        return start + tail_limit + idx + 1

    # 3 - sentence boundary: pick the last match in the tail
    last_match_end: Optional[int] = None
    for m in re.finditer(r"[.?!](?:\s|$)", tail):
        last_match_end = m.end()
    if last_match_end is not None:
        return start + tail_limit + last_match_end

    # 4 - space
    idx = tail.rfind(" ")
    if idx != -1:
        return start + tail_limit + idx + 1

    return target_end


def _map_span_to_pages(
    span_start: int,
    span_end: int,
    page_infos: List[Tuple[int, int, int]],
) -> Tuple[Optional[int], Optional[int]]:
    """
    Map a (start,end) span in the concatenated text back to page_start/page_end.

    page_infos: list of (page_number, start_offset, end_offset) in the concatenated text.
    """
    touched_pages: List[int] = []
    for page_num, ps, pe in page_infos:
        if pe <= span_start:
            continue
        if ps >= span_end:
            break
        touched_pages.append(page_num)

    if not touched_pages:
        return None, None
    return touched_pages[0], touched_pages[-1]