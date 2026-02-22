import pytest

from app.utils.chunking import ChunkOptions, chunk_pages, chunk_text
from app.utils.pdf_loader import PDFPage


def test_chunk_text_returns_empty_for_blank_text():
    chunks = chunk_text("   ", doc_id="doc1")
    assert chunks == []


def test_chunk_text_validates_options():
    with pytest.raises(ValueError):
        chunk_text("hello", options=ChunkOptions(chunk_size=0))
    with pytest.raises(ValueError):
        chunk_text("hello", options=ChunkOptions(chunk_size=100, chunk_overlap=100))
    with pytest.raises(ValueError):
        chunk_text("hello", options=ChunkOptions(chunk_size=100, chunk_overlap=-1))


def test_chunk_text_drops_tiny_tail_chunk():
    text = "A" * 240
    chunks = chunk_text(
        text,
        options=ChunkOptions(chunk_size=200, chunk_overlap=10, min_chunk_size=80),
        doc_id="doc1",
    )

    assert len(chunks) == 1
    assert chunks[0].id == "doc1::c0000"
    assert len(chunks[0].text) == 200


def test_chunk_pages_sets_doc_id_ids_and_page_ranges():
    pages = [
        PDFPage(page_number=1, text="A" * 220),
        PDFPage(page_number=2, text="B" * 220),
    ]
    chunks = chunk_pages(
        pages,
        options=ChunkOptions(chunk_size=120, chunk_overlap=30, min_chunk_size=1),
        doc_id="doc1",
    )

    assert chunks, "Expected chunks from non-empty pages"
    assert all(c.doc_id == "doc1" for c in chunks)
    assert all(c.id.startswith("doc1::p") for c in chunks)

    # Ensure both pages are represented in chunk metadata.
    assert any(c.page_start == 1 for c in chunks)
    assert any(c.page_start == 2 for c in chunks)
    assert all((c.page_start is None or c.page_end is None or c.page_start <= c.page_end) for c in chunks)
