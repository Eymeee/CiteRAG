from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from pypdf import PdfReader


PathLike = Union[str, Path]


@dataclass(frozen=True) # Easier definition + making instances immutable
class PDFPage:
    page_number: int  # 1-based
    text: str


class PDFLoadError(RuntimeError):
    """Raised when a PDF cannot be read or text cannot be extracted."""


def load_pdf_pages(path: PathLike, *, normalize_whitespace: bool = True) -> List[PDFPage]:
    """
    Load a PDF and return extracted text per page.

    Notes:
    - The * in the arguments is just to make calls clearer and prevent accidental argument mix-ups.
    - Works best for PDFs with an embedded text layer.
    - Scanned PDFs (images) typically return empty text; OCR is required.
    """
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {pdf_path.name}")

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        raise PDFLoadError(f"Failed to open PDF: {pdf_path}") from e

    # Encrypted PDFs may require a password; we keep it minimal and explicit.
    if getattr(reader, "is_encrypted", False):
        try:
            # Try empty password first; if it fails, raise.
            ok = reader.decrypt("")  # returns 0/False on failure in many cases
            if not ok:
                raise PDFLoadError(
                    f"PDF is encrypted and cannot be decrypted without a password: {pdf_path}"
                )
        except Exception as e:
            raise PDFLoadError(
                f"PDF is encrypted and cannot be decrypted: {pdf_path}"
            ) from e

    pages: List[PDFPage] = []
    for index, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            raise PDFLoadError(f"Failed extracting text from page {index + 1} of {pdf_path}") from e

        if normalize_whitespace:
            text = _normalize_ws(text)

        pages.append(PDFPage(page_number=index + 1, text=text))

    return pages


def load_pdf_text(path: PathLike, *, normalize_whitespace: bool = True) -> str:
    """Convenience: concatenate all page texts separated by blank lines."""
    pages = load_pdf_pages(path, normalize_whitespace=normalize_whitespace)
    return "\n\n".join(p.text for p in pages if p.text)


def _normalize_ws(text: str) -> str:
    # Minimal normalization: collapse weird whitespace while keeping paragraph breaks.
    # 1 - Normalize line endings
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2 - Trim trailing spaces on lines
    t = "\n".join(line.rstrip() for line in t.split(sep="\n"))

    # 3 - Collapse excessive blank lines (keep at most 2)
    normalized_lines = []
    blank_count = 0
    for line in t.split("\n"):
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                normalized_lines.append("")
        else:
            blank_count = 0
            normalized_lines.append(line)

    # 4 - Final trim
    return "\n".join(normalized_lines).strip()
