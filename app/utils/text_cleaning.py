from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Sequence, TypeVar


# Common PDF ligatures and oddities in extracted text
_LIGATURES = {
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "ft",
    "ﬆ": "st",
}

# Some PDFs contain “soft hyphen” (U+00AD) which should typically be removed
_SOFT_HYPHEN = "\u00ad"

# Control characters except common whitespace we want to preserve (tab/newline)
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

# Hyphenation across line breaks: "exam-\nple" -> "example"
_DEHYPHENATE_RE = re.compile(r"(\w)-\n(\w)")

# Convert single newlines in prose into spaces, while keeping paragraph breaks.
# We treat 2+ newlines as paragraph separators.
_SINGLE_NEWLINE_IN_PARA_RE = re.compile(r"(?<!\n)\n(?!\n)")


@dataclass(frozen=True)
class CleanOptions:
    """
    Options for cleaning extracted PDF text.

    Defaults are tuned for RAG ingestion:
    - preserve paragraph boundaries
    - remove common PDF artifacts
    - avoid aggressive heuristics that might delete meaningful content
    """
    unicode_normal_form: str = "NFKC"
    remove_control_chars: bool = True
    replace_ligatures: bool = True
    remove_soft_hyphen: bool = True
    dehyphenate_linebreaks: bool = True
    normalize_quotes: bool = False  # conservative off by default
    join_single_newlines: bool = True
    collapse_whitespace: bool = True


def clean_text(text: str, *, options: Optional[CleanOptions] = None) -> str:
    """
    Clean extracted text from PDFs for downstream chunking/embedding.

    This is intentionally conservative: it improves consistency without trying
    to infer complex layout structure.
    """
    if not text:
        return ""

    opt = options or CleanOptions()

    # Normalize line endings early
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # Unicode normalization (NFKC helps normalize compatibility characters)
    if opt.unicode_normal_form:
        t = unicodedata.normalize(opt.unicode_normal_form, t)

    if opt.remove_control_chars:
        t = _CONTROL_CHARS_RE.sub("", t)

    if opt.remove_soft_hyphen:
        t = t.replace(_SOFT_HYPHEN, "")

    if opt.replace_ligatures:
        for k, v in _LIGATURES.items():
            t = t.replace(k, v)

    if opt.normalize_quotes:
        t = _normalize_quotes(t)

    if opt.dehyphenate_linebreaks:
        # Apply repeatedly in case of multiple occurrences
        # (regex is cheap; loop avoids missing overlapping patterns)
        while True:
            new_t = _DEHYPHENATE_RE.sub(r"\1\2", t)
            if new_t == t:
                break
            t = new_t

    if opt.join_single_newlines:
        # Keep paragraph breaks (2+ newlines) but join line-wrapped prose
        t = _SINGLE_NEWLINE_IN_PARA_RE.sub(" ", t)

    # Trim trailing spaces per line
    t = "\n".join(line.rstrip() for line in t.split("\n"))

    if opt.collapse_whitespace:
        # Collapse spaces/tabs inside lines (but do not kill newlines)
        t = re.sub(r"[ \t]+", " ", t)
        # Collapse excessive blank lines to max 2
        t = re.sub(r"\n{3,}", "\n\n", t)

    return t.strip()


T = TypeVar("T")


def clean_pages(pages: Sequence[T], *, get_text, set_text, options: Optional[CleanOptions] = None) -> List[T]:
    """
    Clean a sequence of page-like objects.

    Works with your PDFPage dataclass (or any structure) by passing:
      - get_text(obj) -> str
      - set_text(obj, new_text) -> obj (or mutate and return obj)
    """
    out: List[T] = []
    for p in pages:
        cleaned = clean_text(get_text(p), options=options)
        out.append(set_text(p, cleaned))
    return out


def _normalize_quotes(t: str) -> str:
    # Minimal quote normalization; off by default to avoid unintended changes.
    # Curly quotes/apostrophes -> straight, em/en dashes -> hyphen.
    return (
        t.replace("“", '"')
        .replace("”", '"')
        .replace("„", '"')
        .replace("’", "'")
        .replace("‘", "'")
        .replace("—", "-")
        .replace("–", "-")
    )
