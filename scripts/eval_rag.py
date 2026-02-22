from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from app.rag.generate import INSUFFICIENT_CONTEXT_MSG, generate
from app.rag.retrieve import Retriever
from app.storage.metadata_store import MetadataStore
from app.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalExample:
    sample_id: str
    question: str
    doc_ids: list[str] | None
    expected_doc_ids: list[str] | None
    must_refuse: bool | None


def _normalize_doc_ids(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []

    out: list[str] = []
    if isinstance(value, list):
        for item in value:
            as_str = str(item).strip()
            if as_str and as_str not in out:
                out.append(as_str)
        return out

    as_str = str(value).strip()
    return [as_str] if as_str else []


def _parse_example(raw: Any, idx: int) -> EvalExample:
    if not isinstance(raw, dict):
        raise ValueError(f"Example #{idx} must be a JSON object.")

    question = str(raw.get("question", "")).strip()
    if not question:
        raise ValueError(f"Example #{idx} is missing a non-empty 'question'.")

    must_refuse_raw = raw.get("must_refuse")
    if must_refuse_raw is not None and not isinstance(must_refuse_raw, bool):
        raise ValueError(f"Example #{idx} has non-boolean 'must_refuse'.")

    sample_id = str(raw.get("id", idx))
    return EvalExample(
        sample_id=sample_id,
        question=question,
        doc_ids=_normalize_doc_ids(raw.get("doc_ids")),
        expected_doc_ids=_normalize_doc_ids(raw.get("expected_doc_ids")),
        must_refuse=must_refuse_raw,
    )


def load_dataset(path: str | Path) -> list[EvalExample]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            rows = data.get("examples", [])
        elif isinstance(data, list):
            rows = data
        else:
            raise ValueError("JSON dataset must be a list or an object with 'examples'.")
        return [_parse_example(row, i + 1) for i, row in enumerate(rows)]

    examples: list[EvalExample] = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        examples.append(_parse_example(raw, i))
    return examples


def build_retriever(args: argparse.Namespace) -> Retriever:
    index_path = Path(args.faiss_index_path)
    ids_path = Path(args.faiss_ids_path)
    metadata_path = Path(args.metadata_path)

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not ids_path.exists():
        raise FileNotFoundError(f"FAISS ids file not found: {ids_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    dim = int(faiss.read_index(str(index_path)).d)
    vector_store = VectorStore(index_path=index_path, ids_path=ids_path, dim=dim, normalize=True)
    metadata_store = MetadataStore(metadata_path)
    return Retriever(
        vector_store=vector_store,
        metadata_store=metadata_store,
        embed_model_name=args.embedding_model,
        device=args.embedding_device,
    )


def evaluate_example(example: EvalExample, retriever: Retriever, args: argparse.Namespace) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": example.sample_id,
        "question": example.question,
        "doc_ids": example.doc_ids,
        "expected_doc_ids": example.expected_doc_ids,
        "must_refuse": example.must_refuse,
    }

    retrieved = retriever.retrieve(
        example.question,
        top_k=args.top_k,
        min_score=args.min_score,
        doc_ids=example.doc_ids,
    )
    retrieved_doc_ids = sorted({c.doc_id for c in retrieved})
    row["retrieved_doc_ids"] = retrieved_doc_ids
    row["retrieved_count"] = len(retrieved)
    row["top_score"] = float(retrieved[0].score) if retrieved else None

    if example.expected_doc_ids:
        expected = set(example.expected_doc_ids)
        got = set(retrieved_doc_ids)
        row["retrieval_hit"] = bool(expected & got)
        row["retrieval_all"] = expected.issubset(got)
    else:
        row["retrieval_hit"] = None
        row["retrieval_all"] = None

    if args.skip_generation:
        row["answer"] = None
        row["refusal"] = None
        row["has_citation"] = None
        row["grounded_proxy"] = None
        row["refusal_correct"] = None
        return row

    result = generate(
        example.question,
        retriever=retriever,
        model=args.llm_model,
        base_url=args.ollama_base_url,
        top_k=args.top_k,
        min_score=args.min_score,
        doc_ids=example.doc_ids,
    )
    answer = result.answer.strip()
    refusal = answer == INSUFFICIENT_CONTEXT_MSG
    has_citation = bool(re.search(r"\[\d+\]", answer))
    grounded_proxy = refusal if not result.contexts else has_citation

    row["answer"] = answer
    row["refusal"] = refusal
    row["has_citation"] = has_citation
    row["grounded_proxy"] = grounded_proxy
    if example.must_refuse is not None:
        row["refusal_correct"] = (refusal == example.must_refuse)
    else:
        row["refusal_correct"] = None
    return row


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [r[key] for r in rows if isinstance(r.get(key), bool)]
    if not vals:
        return None
    return float(sum(1 for v in vals if v) / len(vals))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    top_scores = [r["top_score"] for r in rows if isinstance(r.get("top_score"), float)]
    retrieved_counts = [int(r.get("retrieved_count", 0)) for r in rows]
    errors = sum(1 for r in rows if r.get("error"))

    return {
        "examples": len(rows),
        "errors": errors,
        "avg_retrieved_count": (sum(retrieved_counts) / len(retrieved_counts)) if retrieved_counts else 0.0,
        "avg_top_score": (sum(top_scores) / len(top_scores)) if top_scores else None,
        "retrieval_hit_rate": _rate(rows, "retrieval_hit"),
        "retrieval_all_rate": _rate(rows, "retrieval_all"),
        "grounded_proxy_rate": _rate(rows, "grounded_proxy"),
        "refusal_accuracy": _rate(rows, "refusal_correct"),
    }


def _fmt_rate(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval and groundedness on a local dataset (JSON/JSONL)."
    )
    parser.add_argument("--dataset", required=True, help="Path to JSON or JSONL dataset.")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("CITERAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        help="SentenceTransformers model used for retrieval.",
    )
    parser.add_argument(
        "--embedding-device",
        default=None,
        help="Optional device for embeddings (e.g. cpu, cuda, mps).",
    )
    parser.add_argument(
        "--llm-model",
        default=os.getenv("CITERAG_OLLAMA_MODEL", "llama3.2"),
        help="Ollama model used for generation checks.",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=os.getenv("CITERAG_OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Ollama server base URL.",
    )
    parser.add_argument("--faiss-index-path", default="data/faiss.index")
    parser.add_argument("--faiss-ids-path", default="data/faiss_ids.json")
    parser.add_argument("--metadata-path", default="data/metadata.json")
    parser.add_argument("--skip-generation", action="store_true", help="Evaluate retrieval only.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of examples.")
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
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

    examples = load_dataset(args.dataset)
    if args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        logger.error("No examples found in dataset.")
        return 1

    retriever = build_retriever(args)
    rows: list[dict[str, Any]] = []

    for i, ex in enumerate(examples, start=1):
        logger.info("Evaluating %d/%d id=%s", i, len(examples), ex.sample_id)
        try:
            rows.append(evaluate_example(ex, retriever, args))
        except Exception as exc:
            logger.exception("Evaluation failed for id=%s: %s", ex.sample_id, exc)
            rows.append(
                {
                    "id": ex.sample_id,
                    "question": ex.question,
                    "error": str(exc),
                }
            )

    summary = summarize(rows)
    print(
        "Summary: "
        f"examples={summary['examples']} "
        f"errors={summary['errors']} "
        f"retrieval_hit_rate={_fmt_rate(summary['retrieval_hit_rate'])} "
        f"retrieval_all_rate={_fmt_rate(summary['retrieval_all_rate'])} "
        f"grounded_proxy_rate={_fmt_rate(summary['grounded_proxy_rate'])} "
        f"refusal_accuracy={_fmt_rate(summary['refusal_accuracy'])}"
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote evaluation report to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
