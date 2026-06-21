#!/usr/bin/env python
"""Build a FAQ .json file (importable via the FAQ "Import JSON" button) from a
public QA dataset. Default source is Databricks Dolly-15k (general open-domain
Q&A) - deliberately different from MS MARCO.

Output format matches POST /faq/import:
    [{"question": "...", "answer": "..."}, ...]

Usage (run with an env that has `datasets`, e.g. the experiments venv):
    HF_HOME=$PWD/.hf_cache python scripts/build_faq_dataset.py \
        --dataset dolly --n 500 --out faq_500.json
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Dict, Iterable, List, Optional, Tuple

# Known datasets -> (hf_id, config, split, extractor). The extractor turns a row
# into (question, answer) or None to skip it.
def _dolly(row: Dict) -> Optional[Tuple[str, str]]:
    # Open-domain QA only: closed_qa/information_extraction need the context blob.
    if row.get("category") not in {"open_qa", "general_qa"}:
        return None
    if (row.get("context") or "").strip():
        return None
    return row.get("instruction", ""), row.get("response", "")


def _truthful(row: Dict) -> Optional[Tuple[str, str]]:
    return row.get("question", ""), row.get("best_answer", "")


def _squad(row: Dict) -> Optional[Tuple[str, str]]:
    answers = (row.get("answers") or {}).get("text") or []
    return row.get("question", ""), (answers[0] if answers else "")


def _wiki_qa(row: Dict) -> Optional[Tuple[str, str]]:
    if int(row.get("label", 0)) != 1:  # keep only correct answer sentences
        return None
    return row.get("question", ""), row.get("answer", "")


DATASETS = {
    "dolly": ("databricks/databricks-dolly-15k", None, "train", _dolly),
    "truthful_qa": ("truthful_qa", "generation", "validation", _truthful),
    "squad": ("rajpurkar/squad", None, "train", _squad),
    "wiki_qa": ("microsoft/wiki_qa", None, "train", _wiki_qa),
}


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\n", " ")).strip()


def iter_pairs(rows: Iterable[Dict], extractor, max_q: int, max_a: int) -> Iterable[Tuple[str, str]]:
    for row in rows:
        pair = extractor(row)
        if not pair:
            continue
        q, a = clean(pair[0]), clean(pair[1])
        if not q or not a:
            continue
        if not q.endswith("?"):
            q = q.rstrip(".") + "?"
        if len(q) > max_q or len(a) > max_a:
            continue
        yield q, a


def main() -> None:
    ap = argparse.ArgumentParser(description="Build an importable FAQ .json from a public QA dataset")
    ap.add_argument("--dataset", default="dolly", choices=sorted(DATASETS), help="Source dataset")
    ap.add_argument("--n", type=int, default=500, help="Number of FAQ pairs")
    ap.add_argument("--out", default="faq_500.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-question-chars", type=int, default=160)
    ap.add_argument("--max-answer-chars", type=int, default=600)
    args = ap.parse_args()

    import random

    from datasets import load_dataset

    hf_id, config, split, extractor = DATASETS[args.dataset]
    print(f"[build_faq] loading {hf_id} ({config or 'default'}, split={split}) ...")
    ds = load_dataset(hf_id, config, split=split)

    seen: set[str] = set()
    pairs: List[Dict[str, str]] = []
    for q, a in iter_pairs(ds, extractor, args.max_question_chars, args.max_answer_chars):
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        pairs.append({"question": q, "answer": a})

    print(f"[build_faq] collected {len(pairs)} unique pairs")
    random.Random(args.seed).shuffle(pairs)
    pairs = pairs[: args.n]

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"[build_faq] wrote {len(pairs)} FAQ pairs -> {args.out}")
    if pairs:
        print(f"[build_faq] example: Q: {pairs[0]['question']}\n             A: {pairs[0]['answer'][:120]}")


if __name__ == "__main__":
    main()
