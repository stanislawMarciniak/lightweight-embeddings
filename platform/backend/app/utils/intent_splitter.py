"""Lightweight rule-based multi-intent splitter (no LLM).

Splits compound questions into independent sub-questions, e.g.
    "How do I reset my password and can I do it in the app?"
    -> ["How do I reset my password", "can I do it in the app"]

Prioritizes speed and low cost: pure regex/heuristics, English + Polish
connectors. Falls back to a single intent when no confident split is found.
"""

from __future__ import annotations

import re
from typing import List

# Connectors that typically join two separate intents. Word-boundary matched.
_CONNECTORS = [
    r"\band also\b",
    r"\band can i\b",
    r"\band do i\b",
    r"\band how\b",
    r"\band what\b",
    r"\band is\b",
    r"\band are\b",
    r"\band\b",
    r"\boraz\b",
    r"\b a tak\u017ce\b",
    r"\bczy mog\u0119\b",
    r"\bi czy\b",
    r"\bi jak\b",
    r"\bi gdzie\b",
    r"\bi\b",
]
_HARD_SEP = re.compile(r"\s*[;\n]\s*|(?<=\?)\s+")
_CONNECTOR_RE = re.compile("|".join(_CONNECTORS), flags=re.IGNORECASE)

_MIN_WORDS = 3  # a fragment shorter than this is not treated as a standalone intent
_MAX_INTENTS = 4


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip(" ,.;")


def _find_balanced_split(part: str):
    """Return (left, right) for the first connector that yields >= _MIN_WORDS on both
    sides, or None. Iterating over all matches avoids false positives such as the
    English pronoun "I" colliding with the Polish connector "i"."""
    for m in _CONNECTOR_RE.finditer(part):
        left = _clean(part[: m.start()])
        right = _clean(part[m.end():])
        if len(left.split()) >= _MIN_WORDS and len(right.split()) >= _MIN_WORDS:
            return left, right
    return None


def _looks_like_two_intents(text: str) -> bool:
    if _HARD_SEP.search(text):
        return True
    if len(re.findall(r"\?", text)) >= 2:
        return True
    return _find_balanced_split(text) is not None


def split_intents(text: str) -> List[str]:
    text = _clean(text)
    if not text:
        return []
    if not _looks_like_two_intents(text):
        return [text]

    # 1) split on hard separators first
    parts: List[str] = [p for p in _HARD_SEP.split(text) if p and p.strip()]

    # 2) within each part, split once on the first balanced connector
    out: List[str] = []
    for part in parts:
        split = _find_balanced_split(part)
        if split:
            out.extend(split)
        else:
            out.append(_clean(part))

    out = [p for p in out if len(p.split()) >= 2]
    if not out:
        return [text]
    # de-duplicate while preserving order, cap count
    seen = set()
    deduped = []
    for p in out:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    return deduped[:_MAX_INTENTS]
