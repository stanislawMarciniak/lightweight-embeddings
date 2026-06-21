from __future__ import annotations

import re
from typing import List


SENTENCE_END_REGEX = re.compile(r"(?<=[.!?])\s+")


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into rough sentences using punctuation-based rules.
    """
    text = text.strip()
    if not text:
        return []
    parts = SENTENCE_END_REGEX.split(text)
    sentences = [s.strip() for s in parts if s.strip()]
    return sentences


WORD_REGEX = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in WORD_REGEX.findall(text)]

