"""Shared text normalization for the encoder and lexical (BM25) retrieval.

The original encoder tokenized with ``text.lower().split()``, so punctuation
stuck to words ("contest?" -> OOV -> empty embedding) and contractions
("what's") were lost. This module fixes both, and is reused by BM25 so the dense
and lexical views see the same tokens.
"""

from __future__ import annotations

import re
from typing import List

_CONTRACTIONS = {
    "what's": "what is", "where's": "where is", "when's": "when is",
    "who's": "who is", "how's": "how is", "that's": "that is",
    "it's": "it is", "there's": "there is", "here's": "here is",
    "i'm": "i am", "you're": "you are", "we're": "we are", "they're": "they are",
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "cannot", "won't": "will not", "wouldn't": "would not",
    "shouldn't": "should not", "couldn't": "could not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "i've": "i have", "you've": "you have", "i'll": "i will", "you'll": "you will",
    "i'd": "i would",
}

# Generic suffix fallbacks (applied after the explicit map).
_SUFFIX_RULES = [
    ("n't", " not"),
    ("'re", " are"),
    ("'ve", " have"),
    ("'ll", " will"),
    ("'d", " would"),
    ("'m", " am"),
    ("'s", " is"),  # possessive vs "is" is ambiguous; "is" is a harmless stopword for retrieval
]

_WORD_RE = re.compile(r"[a-z0-9]+")


def expand_contractions(text: str) -> str:
    text = text.lower()
    for k, v in _CONTRACTIONS.items():
        text = text.replace(k, v)
    for suf, repl in _SUFFIX_RULES:
        text = text.replace(suf, repl)
    return text


def word_tokens(text: str) -> List[str]:
    """Lowercase, expand contractions, strip punctuation -> alphanumeric tokens."""
    return _WORD_RE.findall(expand_contractions(text))


# Light stemming so "teams" matches "team", "people" stays etc. (suffix trim only).
def stem(token: str) -> str:
    for suf in ("ies", "es", "s"):
        if len(token) > len(suf) + 2 and token.endswith(suf):
            if suf == "ies":
                return token[:-3] + "y"
            return token[: -len(suf)]
    return token


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "for", "and", "or", "but", "if", "as",
    "do", "does", "did", "i", "you", "we", "they", "it", "this", "that",
    "with", "by", "from", "my", "your", "our", "can", "could", "would",
    "should", "will", "shall", "may", "have", "has", "had", "am", "me",
}
_WH_WORDS = {"where", "when", "who", "what", "why", "how", "which", "whom"}


def content_terms(text: str) -> List[str]:
    """Topic terms only (no stopwords, no wh-words), stemmed. Used for OOD gating."""
    return [stem(t) for t in word_tokens(text) if t not in _STOPWORDS and t not in _WH_WORDS]


def lexical_terms(text: str) -> List[str]:
    """Terms used for BM25 ranking: drop stopwords but KEEP wh-words (where vs when
    matters for intent), stemmed."""
    return [stem(t) for t in word_tokens(text) if t not in _STOPWORDS]
