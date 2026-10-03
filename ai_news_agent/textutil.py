from __future__ import annotations

import html
import re

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_URL = re.compile(r"https?://\S+")

TWEET_LIMIT = 280
URL_LEN = 23  # X counts every link as 23 chars (t.co)


def strip_html(raw: str) -> str:
    """Remove tags/entities and collapse whitespace."""
    if not raw:
        return ""
    text = _TAG.sub(" ", raw)
    text = html.unescape(text)
    return _WS.sub(" ", text).strip()


def sentences(text: str) -> list[str]:
    text = _WS.sub(" ", text or "").strip()
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def truncate(text: str, limit: int) -> str:
    """Truncate at a word boundary, adding an ellipsis when cut."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[: max(limit - 1, 0)]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:-") + "\u2026"


def tweet_length(text: str) -> int:
    """Length as X counts it: every URL is 23 chars."""
    return len(_URL.sub("x" * URL_LEN, text))
