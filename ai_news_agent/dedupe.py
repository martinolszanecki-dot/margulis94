from __future__ import annotations

import re
import urllib.parse

from .models import Item

_STOP = {"the", "a", "an", "of", "to", "in", "for", "and", "on", "with", "is", "are", "how",
         "why", "what", "new", "its", "it", "at", "by", "from", "as", "your", "you", "we", "our"}
_TRACKING = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "ref", "source"}


def normalize_url(url: str) -> str:
    p = urllib.parse.urlsplit(url.strip())
    host = p.netloc.lower().removeprefix("www.")
    query = urllib.parse.urlencode(
        sorted((k, v) for k, v in urllib.parse.parse_qsl(p.query) if k not in _TRACKING))
    return urllib.parse.urlunsplit(("https", host, p.path.rstrip("/"), query, ""))


def _stem(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def title_tokens(title: str) -> frozenset[str]:
    words = re.findall(r"[a-z0-9]+(?:\.[0-9]+)*", title.lower())
    return frozenset(_stem(w) for w in words if w not in _STOP and len(w) > 1)


def similar(a: frozenset[str], b: frozenset[str], threshold: float = 0.6) -> bool:
    """Titles match if Jaccard >= threshold, or they share >= 4 distinctive tokens
    covering at least half of the shorter title (catches rewordings of one story)."""
    if not a or not b:
        return False
    shared = len(a & b)
    if shared / len(a | b) >= threshold:
        return True
    return shared >= 4 and shared / min(len(a), len(b)) >= 0.5


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def dedupe(items: list[Item], threshold: float = 0.6) -> list[Item]:
    """Merge duplicates by normalized URL, then by fuzzy title similarity.

    The surviving item keeps the best engagement numbers and records how many
    sources covered the story (``mentions``), a strong "this matters" signal.
    """
    kept: list[Item] = []
    by_url: dict[str, Item] = {}
    tokens: dict[int, frozenset[str]] = {}

    def merge(into: Item, other: Item) -> None:
        into.mentions += 1
        if other.source != into.source and other.source not in into.also_seen_in:
            into.also_seen_in.append(other.source)
        into.points = max(into.points, other.points)
        into.comments = max(into.comments, other.comments)
        into.source_weight = max(into.source_weight, other.source_weight)
        if len(other.text) > len(into.text):
            into.text = other.text
        # Prefer a primary source (higher weight, non-HN) as canonical link
        if other.source_weight > into.source_weight - 1e-9 and other.source != "Hacker News" \
                and into.source == "Hacker News":
            into.url, into.source = other.url, other.source

    for item in items:
        key = normalize_url(item.url)
        if key in by_url:
            merge(by_url[key], item)
            continue
        toks = title_tokens(item.title)
        match = next((k for k in kept if similar(toks, tokens[id(k)], threshold)), None)
        if match:
            merge(match, item)
            by_url[key] = match
            continue
        kept.append(item)
        tokens[id(item)] = toks
        by_url[key] = item
    return kept
