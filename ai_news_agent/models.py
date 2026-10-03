from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Item:
    """One news item from any source."""

    title: str
    url: str
    source: str  # human readable source name, e.g. "Hacker News"
    published: Optional[datetime] = None  # timezone-aware UTC
    text: str = ""  # plain-text description / abstract
    points: int = 0  # HN points or similar engagement signal
    comments: int = 0
    source_weight: float = 1.0  # editorial trust in the source
    mentions: int = 1  # how many sources covered it (set by dedupe)
    also_seen_in: list[str] = field(default_factory=list)
    score: float = 0.0  # set by rank()


@dataclass
class Summary:
    summary: str  # what happened, 1-2 sentences
    so_what: str  # why it matters / the opinionated take
