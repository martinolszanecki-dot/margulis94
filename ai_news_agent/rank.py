from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

from .models import Item

# Topics this account cares about. Weights are editorial, tune freely.
TOPIC_BOOSTS = {
    "agent": 1.0, "agents": 1.0, "agentic": 1.0, "mcp": 0.8, "tool use": 0.6,
    "coding": 0.4, "benchmark": 0.3, "open-source": 0.4, "open source": 0.4,
    "release": 0.3, "launch": 0.3, "gpt": 0.3, "claude": 0.3, "gemini": 0.3,
    "reasoning": 0.3, "safety": 0.2, "regulation": 0.2,
}


def topic_score(item: Item) -> float:
    blob = f"{item.title} {item.text[:300]}".lower()
    # title hits count double
    title = item.title.lower()
    s = 0.0
    for term, w in TOPIC_BOOSTS.items():
        if term in title:
            s += 2 * w
        elif term in blob:
            s += w
    return min(s, 4.0)


def score(item: Item, now: Optional[datetime] = None) -> float:
    now = now or datetime.now(timezone.utc)
    age_h = 24.0
    if item.published:
        age_h = max((now - item.published).total_seconds() / 3600, 0.0)
    recency = math.exp(-age_h / 24.0)  # ~37% left after one day
    engagement = math.log1p(item.points + 2 * item.comments) / 2.0
    coverage = 1.5 * (item.mentions - 1)
    base = 1.0 + topic_score(item) + engagement + coverage
    return round(base * item.source_weight * (0.4 + 0.6 * recency), 3)


def rank(items: list[Item], now: Optional[datetime] = None, per_source_cap: int = 3) -> list[Item]:
    """Score, sort, and cap items per source so one feed can't dominate."""
    for i in items:
        i.score = score(i, now)
    out: list[Item] = []
    counts: dict[str, int] = {}
    for i in sorted(items, key=lambda x: x.score, reverse=True):
        if counts.get(i.source, 0) >= per_source_cap:
            continue
        counts[i.source] = counts.get(i.source, 0) + 1
        out.append(i)
    return out
