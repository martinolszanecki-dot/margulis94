"""Pluggable summarizers.

Implement the :class:`Summarizer` protocol (one method, ``summarize``) to add
your own backend. Two ship with the project:

* :class:`ExtractiveSummarizer` - no network, no key. Picks the most informative
  leading sentences and attaches a rule-based "so what". Runs out of the box.
* :class:`OpenAICompatibleSummarizer` - any OpenAI-style ``/chat/completions``
  endpoint (OpenAI, xAI, Groq, Together, local servers...). Falls back to the
  extractive summarizer for an item if the call fails.
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from typing import Optional, Protocol, runtime_checkable

from .models import Item, Summary
from .textutil import sentences, truncate

log = logging.getLogger("ai_news_agent")


@runtime_checkable
class Summarizer(Protocol):
    name: str

    def summarize(self, item: Item) -> Summary:  # pragma: no cover - interface
        ...


# --- extractive fallback -----------------------------------------------------

# (category, keywords, short insight tails). Every so_what is title-grounded;
# the tail only adds a builder angle. Never return a bare canned paragraph.
_TAKES = [
    ("agents", ("agentic", "computer use", "tool-use", "tool use", "mcp", "autonomous", "agents", "agent"), (
        "reliability, permissions and cost per task beat raw demo polish",
        "treat tool access as product design, not a checkbox",
        "keep the stack swappable until the winners are actually clear",
    )),
    ("opensource", ("open-source", "open source", "show hn", "github.com", "weights", "apache", "mit license"), (
        "read the code before the hype and measure on your own workload",
        "small public projects often show next year's patterns first",
    )),
    ("release", ("now available", "rolling out", "rolls out", "unveil", "announce", "introduc", "launch", "release"), (
        "re-run your own evals before you switch defaults",
        "treat launch numbers as marketing until someone else reproduces them",
    )),
    ("research", ("arxiv", "we propose", "we present", "benchmark", "paper", "study"), (
        "ask whether it survives messy real-world tasks, not just the paper setup",
        "treat the headline score as a hypothesis for your workload",
    )),
    ("policy", ("lawsuit", "copyright", "privacy", "regulat", "senate", "court", "policy", " bans", " ban ", "banned", "safety", "misconduct", "risk", "unintended", "eval"), (
        "rules and containment decide what you can ship and where",
        "expect more hard stops on live tools, not fewer",
    )),
    ("business", ("valuation", "acquisition", "partnership", "funding", "revenue", "billion", "acquire", "raises", "invest"), (
        "capital still decides who gets compute, and compute who competes",
    )),
]
_DEFAULT_TAILS = (
    "judge it by what changes in your stack this month, not the headline alone",
    "useful only if people still use it a month from now",
)


def _blob(item: Item) -> str:
    return f"{item.title} {item.text[:300]}".lower()


def classify(item: Item) -> str:
    """Pick the category with the strongest keyword hits (longer phrases count more).

    Title hits count double so a body full of the word "agent" cannot bury a
    clearer policy or business cue in the headline.
    """
    title = item.title.lower()
    blob = _blob(item)
    best_cat, best_score = "other", 0
    for cat, kws, _ in _TAKES:
        score = 0
        for k in kws:
            if k in title:
                score += 2 * max(1, len(k.split()))
            elif k in blob:
                # Prefer multi-word / longer cues so "agentic" beats bare "agent".
                score += max(1, len(k.split()))
        if score > best_score:
            best_cat, best_score = cat, score
    return best_cat


def _pick(variants: tuple, key: str, used: set[str]) -> str | None:
    if not variants:
        return None
    start = sum(map(ord, key)) % len(variants)
    for i in range(len(variants)):
        candidate = variants[(start + i) % len(variants)]
        if candidate not in used:
            return candidate
    return None


def _short_title(item: Item, limit: int = 64) -> str:
    title = re.sub(r"\s+", " ", item.title).strip().rstrip(".")
    if len(title) > limit:
        return title[: limit - 3].rstrip() + "..."
    return title


def _body_cue(item: Item) -> str | None:
    """One short concrete cue from the article body (not a full sentence dump)."""
    title_l = item.title.lower()
    for s in sentences(item.text):
        s = re.sub(r"\s+", " ", s).strip().rstrip(".")
        low = s.lower()
        if len(s) < 28 or low.startswith(("the post ", "read more", "click ", "subscribe")):
            continue
        if low == title_l.rstrip("."):
            continue
        # Prefer a cue that adds a fact word not already in the title.
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", s)
                 if w.lower() not in title_l and w.lower() not in {
                     "that", "this", "with", "from", "have", "been", "were", "their", "about"}]
        if not words:
            continue
        cue = s
        if len(cue) > 72:
            cue = cue[:69].rstrip() + "..."
        return cue
    return None


class ExtractiveSummarizer:
    """Offline summarizer: leading sentences + story-grounded take (a heuristic)."""

    name = "extractive"

    def __init__(self, max_chars: int = 280):
        self.max_chars = max_chars
        self._used_takes: set[str] = set()
        self._used_tails: set[str] = set()

    def reset_takes(self) -> None:
        self._used_takes.clear()
        self._used_tails.clear()

    def summarize(self, item: Item) -> Summary:
        sents = [s for s in sentences(item.text) if len(s) > 25 and not s.lower().startswith(("the post ", "read more"))]
        body = " ".join(sents[:2]) if sents else ""
        if not body:
            if item.source == "Hacker News":
                body = (f"Trending on Hacker News with {item.points} points and "
                        f"{item.comments} comments.")
            else:
                body = f"New from {item.source}: {item.title}."
        take = self._unique_take(item)
        if item.mentions > 1:
            take += f" Covered by {item.mentions} sources, so it is not just noise."
        return Summary(summary=truncate(body, self.max_chars), so_what=take)

    def _unique_take(self, item: Item) -> str:
        """Always name this story; fold in a short body cue when it fits."""
        title = _short_title(item)
        primary = classify(item)
        ordered = [primary] + [c for c, _, v in _TAKES if c != primary] + ["other"]
        tail = None
        for cat in ordered:
            variants = next((v for c, _, v in _TAKES if c == cat), _DEFAULT_TAILS)
            tail = _pick(variants, item.title, self._used_tails)
            if tail:
                self._used_tails.add(tail)
                break
        if not tail:
            tail = _pick(_DEFAULT_TAILS, item.title, self._used_tails) or _DEFAULT_TAILS[0]
            self._used_tails.add(tail)
        cue = _body_cue(item)
        if cue:
            take = f'On "{title}" ({cue}): {tail}.'
        else:
            take = f'On "{title}": {tail}.'
        # Hard cap so thread tweets still have room for the headline + link.
        if len(take) > 200:
            take = f'On "{title}": {tail}.'
            if len(take) > 200:
                take = take[:197].rstrip() + "..."
        if take in self._used_takes:
            take = take[:-1] + f" ({item.source})."
        self._used_takes.add(take)
        return take


# --- LLM backend ---------------------------------------------------------------

_PROMPT = (
    "You write an opinionated daily AI-news digest for builders of AI agents. "
    "Given one news item, reply with ONLY JSON: "
    '{"summary": "<what happened, max 2 sentences, factual>", '
    '"so_what": "<one sharp, specific sentence: why it matters to people building with AI>"}. '
    "Use only facts present in the input. No hype, no em-dashes. "
    "The so_what must be unique to THIS story, not a generic line about agents."
)


class OpenAICompatibleSummarizer:
    name = "llm"

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1",
                 model: str = "gpt-4o-mini", fallback: Optional[Summarizer] = None,
                 timeout: int = 40):
        self.api_key, self.base_url, self.model = api_key, base_url.rstrip("/"), model
        self.timeout = timeout
        self.fallback = fallback or ExtractiveSummarizer()

    @classmethod
    def from_env(cls) -> Optional["OpenAICompatibleSummarizer"]:
        key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            return None
        return cls(key, os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
                   os.environ.get("LLM_MODEL", "gpt-4o-mini"))

    def _chat(self, user: str) -> str:
        body = json.dumps({"model": self.model, "temperature": 0.4, "messages": [
            {"role": "system", "content": _PROMPT}, {"role": "user", "content": user}]}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:  # noqa: S310
            return json.loads(r.read())["choices"][0]["message"]["content"]

    def summarize(self, item: Item) -> Summary:
        try:
            raw = self._chat(f"Source: {item.source}\nTitle: {item.title}\nText: {item.text[:1500]}")
            m = re.search(r"\{.*\}", raw, re.S)
            data = json.loads(m.group(0)) if m else {}
            if data.get("summary") and data.get("so_what"):
                return Summary(summary=str(data["summary"]).strip(), so_what=str(data["so_what"]).strip())
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM summarize failed for %r: %s", item.title[:50], exc)
        return self.fallback.summarize(item)


def get_summarizer(kind: str = "auto") -> Summarizer:
    if kind in ("auto", "llm"):
        llm = OpenAICompatibleSummarizer.from_env()
        if llm:
            return llm
        if kind == "llm":
            raise SystemExit("--summarizer llm needs LLM_API_KEY or OPENAI_API_KEY in the environment")
    return ExtractiveSummarizer()
