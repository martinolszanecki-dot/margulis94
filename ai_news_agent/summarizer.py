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

# (category, keywords, so_what variants). Best-scoring category wins; within a
# digest run we never reuse the same base take, and we ground leftovers in the title.
_TAKES = [
    ("agents", ("agentic", "computer use", "tool-use", "tool use", "mcp", "autonomous", "agents", "agent"), (
        "Agents are moving from demos to infrastructure. Reliability, permissions and cost per task now matter more than raw capability.",
        "Whoever controls what agents may touch (files, browsers, payments) will shape this market. Build with least privilege from day one.",
        "The agent stack is still forming. Pick pieces you can swap out, because the winners are not settled.",
    )),
    ("opensource", ("open-source", "open source", "show hn", "github.com", "weights", "apache", "mit license"), (
        "Open releases lower the cost of experimentation and shift leverage from model owners to builders. Read the code before the hype.",
        "Early builder signal: small projects like this are where next year's patterns show up first.",
    )),
    ("release", ("now available", "rolling out", "rolls out", "unveil", "announce", "introduc", "launch", "release"), (
        "A new launch means your default stack may be stale. Re-run your own evals before switching, not the vendor's.",
        "Launch posts are marketing until someone reproduces the numbers. Wait for independent tests.",
    )),
    ("research", ("arxiv", "we propose", "we present", "benchmark", "paper", "study"), (
        "Research signal, not product yet. The useful question is whether it survives messy real-world tasks.",
        "Benchmarks move fast and transfer slowly. Treat the headline number as a hypothesis for your own workload.",
    )),
    ("policy", ("lawsuit", "copyright", "privacy", "regulat", "senate", "court", "policy", " bans", " ban ", "banned", "safety", "misconduct", "risk"), (
        "Rules, lawsuits and platform policies decide what can ship and where. Treat them as roadmap constraints.",
        "Platform owners are starting to set the terms for AI agents. Expect more permission prompts, not fewer.",
    )),
    ("business", ("valuation", "acquisition", "partnership", "funding", "revenue", "billion", "acquire", "raises", "invest"), (
        "Follow the money: capital decides who gets compute, and compute decides who gets to compete.",
    )),
]
_DEFAULT_TAKES = (
    "Worth tracking: it adds to the picture of where AI tooling is heading. The test is whether people still use it a month from now.",
    "A useful data point on how builders and the press are framing AI right now. Judge it by what ships, not what is said.",
)


def _blob(item: Item) -> str:
    return f"{item.title} {item.text[:300]}".lower()


def classify(item: Item) -> str:
    """Pick the category with the strongest keyword hits (longer phrases count more)."""
    blob = _blob(item)
    best_cat, best_score = "other", 0
    for cat, kws, _ in _TAKES:
        score = 0
        for k in kws:
            if k in blob:
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


def _title_grounded_take(item: Item) -> str:
    """Last resort: one take that clearly names this story, so digests stay unique."""
    title = re.sub(r"\s+", " ", item.title).strip().rstrip(".")
    if len(title) > 90:
        title = title[:87].rstrip() + "..."
    return (
        f"On \"{title}\": judge it by what changes in your stack this month, "
        f"not by the headline alone."
    )


class ExtractiveSummarizer:
    """Offline summarizer: leading sentences + rule-based take (a heuristic)."""

    name = "extractive"

    def __init__(self, max_chars: int = 280):
        self.max_chars = max_chars
        self._used_takes: set[str] = set()

    def reset_takes(self) -> None:
        self._used_takes.clear()

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
        primary = classify(item)
        # Try primary category, then every other category, then defaults, then title.
        ordered = [primary] + [c for c, _, _ in _TAKES if c != primary] + ["other"]
        for cat in ordered:
            variants = next((v for c, _, v in _TAKES if c == cat), _DEFAULT_TAKES)
            picked = _pick(variants, item.title, self._used_takes)
            if picked:
                self._used_takes.add(picked)
                return picked
        grounded = _title_grounded_take(item)
        # If somehow even that collided, add a short disambiguator.
        if grounded in self._used_takes:
            grounded = grounded[:-1] + f" ({item.source})."
        self._used_takes.add(grounded)
        return grounded


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
