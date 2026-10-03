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

# (category, keywords, so_what variants). First matching category wins; a variant is
# picked from the title hash so a digest does not repeat the same sentence.
_TAKES = [
    ("agents", ("agent", "agentic", "mcp", "tool use", "tool-use", "autonomous", "computer use"), (
        "Agents are moving from demos to infrastructure. Reliability, permissions and cost per task now matter more than raw capability.",
        "Whoever controls what agents may touch (files, browsers, payments) will shape this market. Build with least privilege from day one.",
        "The agent stack is still forming. Pick pieces you can swap out, because the winners are not settled.",
    )),
    ("opensource", ("open-source", "open source", "show hn", "github.com", "weights", "apache", "mit license"), (
        "Open releases lower the cost of experimentation and shift leverage from model owners to builders. Read the code before the hype.",
        "Early builder signal: small projects like this are where next year's patterns show up first.",
    )),
    ("release", ("launch", "release", "introduc", "announce", "unveil", "now available", "rolls out", "rolling out"), (
        "A new launch means your default stack may be stale. Re-run your own evals before switching, not the vendor's.",
        "Launch posts are marketing until someone reproduces the numbers. Wait for independent tests.",
    )),
    ("research", ("paper", "arxiv", "we propose", "we present", "benchmark", "study", "researchers"), (
        "Research signal, not product yet. The useful question is whether it survives messy real-world tasks.",
        "Benchmarks move fast and transfer slowly. Treat the headline number as a hypothesis for your own workload.",
    )),
    ("policy", ("regulat", "lawsuit", "court", "senate", "policy", "ban ", "safety", "copyright", "privacy", "risk"), (
        "Rules, lawsuits and platform policies decide what can ship and where. Treat them as roadmap constraints.",
        "Platform owners are starting to set the terms for AI agents. Expect more permission prompts, not fewer.",
    )),
    ("business", ("funding", "raises", "valuation", "acquire", "acquisition", "revenue", "billion", "invest", "partnership"), (
        "Follow the money: capital decides who gets compute, and compute decides who gets to compete.",
    )),
]
_DEFAULT_TAKES = (
    "Worth tracking: it adds to the picture of where AI tooling is heading. The test is whether people still use it a month from now.",
    "A useful data point on how builders and the press are framing AI right now. Judge it by what ships, not what is said.",
)


def classify(item: Item) -> str:
    blob = f"{item.title} {item.text[:300]}".lower()
    for cat, kws, _ in _TAKES:
        if any(k in blob for k in kws):
            return cat
    return "other"


def _pick(variants: tuple, key: str) -> str:
    return variants[sum(map(ord, key)) % len(variants)]


class ExtractiveSummarizer:
    """Offline summarizer: leading sentences + rule-based take (a heuristic)."""

    name = "extractive"

    def __init__(self, max_chars: int = 280):
        self.max_chars = max_chars

    def summarize(self, item: Item) -> Summary:
        sents = [s for s in sentences(item.text) if len(s) > 25 and not s.lower().startswith(("the post ", "read more"))]
        body = " ".join(sents[:2]) if sents else ""
        if not body:
            if item.source == "Hacker News":
                body = (f"Trending on Hacker News with {item.points} points and "
                        f"{item.comments} comments.")
            else:
                body = f"New from {item.source}: {item.title}."
        cat = classify(item)
        variants = next((v for c, _, v in _TAKES if c == cat), _DEFAULT_TAKES)
        take = _pick(variants, item.title)
        if item.mentions > 1:
            take += f" Covered by {item.mentions} sources, so it is not just noise."
        return Summary(summary=truncate(body, self.max_chars), so_what=take)


# --- LLM backend ---------------------------------------------------------------

_PROMPT = (
    "You write an opinionated daily AI-news digest for builders of AI agents. "
    "Given one news item, reply with ONLY JSON: "
    '{"summary": "<what happened, max 2 sentences, factual>", '
    '"so_what": "<one sharp, specific sentence: why it matters to people building with AI>"}. '
    "Use only facts present in the input. No hype, no em-dashes."
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
