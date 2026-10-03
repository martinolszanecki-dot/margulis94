"""Fetchers for public sources. Standard library only, no API keys."""
from __future__ import annotations

import gzip
import json
import logging
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Callable, Optional

from .models import Item
from .textutil import strip_html

log = logging.getLogger("ai_news_agent")

USER_AGENT = "ai-news-agent/0.1 (+open-source daily AI digest)"
TIMEOUT = 20

# (name, url, weight, needs_ai_filter)
# needs_ai_filter: general feeds where we keep only AI-related entries.
RSS_FEEDS = [
    ("OpenAI", "https://openai.com/news/rss.xml", 1.3, False),
    ("Google DeepMind", "https://deepmind.google/blog/rss.xml", 1.2, False),
    ("Hugging Face", "https://huggingface.co/blog/feed.xml", 1.0, False),
    ("Simon Willison", "https://simonwillison.net/atom/everything/", 1.2, True),
    ("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/", 1.0, False),
    ("The Verge AI", "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml", 1.0, False),
    ("MIT Tech Review AI", "https://www.technologyreview.com/topic/artificial-intelligence/feed", 1.0, False),
    ("Ars Technica AI", "https://arstechnica.com/ai/feed/", 1.0, False),
]

HN_QUERIES = ["AI agent", "LLM", "OpenAI", "Anthropic", "Claude", "Gemini", "GPT", "AI model"]

AI_TERMS = (
    " ai ", "a.i.", "llm", "gpt", "openai", "anthropic", "claude", "gemini", "agent",
    "machine learning", "neural", "chatbot", "deepmind", "mistral", "llama", "transformer",
    "language model", "copilot", "diffusion", "artificial intelligence", "mcp", "rag ",
)


def looks_like_ai(title: str, text: str = "") -> bool:
    blob = f" {title} {text[:400]} ".lower()
    return any(t in blob for t in AI_TERMS)


def http_get(url: str, timeout: int = TIMEOUT, retries: int = 2) -> bytes:
    """GET with a small retry; transparently handles gzip (some servers send it unasked)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last: Exception = RuntimeError("unreachable")
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (https only)
                data = resp.read()
            if data[:2] == b"\x1f\x8b":
                data = gzip.decompress(data)
            return data
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise last


def _parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    value = value.strip()
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(el: ET.Element, *names: str) -> str:
    for child in el:
        if _local(child.tag) in names and (child.text or "").strip():
            return child.text.strip()
    return ""


def parse_feed(xml_bytes: bytes, source: str, weight: float = 1.0) -> list[Item]:
    """Parse RSS 2.0 or Atom into Items."""
    root = ET.fromstring(xml_bytes)
    entries = [e for e in root.iter() if _local(e.tag) in ("item", "entry")]
    items: list[Item] = []
    for e in entries:
        title = strip_html(_child_text(e, "title"))
        link = _child_text(e, "link")
        if not link:  # Atom: <link href="..."/>
            for c in e:
                if _local(c.tag) == "link" and c.get("href"):
                    if c.get("rel") in (None, "alternate"):
                        link = c.get("href")
                        break
        desc = _child_text(e, "description", "summary", "content", "encoded")
        published = _parse_date(_child_text(e, "pubDate", "published", "updated", "date"))
        if title and link:
            items.append(Item(title=title, url=link, source=source, published=published,
                              text=strip_html(desc), source_weight=weight))
    return items


def fetch_rss(name: str, url: str, weight: float, ai_filter: bool,
              getter: Callable[[str], bytes] = http_get) -> list[Item]:
    items = parse_feed(getter(url), name, weight)
    if ai_filter:
        items = [i for i in items if looks_like_ai(i.title, i.text)]
    return items


def fetch_hn(since: datetime, min_points: int = 30,
             getter: Callable[[str], bytes] = http_get) -> list[Item]:
    """Hacker News stories via the public Algolia API."""
    seen: dict[str, Item] = {}
    for q in HN_QUERIES:
        params = urllib.parse.urlencode({
            "query": q, "tags": "story", "hitsPerPage": 30,
            "numericFilters": f"created_at_i>{int(since.timestamp())},points>{min_points}",
        })
        data = json.loads(getter(f"https://hn.algolia.com/api/v1/search?{params}"))
        for h in data.get("hits", []):
            title = h.get("title") or ""
            oid = h.get("objectID")
            if not title or oid in seen or not looks_like_ai(title):
                continue
            url = h.get("url") or f"https://news.ycombinator.com/item?id={oid}"
            seen[oid] = Item(
                title=title, url=url, source="Hacker News",
                published=_parse_date(h.get("created_at")),
                text=strip_html(h.get("story_text") or ""),
                points=h.get("points") or 0, comments=h.get("num_comments") or 0,
                source_weight=1.1,
            )
    return list(seen.values())


def fetch_arxiv(max_results: int = 25, getter: Callable[[str], bytes] = http_get) -> list[Item]:
    """Latest arXiv papers on agents (cs.AI / cs.CL / cs.LG)."""
    q = "(cat:cs.AI OR cat:cs.CL OR cat:cs.LG) AND (ti:agent OR ti:agents OR ti:agentic)"
    params = urllib.parse.urlencode({
        "search_query": q, "sortBy": "submittedDate", "sortOrder": "descending",
        "max_results": max_results,
    })
    items = parse_feed(getter(f"https://export.arxiv.org/api/query?{params}"), "arXiv", 0.6)
    for i in items:
        i.title = " ".join(i.title.split())
    return items


def fetch_all(hours: int = 36, getter: Callable[[str], bytes] = http_get,
              now: Optional[datetime] = None) -> list[Item]:
    """Fetch every source in parallel; a failing source never aborts the run."""
    now = now or datetime.now(timezone.utc)
    since = now - timedelta(hours=hours)

    jobs: list[tuple[str, Callable[[], list[Item]]]] = [
        (n, (lambda n=n, u=u, w=w, f=f: fetch_rss(n, u, w, f, getter)))
        for n, u, w, f in RSS_FEEDS
    ]
    jobs.append(("Hacker News", lambda: fetch_hn(since, getter=getter)))
    jobs.append(("arXiv", lambda: fetch_arxiv(getter=getter)))

    def run(job):
        name, fn = job
        try:
            result = fn()
            log.info("%-20s %3d items", name, len(result))
            return result
        except Exception as exc:  # noqa: BLE001 - isolate per-source failures
            log.warning("%-20s FAILED: %s", name, exc)
            return []

    with ThreadPoolExecutor(max_workers=8) as pool:
        batches = list(pool.map(run, jobs))

    items = [i for batch in batches for i in batch]
    # keep only items inside the window (items with no date are kept)
    return [i for i in items if i.published is None or i.published >= since]
