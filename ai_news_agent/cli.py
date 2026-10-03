from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from .dedupe import dedupe
from .digest import format_thread, render_markdown, render_thread
from .rank import rank
from .sources import fetch_all
from .summarizer import get_summarizer


def build(hours: int, top: int, summarizer_kind: str, repo_url: str = "", getter=None):
    kwargs = {"getter": getter} if getter else {}
    now = datetime.now(timezone.utc)
    raw = fetch_all(hours=hours, now=now, **kwargs)
    unique = dedupe(raw)
    ranked = rank(unique, now=now)[:top]
    summarizer = get_summarizer(summarizer_kind)
    entries = [(i, summarizer.summarize(i)) for i in ranked]
    day = datetime.now().astimezone().date()
    return {
        "raw": len(raw), "unique": len(unique), "entries": entries, "summarizer": summarizer.name,
        "markdown": render_markdown(day, entries, summarizer.name),
        "thread": render_thread(day, entries, repo_url=repo_url), "day": day,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ai-news-agent", description=__doc__)
    p.add_argument("--hours", type=int, default=36, help="lookback window (default 36)")
    p.add_argument("--top", type=int, default=7, help="items in the digest (default 7)")
    p.add_argument("--summarizer", choices=["auto", "extractive", "llm"], default="auto")
    p.add_argument("--out-dir", default="output", help="where to write files (default ./output)")
    p.add_argument("--repo-url", default="", help="optional link for the closing tweet")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    r = build(a.hours, a.top, a.summarizer, a.repo_url)
    if not r["entries"]:
        print("No items found (network down?).", file=sys.stderr)
        return 1
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = r["day"].isoformat()
    (out / f"digest-{stamp}.md").write_text(r["markdown"], encoding="utf-8")
    (out / f"thread-{stamp}.txt").write_text(format_thread(r["thread"]), encoding="utf-8")
    print(f"Fetched {r['raw']} items, {r['unique']} after dedupe, wrote top {len(r['entries'])} "
          f"using '{r['summarizer']}' summarizer to {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
