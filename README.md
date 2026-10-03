# ai-news-agent

A small, dependency-free Python agent that reads the day's AI news from public sources, removes duplicates, ranks what matters for people building with AI agents, and writes:

1. an opinionated **markdown digest** (every item has a "So what"), and
2. a **tweet-thread version** (hook first, every tweet <= 280 chars, links counted as 23 chars like X does).

No paid keys needed. With no LLM key it uses a simple extractive summarizer; add any OpenAI-compatible key and the same pipeline produces sharper takes.

## Quick start

```bash
git clone <this repo> && cd ai-news-agent
python3 -m ai_news_agent -v                 # writes ./output/digest-YYYY-MM-DD.md and thread-YYYY-MM-DD.txt
python3 -m ai_news_agent --top 10 --hours 48 --out-dir out
python3 -m pytest -q                         # tests (pip install -r requirements.txt)
```

Python 3.9+ (developed on 3.13). Runtime uses only the standard library.

### Optional: LLM summaries

Any OpenAI-compatible `/chat/completions` endpoint works (OpenAI, xAI, Groq, Together, local servers):

```bash
export LLM_API_KEY=...                       # or OPENAI_API_KEY
export LLM_BASE_URL=https://api.openai.com/v1   # default
export LLM_MODEL=gpt-4o-mini                 # default
python3 -m ai_news_agent --summarizer llm
```

`--summarizer auto` (default) uses the LLM if a key is set, otherwise the extractive fallback. If an LLM call fails for one item, that item falls back to extractive, so a run never dies on a flaky API.

## Architecture

```
sources.py   fetch RSS/Atom feeds + Hacker News (Algolia API) + arXiv, in parallel
   |          (one failing source never aborts the run; gzip + retry handled)
dedupe.py    normalize URLs, then fuzzy-match titles; merged items count "mentions"
   |
rank.py      score = (1 + topic boost + engagement + cross-source coverage)
   |                  x source weight x recency decay; per-source cap of 3
summarizer.py  Summarizer protocol -> Summary(summary, so_what)
   |            ExtractiveSummarizer (offline) | OpenAICompatibleSummarizer
digest.py    markdown + tweet thread rendering (280-char fitting)
cli.py       argparse entry point (python -m ai_news_agent)
```

### Sources (10)

OpenAI, Google DeepMind, Hugging Face, Simon Willison, TechCrunch AI, The Verge AI, MIT Technology Review AI, Ars Technica AI (RSS/Atom), Hacker News (AI-related stories with >30 points), arXiv (cs.AI/cs.CL/cs.LG papers with "agent" in the title). Edit `RSS_FEEDS` in `sources.py` to add your own.

### Writing your own summarizer

```python
from ai_news_agent.models import Item, Summary

class MySummarizer:
    name = "mine"
    def summarize(self, item: Item) -> Summary:
        return Summary(summary="...", so_what="...")
```

Pass it to the pipeline in `cli.build` (or extend `get_summarizer`).

## Honest limitations

- The extractive "So what" is a rule-based heuristic keyed on topic words. It is a placeholder for judgment, not judgment. Use the LLM backend or edit by hand before publishing.
- Ranking weights in `rank.py` are editorial guesses, not tuned on data.
- Title-based dedupe can occasionally miss rewordings or merge distinct stories.
- Feeds change; if a source breaks, the log (`-v`) shows which one.

## Sample output

See [`sample/`](sample/) for a real run (2026-10-03).

## License

MIT
