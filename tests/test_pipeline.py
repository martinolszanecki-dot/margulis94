from datetime import datetime, timedelta, timezone

from ai_news_agent import sources
from ai_news_agent.dedupe import dedupe, normalize_url
from ai_news_agent.digest import render_markdown, render_thread
from ai_news_agent.models import Item
from ai_news_agent.rank import rank, score
from ai_news_agent.summarizer import ExtractiveSummarizer, OpenAICompatibleSummarizer, Summarizer
from ai_news_agent.textutil import TWEET_LIMIT, tweet_length

NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)

RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel>
<item><title>Acme launches agent toolkit</title><link>https://acme.dev/a?utm_source=x</link>
<description>&lt;p&gt;Acme released a toolkit for building agents. It supports MCP.&lt;/p&gt;</description>
<pubDate>Fri, 02 Oct 2026 10:00:00 GMT</pubDate></item></channel></rss>"""

ATOM = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry>
<title>Paper on agents</title><link rel="alternate" href="https://arxiv.org/abs/1"/>
<summary>We propose a method.</summary><updated>2026-10-02T09:00:00Z</updated></entry></feed>"""


def item(title, url, source="Blog", **kw):
    return Item(title=title, url=url, source=source, published=kw.pop("published", NOW), **kw)


def test_parse_rss_and_atom():
    r = sources.parse_feed(RSS, "Acme")
    assert r[0].title == "Acme launches agent toolkit"
    assert "<p>" not in r[0].text and r[0].published.tzinfo is not None
    a = sources.parse_feed(ATOM, "arXiv")
    assert a[0].url == "https://arxiv.org/abs/1" and a[0].published.year == 2026


def test_fetch_all_survives_a_failing_source():
    def getter(url):
        if "openai" in url:
            raise OSError("boom")
        if "hn.algolia" in url:
            return b'{"hits": []}'
        return RSS
    items = sources.fetch_all(hours=10_000, getter=getter, now=NOW)
    assert items  # other sources still returned data


def test_normalize_url_strips_tracking():
    assert normalize_url("http://www.x.com/a/?utm_source=t&id=1") == "https://x.com/a?id=1"


def test_dedupe_by_url_and_by_title():
    items = [
        item("Apple tightens Full Disk Access due to AI agents risks", "https://a.com/1", "TechCrunch"),
        item("Apple limits Mac Full Disk Access as AI agents increase risk", "https://b.com/2", "Verge"),
        item("Totally different story about GPUs", "https://c.com/3"),
        item("Totally different story about GPUs", "https://c.com/3?utm_source=z", "Other"),
    ]
    out = dedupe(items)
    assert len(out) == 2
    assert out[0].mentions == 2 and "Verge" in out[0].also_seen_in


def test_rank_prefers_fresh_engaged_covered_items():
    old = item("Agent thing", "https://x/1", published=NOW - timedelta(hours=40))
    fresh = item("Agent thing two", "https://x/2", points=300, comments=100, mentions=3)
    assert score(fresh, NOW) > score(old, NOW)
    assert rank([old, fresh], NOW)[0] is fresh


def test_rank_caps_per_source():
    many = [item(f"Agent story {i}", f"https://x/{i}", "Same") for i in range(10)]
    assert len(rank(many, NOW, per_source_cap=3)) == 3


def test_extractive_summarizer_implements_protocol():
    s = ExtractiveSummarizer()
    assert isinstance(s, Summarizer)
    it = item("Acme launches agent toolkit", "https://a/1",
              text="Acme released a toolkit for building agents. It supports MCP. Third sentence here.")
    out = s.summarize(it)
    assert out.summary.startswith("Acme released") and out.so_what


def test_llm_summarizer_falls_back_on_error():
    llm = OpenAICompatibleSummarizer("k", base_url="http://127.0.0.1:9", timeout=1)
    out = llm.summarize(item("Agents ship", "https://a/1", text="Some long enough description text here."))
    assert out.so_what  # fell back to extractive


def test_thread_hook_first_and_all_tweets_fit():
    s = ExtractiveSummarizer()
    long_title = "A " + "very long headline about agents " * 20
    entries = [(item(long_title, f"https://example.com/{'p' * 150}{i}"), None) for i in range(6)]
    entries = [(it, s.summarize(it)) for it, _ in entries]
    tweets = render_thread(NOW.date(), entries, repo_url="https://example.com/repo")
    assert tweets[0].startswith("AI agents + AI news")
    assert all(tweet_length(t) <= TWEET_LIMIT for t in tweets)
    assert len(tweets) == 5 + 2


def test_markdown_contains_so_what():
    it = item("Agents ship", "https://a/1", text="Something happened in the agent world today.")
    md = render_markdown(NOW.date(), [(it, ExtractiveSummarizer().summarize(it))], "extractive")
    assert "**So what:**" in md and "[Agents ship](https://a/1)" in md


def test_extractive_so_whats_unique_in_batch():
    """Agent-heavy titles used to collide on the same canned take."""
    s = ExtractiveSummarizer()
    titles = [
        "Let your AI agents paint big arrows on screen",
        "Show HN: life dashboard with an MCP server for AI agents",
        "Google brings agentic AI to Gemini for businesses",
        "Anthropic bans abusive behavior toward Claude",
        "OpenAI revenue reportedly lower than projected",
        "Fired OpenAI safety researchers warn of chilling effect",
        "OpenAI, the Partition Principle, and Mathematics",
    ]
    takes = []
    for i, title in enumerate(titles):
        out = s.summarize(item(title, f"https://ex/{i}", text=f"Detail about {title}. " * 3))
        base = out.so_what.split(" Covered by")[0]
        takes.append(base)
    assert len(takes) == len(set(takes))


def test_extractive_so_whats_ground_in_story():
    """Takes must name the story, not only recycle category canned lines."""
    s = ExtractiveSummarizer()
    cases = [
        ("Anthropic cuts live internet from internal agent evals",
         "Anthropic turned off live internet access for all internal evaluations after unintended form submissions."),
        ("Show HN: life dashboard with an MCP server for AI agents",
         "A self-hosted dashboard exposes calendar notes through an MCP server for local agents."),
        ("OpenAI revenue reportedly lower than projected",
         "A report said OpenAI revenue trails prior projections by a large margin."),
    ]
    for i, (title, text) in enumerate(cases):
        out = s.summarize(item(title, f"https://ex/g{i}", text=text + " Extra context for builders here."))
        base = out.so_what.split(" Covered by")[0]
        # Title signal must appear (first distinctive token longer than 4 chars).
        token = next(w for w in title.replace(":", " ").split() if len(w) > 4)
        assert token.lower() in base.lower(), (token, base)
        assert not base.startswith("Agents are moving"), base
