import pytest

import src.collect as collect
from src.collectors.collection_result import CollectionResult
from src.configuration import load_sources_config, SourceConfig


# ── Helpers ──────────────────────────────────────────────────────────────────

def write_config(tmp_path, contents):
    config_file = tmp_path / "sources.yaml"
    config_file.write_text(contents, encoding="utf-8")
    return config_file


# ── Live sources.yaml integration tests ──────────────────────────────────────

EXPECTED_SOURCES = [
    # name, type, url, max_articles, content_type
    ("OpenAI", "rss", "https://openai.com/news/rss.xml", 20, "article"),
    ("Google DeepMind", "rss", "https://deepmind.google/blog/rss.xml", 20, "article"),
    ("Anthropic News", "sitemap", "https://www.anthropic.com/sitemap.xml", 20, "article"),
    ("Hugging Face Papers", "api", "https://huggingface.co/api/daily_papers", 20, "paper"),
    ("Hugging Face Blog", "rss", "https://huggingface.co/blog/feed.xml", 20, "article"),
    ("Inc42", "rss", "https://inc42.com/feed/", 20, "article"),
    ("Latent Space", "rss", "https://podcast.latent.space/feed.xml", 20, "podcast"),
    ("TLDR AI", "rss", "https://tldr.tech/api/rss/ai", 20, "newsletter"),
    ("Lenny's Newsletter", "rss", "https://lenny.substack.com/feed", 10, "newsletter"),
    ("Interconnects AI", "rss", "https://www.interconnects.ai/feed", 10, "newsletter"),
    ("AI Engineer", "rss", "https://www.youtube.com/feeds/videos.xml?channel_id=UCLKPca3kwwd-B59HNr-_lvA", 10, "video"),
    ("Hugging Face Spaces", "api_hf_spaces", "https://huggingface.co/api/spaces", 20, "space"),
    ("X / Twitter", "twitter", "https://api.twitter.com/2/tweets/search/recent", 10, "social"),
    ("Product Hunt", "producthunt", "https://api.producthunt.com/v2/api/graphql", 20, "launch"),
]


def test_loads_all_sources_and_article_limits():
    """Live sources.yaml must contain exactly the expected sources in order."""
    sources = load_sources_config()

    actual = [
        (s.name, s.type, s.url, s.max_articles, s.content_type)
        for s in sources
    ]

    assert actual == EXPECTED_SOURCES


def test_article_extraction_enabled_sources():
    """Verify that OpenAI, DeepMind, Anthropic, and Inc42 have article extraction enabled."""
    sources = load_sources_config()
    sources_with_extraction = {
        s.name for s in sources if s.extract_articles
    }
    assert sources_with_extraction == {
        "OpenAI",
        "Google DeepMind",
        "Anthropic News",
        "Inc42",
    }


def test_only_rss_and_sitemap_sources_have_extraction():
    sources = load_sources_config()
    for source in sources:
        if source.extract_articles:
            assert source.type in ("rss", "sitemap"), (
                f"{source.name} has extract_articles=True but type={source.type!r}"
            )


def test_content_types_are_classified():
    """Every source must have a valid non-empty content_type."""
    sources = load_sources_config()
    valid = {
        "article",
        "newsletter",
        "podcast",
        "paper",
        "space",
        "video",
        "social",
        "launch",
    }
    for source in sources:
        assert source.content_type in valid, (
            f"{source.name} has unexpected content_type={source.content_type!r}"
        )


def test_source_config_defaults():
    """SourceConfig defaults must be sane when not provided in YAML."""
    cfg = SourceConfig(
        name="Test",
        type="rss",
        url="https://example.com/feed.xml",
        max_articles=5,
    )
    assert cfg.extract_articles is False
    assert cfg.content_type == "article"


# ── Error-path tests (use temporary YAML files) ───────────────────────────────

def test_missing_configuration_file_has_clear_error(tmp_path):
    missing_file = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="not found"):
        load_sources_config(missing_file)


def test_malformed_yaml_has_clear_error(tmp_path):
    config_file = write_config(tmp_path, "sources: [")

    with pytest.raises(ValueError, match="Malformed YAML"):
        load_sources_config(config_file)


def test_missing_required_fields_have_clear_error(tmp_path):
    config_file = write_config(
        tmp_path,
        "sources:\n  - name: OpenAI\n    type: rss\n"
    )

    with pytest.raises(ValueError, match="missing required fields:"):
        load_sources_config(config_file)


def test_unsupported_source_type_has_clear_error(tmp_path):
    config_file = write_config(
        tmp_path,
        "sources:\n"
        "  - name: Example\n"
        "    type: scraper\n"
        "    url: https://example.com\n"
        "    max_articles: 5\n"
    )

    with pytest.raises(ValueError, match="Unsupported source type 'scraper'"):
        load_sources_config(config_file)


def test_non_string_source_type_has_clear_error(tmp_path):
    config_file = write_config(
        tmp_path,
        "sources:\n"
        "  - name: Example\n"
        "    type: [rss]\n"
        "    url: https://example.com\n"
        "    max_articles: 5\n"
    )

    with pytest.raises(ValueError, match="Unsupported source type"):
        load_sources_config(config_file)


def test_invalid_content_type_has_clear_error(tmp_path):
    config_file = write_config(
        tmp_path,
        "sources:\n"
        "  - name: Example\n"
        "    type: rss\n"
        "    url: https://example.com/feed\n"
        "    max_articles: 5\n"
        "    content_type: unknown_type\n"
    )

    with pytest.raises(ValueError, match="unsupported.*content_type"):
        load_sources_config(config_file)


# ── Dispatch test ─────────────────────────────────────────────────────────────

def test_collection_dispatch_uses_configured_urls_and_limits(monkeypatch):
    """collect_all_sources must call the right collector for each source type."""
    calls = []

    def fake_rss(source, feed_url, max_articles):
        calls.append(("rss", source, feed_url, max_articles))
        return CollectionResult.from_articles(source, [])

    def fake_huggingface(max_articles, api_url):
        calls.append(("api", api_url, max_articles))
        return CollectionResult.from_articles("Hugging Face Papers", [])

    def fake_anthropic(max_articles, sitemap_url):
        calls.append(("sitemap", sitemap_url, max_articles))
        return CollectionResult.from_articles("Anthropic News", [])

    def fake_spaces(max_articles, api_url):
        calls.append(("api_hf_spaces", api_url, max_articles))
        return CollectionResult.from_articles("Hugging Face Spaces", [])

    def fake_twitter(max_articles, search_url):
        calls.append(("twitter", search_url, max_articles))
        return CollectionResult.from_articles("X / Twitter", [])

    def fake_producthunt(max_articles, api_url):
        calls.append(("producthunt", api_url, max_articles))
        return CollectionResult.from_articles("Product Hunt", [])

    monkeypatch.setattr(collect, "collect_rss", fake_rss)
    monkeypatch.setattr(collect, "collect_huggingface_papers", fake_huggingface)
    monkeypatch.setattr(collect, "collect_anthropic_news", fake_anthropic)
    monkeypatch.setattr(collect, "collect_hf_spaces", fake_spaces)
    monkeypatch.setattr(collect, "collect_twitter", fake_twitter)
    monkeypatch.setattr(collect, "collect_producthunt", fake_producthunt)

    results = collect.collect_all_sources()
    assert len(results) == len(EXPECTED_SOURCES)
    assert all(r.status.value in ("EMPTY", "OK") for r in results)

    # Check each source dispatched with the correct type tag and URL
    rss_calls = [(c[1], c[2], c[3]) for c in calls if c[0] == "rss"]
    expected_rss = [
        (name, url, limit)
        for name, stype, url, limit, _ in EXPECTED_SOURCES
        if stype == "rss"
    ]
    assert rss_calls == expected_rss

    api_calls = [(c[1], c[2]) for c in calls if c[0] == "api"]
    assert api_calls == [
        ("https://huggingface.co/api/daily_papers", 20)
    ]

    sitemap_calls = [(c[1], c[2]) for c in calls if c[0] == "sitemap"]
    assert sitemap_calls == [("https://www.anthropic.com/sitemap.xml", 20)]

    spaces_calls = [(c[1], c[2]) for c in calls if c[0] == "api_hf_spaces"]
    assert spaces_calls == [("https://huggingface.co/api/spaces", 20)]

    twitter_calls = [(c[1], c[2]) for c in calls if c[0] == "twitter"]
    assert twitter_calls == [("https://api.twitter.com/2/tweets/search/recent", 10)]

    ph_calls = [(c[1], c[2]) for c in calls if c[0] == "producthunt"]
    assert ph_calls == [("https://api.producthunt.com/v2/api/graphql", 20)]
