"""
Tests for the new collectors:
  - Anthropic News (sitemap-based)
  - Hugging Face Spaces (API)
  - RSS feeds for TLDR AI, Hugging Face Blog, Inc42, Lenny's Newsletter
  - content_type propagation
  - cross-source deduplication still works with new sources
"""

from unittest.mock import MagicMock, patch

import pytest

from src.collectors.anthropic_collector import (
    _parse_sitemap,
    collect_anthropic_news,
)
from src.collectors.collection_result import CollectionResult, CollectionStatus
from src.collectors.producthunt_collector import collect_producthunt
from src.collectors.spaces_collector import collect_hf_spaces
from src.collectors.twitter_collector import collect_twitter
from src.processing.deduplicate import deduplicate_articles


# ── Helpers ──────────────────────────────────────────────────────────────────

SAMPLE_SITEMAP_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://www.anthropic.com/news/claude-4</loc>
    <lastmod>2026-11-01</lastmod>
  </url>
  <url>
    <loc>https://www.anthropic.com/news/model-spec</loc>
    <lastmod>2026-10-15</lastmod>
  </url>
  <url>
    <loc>https://www.anthropic.com/research/scaling</loc>
    <lastmod>2026-10-20</lastmod>
  </url>
</urlset>
"""

SAMPLE_SPACES_JSON = [
    {
        "id": "stabilityai/stable-diffusion",
        "likes": 5000,
        "trendingScore": 99.5,
        "sdk": "gradio",
        "tags": ["image-generation", "diffusion"],
        "createdAt": "2023-01-01T00:00:00.000Z",
        "lastModified": "2026-11-01T12:00:00.000Z",
    },
    {
        "id": "openai/whisper",
        "likes": 3000,
        "trendingScore": 85.0,
        "sdk": "gradio",
        "tags": ["asr", "speech"],
        "createdAt": "2023-03-01T00:00:00.000Z",
        "lastModified": "2026-10-28T08:00:00.000Z",
    },
]


# ── Anthropic: sitemap parsing ────────────────────────────────────────────────

class TestAnthropicSitemapParsing:
    def test_filters_only_news_urls(self):
        entries = _parse_sitemap(SAMPLE_SITEMAP_XML)
        urls = [e[0] for e in entries]
        assert all("/news/" in u for u in urls)
        assert "https://www.anthropic.com/research/scaling" not in urls

    def test_sorted_newest_first(self):
        entries = _parse_sitemap(SAMPLE_SITEMAP_XML)
        dates = [e[1] for e in entries]
        assert dates == sorted(dates, reverse=True)

    def test_returns_url_and_date_tuples(self):
        entries = _parse_sitemap(SAMPLE_SITEMAP_XML)
        assert len(entries) == 2
        url, date = entries[0]
        assert url.startswith("https://")
        assert date == "2026-11-01"

    def test_empty_sitemap_returns_empty_list(self):
        empty_xml = '<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"></urlset>'
        entries = _parse_sitemap(empty_xml)
        assert entries == []

    def test_sitemap_without_lastmod(self):
        xml = """\
<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
<url><loc>https://www.anthropic.com/news/no-date</loc></url>
</urlset>"""
        entries = _parse_sitemap(xml)
        assert len(entries) == 1
        assert entries[0][1] == ""  # empty lastmod


# ── Anthropic: collector integration ─────────────────────────────────────────

class TestAnthropicCollector:
    def _make_sitemap_response(self):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.text = SAMPLE_SITEMAP_XML
        return resp

    def test_returns_collection_result(self):
        with (
            patch("src.collectors.anthropic_collector.requests.get") as mock_get,
            patch("src.collectors.anthropic_collector.extract_article_content") as mock_extract,
        ):
            mock_get.return_value = self._make_sitemap_response()
            mock_extract.return_value = None  # stub — no body extracted
            result = collect_anthropic_news(max_articles=5)

        assert isinstance(result, CollectionResult)

    def test_articles_have_required_fields(self):
        with (
            patch("src.collectors.anthropic_collector.requests.get") as mock_get,
            patch("src.collectors.anthropic_collector.extract_article_content") as mock_extract,
        ):
            mock_get.return_value = self._make_sitemap_response()
            mock_extract.return_value = None
            result = collect_anthropic_news(max_articles=5)

        for article in result.articles:
            assert "title" in article
            assert "url" in article
            assert article["source"] == "Anthropic News"
            assert article["publisher"] == "Anthropic"
            assert article["content_type"] == "article"

    def test_max_articles_respected(self):
        with (
            patch("src.collectors.anthropic_collector.requests.get") as mock_get,
            patch("src.collectors.anthropic_collector.extract_article_content") as mock_extract,
        ):
            mock_get.return_value = self._make_sitemap_response()
            mock_extract.return_value = None
            result = collect_anthropic_news(max_articles=1)

        assert len(result.articles) <= 1

    def test_full_extraction_used_when_available(self):
        extracted_data = {
            "title": "Claude 4 Announced",
            "url": "https://www.anthropic.com/news/claude-4",
            "published": "2026-11-01",
            "description": "Anthropic launches Claude 4.",
            "full_text": "Full article text here...",
            "author": "Anthropic Team",
            "image": "",
            "canonical_url": "https://www.anthropic.com/news/claude-4",
        }
        with (
            patch("src.collectors.anthropic_collector.requests.get") as mock_get,
            patch("src.collectors.anthropic_collector.extract_article_content") as mock_extract,
        ):
            mock_get.return_value = self._make_sitemap_response()
            mock_extract.return_value = extracted_data
            result = collect_anthropic_news(max_articles=1)

        assert result.articles[0]["source_type"] == "full_article"
        assert result.articles[0]["content"] == "Full article text here..."

    def test_network_failure_returns_failed_result(self):
        import requests as req

        with patch("src.collectors.anthropic_collector.requests.get") as mock_get:
            mock_get.side_effect = req.exceptions.ConnectionError("timeout")
            result = collect_anthropic_news()

        assert result.status == CollectionStatus.FAILED
        assert "Network" in result.error

    def test_empty_news_sitemap_returns_failed(self):
        empty_xml = '<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"></urlset>'
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.text = empty_xml

        with patch("src.collectors.anthropic_collector.requests.get", return_value=resp):
            result = collect_anthropic_news()

        assert result.status == CollectionStatus.FAILED


# ── Hugging Face Spaces collector ─────────────────────────────────────────────

class TestSpacesCollector:
    def _make_spaces_response(self, data=None):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = data if data is not None else SAMPLE_SPACES_JSON
        return resp

    def test_returns_collection_result(self):
        with patch("src.collectors.spaces_collector.requests.get") as mock_get:
            mock_get.return_value = self._make_spaces_response()
            result = collect_hf_spaces(max_articles=5)

        assert isinstance(result, CollectionResult)

    def test_articles_have_required_fields(self):
        with patch("src.collectors.spaces_collector.requests.get") as mock_get:
            mock_get.return_value = self._make_spaces_response()
            result = collect_hf_spaces(max_articles=5)

        assert len(result.articles) == 2
        for article in result.articles:
            assert "title" in article
            assert "url" in article
            assert article["source"] == "Hugging Face Spaces"
            assert article["source_type"] == "space"
            assert article["content_type"] == "space"
            assert "huggingface.co/spaces/" in article["url"]

    def test_max_articles_param_passed_to_api(self):
        with patch("src.collectors.spaces_collector.requests.get") as mock_get:
            mock_get.return_value = self._make_spaces_response()
            collect_hf_spaces(max_articles=7)

        _, kwargs = mock_get.call_args
        assert kwargs["params"]["limit"] == 7

    def test_metadata_includes_sdk_and_tags(self):
        with patch("src.collectors.spaces_collector.requests.get") as mock_get:
            mock_get.return_value = self._make_spaces_response()
            result = collect_hf_spaces()

        article = result.articles[0]
        assert article["metadata"]["sdk"] == "gradio"
        assert "image-generation" in article["metadata"]["tags"]

    def test_non_list_response_returns_failed(self):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"error": "not a list"}

        with patch("src.collectors.spaces_collector.requests.get", return_value=resp):
            result = collect_hf_spaces()

        assert result.status == CollectionStatus.FAILED

    def test_network_failure_returns_failed(self):
        import requests as req

        with patch("src.collectors.spaces_collector.requests.get") as mock_get:
            mock_get.side_effect = req.exceptions.Timeout("timeout")
            result = collect_hf_spaces()

        assert result.status == CollectionStatus.FAILED

    def test_items_without_id_are_skipped(self):
        data = [{"likes": 10}, {"id": "owner/space", "likes": 5}]
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = data

        with patch("src.collectors.spaces_collector.requests.get", return_value=resp):
            result = collect_hf_spaces()

        # Only the second item with an id should be collected
        assert len(result.articles) == 1
        assert "owner/space" in result.articles[0]["url"]


# ── RSS collector — new sources use existing rss_collector ───────────────────

class TestNewRssSources:
    """
    TLDR AI, HF Blog, Inc42, and Lenny all use the existing collect_rss()
    function.  We only need to verify that the source YAML entries are valid
    and that the collector can be called with their URLs.
    """

    RSS_SOURCES = [
        ("TLDR AI", "https://tldr.tech/api/rss/ai", "newsletter"),
        ("Hugging Face Blog", "https://huggingface.co/blog/feed.xml", "article"),
        ("Inc42", "https://inc42.com/feed/", "article"),
        ("Lenny's Newsletter", "https://lenny.substack.com/feed", "newsletter"),
    ]

    @pytest.mark.parametrize("name,url,content_type", RSS_SOURCES)
    def test_rss_source_config_loaded(self, name, url, content_type):
        from src.configuration import load_sources_config
        sources = load_sources_config()
        source = next((s for s in sources if s.name == name), None)
        assert source is not None, f"Source '{name}' not found in sources.yaml"
        assert source.url == url
        assert source.type == "rss"
        assert source.content_type == content_type


# ── content_type propagation ──────────────────────────────────────────────────

class TestContentTypePropagation:
    def test_anthropic_articles_have_content_type_article(self):
        with (
            patch("src.collectors.anthropic_collector.requests.get") as mock_get,
            patch("src.collectors.anthropic_collector.extract_article_content") as mock_extract,
        ):
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            resp.text = SAMPLE_SITEMAP_XML
            mock_get.return_value = resp
            mock_extract.return_value = None
            result = collect_anthropic_news(max_articles=2)

        for article in result.articles:
            assert article.get("content_type") == "article"

    def test_spaces_articles_have_content_type_space(self):
        with patch("src.collectors.spaces_collector.requests.get") as mock_get:
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            resp.json.return_value = SAMPLE_SPACES_JSON
            mock_get.return_value = resp
            result = collect_hf_spaces(max_articles=2)

        for article in result.articles:
            assert article.get("content_type") == "space"


# ── Cross-source deduplication ────────────────────────────────────────────────

class TestCrossSourceDeduplication:
    def test_identical_urls_across_sources_are_deduplicated(self):
        """The same story URL from two different sources → deduplicated to one."""
        article_a = {
            "source": "OpenAI",
            "title": "GPT-5 Released",
            "url": "https://openai.com/blog/gpt-5",
            "published": "2026-11-01",
            "summary": "OpenAI releases GPT-5.",
        }
        article_b = {
            "source": "Hugging Face Blog",
            "title": "GPT-5 Released",
            "url": "https://openai.com/blog/gpt-5",  # same URL
            "published": "2026-11-01",
            "summary": "GPT-5 is now available.",
        }
        result = deduplicate_articles([article_a, article_b])
        assert len(result) == 1

    def test_different_urls_are_not_deduplicated(self):
        article_a = {
            "source": "OpenAI",
            "title": "GPT-5",
            "url": "https://openai.com/blog/gpt-5",
            "published": "2026-11-01",
        }
        article_b = {
            "source": "Anthropic News",
            "title": "Claude 4",
            "url": "https://www.anthropic.com/news/claude-4",
            "published": "2026-11-01",
        }
        result = deduplicate_articles([article_a, article_b])
        assert len(result) == 2

    def test_space_entries_not_deduplicated_against_articles(self):
        """Space entries have unique URLs and should never be deduplicated away."""
        article = {
            "source": "OpenAI",
            "title": "Whisper",
            "url": "https://openai.com/research/whisper",
            "published": "2026-11-01",
        }
        space = {
            "source": "Hugging Face Spaces",
            "title": "Whisper (openai/whisper)",
            "url": "https://huggingface.co/spaces/openai/whisper",
            "published": "2026-11-01",
        }
        result = deduplicate_articles([article, space])
        assert len(result) == 2


# ── X / Twitter Collector Tests ───────────────────────────────────────────────

class TestTwitterCollector:
    def test_missing_token_returns_failed_with_setup_instructions(self, monkeypatch):
        monkeypatch.delenv("TWITTER_BEARER_TOKEN", raising=False)
        monkeypatch.delenv("X_BEARER_TOKEN", raising=False)
        result = collect_twitter(bearer_token=None)
        assert result.status == CollectionStatus.FAILED
        assert "Missing TWITTER_BEARER_TOKEN" in result.error
        assert "developer.x.com" in result.error

    def test_valid_twitter_response_normalizes_articles(self):
        sample_response = {
            "data": [
                {
                    "id": "1890000000000000001",
                    "text": "Excited to introduce our new open-weights reasoning model for AI agents!",
                    "created_at": "2026-11-04T12:00:00.000Z",
                    "author_id": "user123",
                    "public_metrics": {"retweet_count": 50, "like_count": 200},
                    "lang": "en",
                }
            ]
        }
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = sample_response
        resp.raise_for_status.return_value = None

        with patch("src.collectors.twitter_collector.requests.get", return_value=resp):
            result = collect_twitter(bearer_token="mock_valid_token")

        assert result.status == CollectionStatus.SUCCESS
        assert len(result.articles) == 1
        art = result.articles[0]
        assert art["source"] == "X / Twitter"
        assert art["source_type"] == "social_post"
        assert art["content_type"] == "social"
        assert "1890000000000000001" in art["url"]
        assert art["metadata"]["author_id"] == "user123"

    def test_unauthorized_token_returns_failed_with_clear_message(self):
        resp = MagicMock()
        resp.status_code = 401

        with patch("src.collectors.twitter_collector.requests.get", return_value=resp):
            result = collect_twitter(bearer_token="bad_token")

        assert result.status == CollectionStatus.FAILED
        assert "401" in result.error

    def test_rate_limited_returns_failed_with_clear_message(self):
        resp = MagicMock()
        resp.status_code = 429

        with patch("src.collectors.twitter_collector.requests.get", return_value=resp):
            result = collect_twitter(bearer_token="rate_limited_token")

        assert result.status == CollectionStatus.FAILED
        assert "429" in result.error


# ── Product Hunt Collector Tests ──────────────────────────────────────────────

class TestProductHuntCollector:
    def test_missing_token_returns_failed_with_setup_instructions(self, monkeypatch):
        monkeypatch.delenv("PRODUCT_HUNT_DEVELOPER_TOKEN", raising=False)
        monkeypatch.delenv("PRODUCTHUNT_TOKEN", raising=False)
        result = collect_producthunt(developer_token=None)
        assert result.status == CollectionStatus.FAILED
        assert "Missing PRODUCT_HUNT_DEVELOPER_TOKEN" in result.error
        assert "producthunt.com" in result.error

    def test_valid_producthunt_response_normalizes_articles(self):
        sample_response = {
            "data": {
                "posts": {
                    "edges": [
                        {
                            "node": {
                                "id": "450001",
                                "name": "AgentStudio AI",
                                "tagline": "Build autonomous agents in minutes",
                                "description": "No-code platform to build, evaluate and deploy agentic workflows.",
                                "url": "https://www.producthunt.com/posts/agentstudio-ai",
                                "createdAt": "2026-11-04T08:00:00.000Z",
                                "votesCount": 450,
                                "topics": {
                                    "edges": [
                                        {"node": {"name": "Artificial Intelligence"}},
                                        {"node": {"name": "Developer Tools"}},
                                    ]
                                },
                            }
                        }
                    ]
                }
            }
        }
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = sample_response
        resp.raise_for_status.return_value = None

        with patch("src.collectors.producthunt_collector.requests.post", return_value=resp):
            result = collect_producthunt(developer_token="mock_dev_token")

        assert result.status == CollectionStatus.SUCCESS
        assert len(result.articles) == 1
        launch = result.articles[0]
        assert launch["source"] == "Product Hunt"
        assert launch["source_type"] == "product_launch"
        assert launch["content_type"] == "launch"
        assert "AgentStudio AI: Build autonomous agents in minutes" in launch["title"]
        assert launch["metadata"]["votes_count"] == 450
        assert "Artificial Intelligence" in launch["metadata"]["topics"]

    def test_unauthorized_token_returns_failed_with_clear_message(self):
        resp = MagicMock()
        resp.status_code = 401

        with patch("src.collectors.producthunt_collector.requests.post", return_value=resp):
            result = collect_producthunt(developer_token="invalid_token")

        assert result.status == CollectionStatus.FAILED
        assert "401" in result.error


# ── AI Engineer YouTube Tests ─────────────────────────────────────────────────

class TestAIEngineerVideo:
    def test_ai_engineer_is_configured_with_video_content_type(self):
        from src.configuration import load_sources_config
        sources = load_sources_config()
        ai_eng = next((s for s in sources if s.name == "AI Engineer"), None)
        assert ai_eng is not None
        assert ai_eng.type == "rss"
        assert ai_eng.content_type == "video"
        assert "channel_id=UCLKPca3kwwd-B59HNr-_lvA" in ai_eng.url


