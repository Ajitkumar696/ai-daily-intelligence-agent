import asyncio
from types import SimpleNamespace

import pytest

from src.collectors import article_extractor
from src.collectors.crawl4ai_collector import Crawl4AIAdapter
from src.collectors.collection_result import CollectionStatus


ARTICLE_URL = "https://deepmind.google/blog/example"
HTML = """<html><head><title>Page title</title>
<meta name="description" content="A description">
<link rel="canonical" href="/blog/canonical">
</head><body><main><h2>Research</h2><p>Article body text.</p></main></body></html>"""


class FakeCrawler:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.urls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def arun(self, url, config):
        self.urls.append((url, config))
        if self.error:
            raise self.error
        return self.response


class FakeConfig:
    def __init__(self, **kwargs):
        self.options = kwargs


def fake_factory(crawler):
    return lambda: crawler, FakeConfig


@pytest.fixture(autouse=True)
def allow_robots(monkeypatch):
    monkeypatch.setattr(
        article_extractor,
        "_is_allowed_by_robots",
        lambda _url: (True, None),
    )
    monkeypatch.setattr("src.collectors.crawl4ai_collector.asyncio.sleep", lambda *_: asyncio.sleep(0))


def test_crawl_extracts_metadata_and_returns_collection_result():
    crawler = FakeCrawler(SimpleNamespace(
        success=True, url=ARTICLE_URL, cleaned_html=HTML, html="", error_message=""
    ))
    result = Crawl4AIAdapter(fake_factory(crawler)).collect(
        "Google DeepMind", [ARTICLE_URL], ["deepmind.google"]
    )

    assert result.status == CollectionStatus.SUCCESS
    assert len(result.articles) == 1
    article = result.articles[0]
    assert article["url"] == ARTICLE_URL
    assert article["extracted_content"]["canonical_url"] == (
        "https://deepmind.google/blog/canonical"
    )
    assert article["extracted_content"]["title"] == "Page title"
    assert article["extracted_content"]["published"] == ""
    assert article["extracted_content"]["full_text"] == "Research\n\nArticle body text."
    assert crawler.urls[0][1].options["check_robots_txt"] is True


def test_duplicate_urls_are_crawled_only_once():
    crawler = FakeCrawler(SimpleNamespace(
        success=True, url=ARTICLE_URL, cleaned_html=HTML, html="", error_message=""
    ))
    result = Crawl4AIAdapter(fake_factory(crawler)).collect(
        "Google DeepMind", [ARTICLE_URL, ARTICLE_URL], ["deepmind.google"]
    )
    assert len(crawler.urls) == 1
    assert len(result.articles) == 1


def test_missing_optional_dependency_returns_failed_result(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def without_crawl4ai(name, *args, **kwargs):
        if name == "crawl4ai":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_crawl4ai)
    result = Crawl4AIAdapter().collect(
        "Google DeepMind", [ARTICLE_URL], ["deepmind.google"]
    )
    assert result.status == CollectionStatus.FAILED
    assert "optional requirements-crawl4ai.txt" in result.error


@pytest.mark.parametrize("response", [
    SimpleNamespace(success=True, url=ARTICLE_URL, cleaned_html="", html="", error_message=""),
    SimpleNamespace(success=False, url=ARTICLE_URL, cleaned_html="", html="", error_message="network error"),
])
def test_empty_or_failed_crawl_is_a_safe_empty_result(response):
    crawler = FakeCrawler(response)
    result = Crawl4AIAdapter(fake_factory(crawler)).collect(
        "Google DeepMind", [ARTICLE_URL], ["deepmind.google"]
    )
    assert result.status == CollectionStatus.EMPTY
    assert result.articles == []
    assert result.warnings


def test_timeout_is_reported_without_retrying():
    crawler = FakeCrawler(error=asyncio.TimeoutError("timed out"))
    result = Crawl4AIAdapter(fake_factory(crawler)).collect(
        "Google DeepMind", [ARTICLE_URL], ["deepmind.google"]
    )
    assert result.status == CollectionStatus.EMPTY
    assert len(crawler.urls) == 1
    assert "timed out" in result.warnings[0]


def test_disallowed_domain_is_rejected_before_browser_launch():
    launched = []
    result = Crawl4AIAdapter(lambda: launched.append(True)).collect(
        "Google DeepMind", ["https://outside.example/article"], ["deepmind.google"]
    )
    assert result.status == CollectionStatus.FAILED
    assert launched == []


def test_cross_domain_redirect_and_canonical_are_not_used():
    crawler = FakeCrawler(SimpleNamespace(
        success=True,
        url="https://outside.example/article",
        cleaned_html=HTML,
        html="",
        error_message="",
    ))
    result = Crawl4AIAdapter(fake_factory(crawler)).collect(
        "Google DeepMind", [ARTICLE_URL], ["deepmind.google"]
    )
    assert result.status == CollectionStatus.EMPTY
    assert "cross-domain redirect" in result.warnings[0]


def test_robots_denial_skips_browser(monkeypatch):
    crawler = FakeCrawler()
    monkeypatch.setattr(
        article_extractor, "_is_allowed_by_robots", lambda _url: (False, "disallowed")
    )
    result = Crawl4AIAdapter(fake_factory(crawler)).collect(
        "Google DeepMind", [ARTICLE_URL], ["deepmind.google"]
    )
    assert result.status == CollectionStatus.EMPTY
    assert crawler.urls == []
    assert "disallowed" in result.warnings[0]
