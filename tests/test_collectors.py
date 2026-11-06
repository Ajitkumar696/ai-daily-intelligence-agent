import json

import requests

from src.collectors import article_extractor, huggingface_collector, rss_collector
from src.collectors.collection_result import CollectionStatus


class MockResponse:
    def __init__(
        self,
        *,
        content=b"",
        text="",
        payload=None,
        error=None,
        json_error=None,
        headers=None,
        status_code=200
    ):
        self.content = content
        self.text = text
        self.payload = payload
        self.error = error
        self.json_error = json_error
        self.headers = headers or {}
        self.status_code = status_code

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        if self.json_error:
            raise self.json_error
        return self.payload


VALID_RSS = b"""<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Example AI Feed</title>
    <item>
      <title>Example launch</title>
      <link>https://example.com/launch</link>
      <pubDate>Sun, 01 Nov 2026 10:00:00 GMT</pubDate>
      <guid>urn:example:launch</guid>
      <author>news@example.com</author>
      <category>Research</category>
      <description><![CDATA[<p>A new AI model.</p>]]></description>
    </item>
  </channel>
</rss>
"""

EMPTY_RSS = b"""<?xml version="1.0"?>
<rss version="2.0"><channel><title>Empty Feed</title></channel></rss>
"""


def test_rss_collector_returns_success_with_normalized_article(monkeypatch):
    monkeypatch.setattr(
        rss_collector.requests,
        "get",
        lambda *args, **kwargs: MockResponse(content=VALID_RSS)
    )

    result = rss_collector.collect_rss(
        "Example",
        "https://example.com/feed.xml",
        max_articles=1
    )

    assert result.status == CollectionStatus.SUCCESS
    assert len(result.articles) == 1
    assert result.articles[0]["source"] == "Example"
    assert result.articles[0]["title"] == "Example launch"
    assert result.articles[0]["url"] == "https://example.com/launch"
    assert result.articles[0]["summary"] == "A new AI model."
    assert result.articles[0]["publisher"] == "Example"
    assert result.articles[0]["author"] == "news@example.com"
    assert result.articles[0]["metadata"]["id"] == "urn:example:launch"
    assert result.articles[0]["metadata"]["tags"][0]["term"] == "Research"


def test_rss_collector_distinguishes_valid_empty_feed(monkeypatch):
    monkeypatch.setattr(
        rss_collector.requests,
        "get",
        lambda *args, **kwargs: MockResponse(content=EMPTY_RSS)
    )

    result = rss_collector.collect_rss("Example", "https://example.com/feed")

    assert result.status == CollectionStatus.EMPTY
    assert result.articles == []
    assert result.error is None


def test_rss_collector_reports_network_timeout(monkeypatch):
    def timeout(*args, **kwargs):
        raise requests.Timeout("feed request timed out")

    monkeypatch.setattr(rss_collector.requests, "get", timeout)

    result = rss_collector.collect_rss("Example", "https://example.com/feed")

    assert result.status == CollectionStatus.FAILED
    assert "timed out" in result.error


def test_rss_collector_reports_http_error(monkeypatch):
    response = MockResponse(error=requests.HTTPError("503 Service Unavailable"))
    monkeypatch.setattr(
        rss_collector.requests,
        "get",
        lambda *args, **kwargs: response
    )

    result = rss_collector.collect_rss("Example", "https://example.com/feed")

    assert result.status == CollectionStatus.FAILED
    assert "503 Service Unavailable" in result.error


def test_rss_collector_reports_malformed_feed(monkeypatch):
    monkeypatch.setattr(
        rss_collector.requests,
        "get",
        lambda *args, **kwargs: MockResponse(content=b"<rss><broken")
    )

    result = rss_collector.collect_rss("Example", "https://example.com/feed")

    assert result.status == CollectionStatus.FAILED
    assert "Unable to parse RSS feed" in result.error


def test_huggingface_collector_uses_configured_api_and_returns_articles(
    monkeypatch
):
    payload = [
        {
            "title": "Paper title",
            "summary": "Paper summary",
            "publishedAt": "2026-11-01T10:00:00Z",
            "upvotes": 42,
            "paper": {
                "id": "abc123",
                "authors": [{"name": "Example Author"}]
            }
        }
    ]
    calls = []

    def mock_get(url, **kwargs):
        calls.append((url, kwargs))
        return MockResponse(payload=payload)

    monkeypatch.setattr(huggingface_collector.requests, "get", mock_get)
    api_url = "https://example.com/papers"

    result = huggingface_collector.collect_huggingface_papers(
        max_articles=5,
        api_url=api_url
    )

    assert result.status == CollectionStatus.SUCCESS
    assert len(result.articles) == 1
    assert result.articles[0]["title"] == "Paper title"
    assert result.articles[0]["url"] == "https://huggingface.co/papers/abc123"
    assert result.articles[0]["summary"] == "Paper summary"
    assert result.articles[0]["authors"] == ["Example Author"]
    assert result.articles[0]["metadata"]["paper_id"] == "abc123"
    assert result.articles[0]["metadata"]["upvotes"] == 42
    assert calls[0][0] == api_url
    assert calls[0][1]["params"] == {"limit": 5}


def test_huggingface_collector_distinguishes_empty_response(monkeypatch):
    monkeypatch.setattr(
        huggingface_collector.requests,
        "get",
        lambda *args, **kwargs: MockResponse(payload=[])
    )

    result = huggingface_collector.collect_huggingface_papers()

    assert result.status == CollectionStatus.EMPTY
    assert result.articles == []


def test_huggingface_collector_reports_invalid_json(monkeypatch):
    monkeypatch.setattr(
        huggingface_collector.requests,
        "get",
        lambda *args, **kwargs: MockResponse(
            json_error=ValueError("not valid JSON")
        )
    )

    result = huggingface_collector.collect_huggingface_papers()

    assert result.status == CollectionStatus.FAILED
    assert "Invalid API data" in result.error
    assert "not valid JSON" in result.error


def test_huggingface_collector_reports_http_error(monkeypatch):
    response = MockResponse(error=requests.HTTPError("429 Too Many Requests"))
    monkeypatch.setattr(
        huggingface_collector.requests,
        "get",
        lambda *args, **kwargs: response
    )

    result = huggingface_collector.collect_huggingface_papers()

    assert result.status == CollectionStatus.FAILED
    assert "429 Too Many Requests" in result.error


def test_huggingface_collector_rejects_unexpected_response_shape(monkeypatch):
    monkeypatch.setattr(
        huggingface_collector.requests,
        "get",
        lambda *args, **kwargs: MockResponse(payload={"papers": []})
    )

    result = huggingface_collector.collect_huggingface_papers()

    assert result.status == CollectionStatus.FAILED
    assert "must be a list" in result.error


def test_article_extractor_obeys_robots_and_extracts_mocked_html(monkeypatch):
    article_data = {
        "@type": "NewsArticle",
        "headline": "Example article",
        "description": "Article description",
        "author": {"name": "Example Author"},
        "datePublished": "2026-11-01",
        "publisher": {"name": "Example Publisher"}
    }
    html = (
        '<html><head><title>Page title</title>'
        '<link rel="canonical" href="/canonical-article">'
        '<script type="application/ld+json">'
        f"{json.dumps(article_data)}"
        "</script></head><body><main>"
        "<nav>Navigation must be omitted.</nav>"
        "<p>First paragraph.</p>"
        '<div class="advertisement"><p>Advertisement must be omitted.</p></div>'
        "<p>Second paragraph.</p>"
        "</main></body></html>"
    )
    calls = []

    def mock_get(url, **kwargs):
        calls.append(url)
        if url.endswith("/robots.txt"):
            return MockResponse(
                text="User-agent: *\nAllow: /\n"
            )
        return MockResponse(text=html)

    monkeypatch.setattr(article_extractor.requests, "get", mock_get)
    url = "https://example.com/article"

    article = article_extractor.extract_article_content(url)

    assert calls == ["https://example.com/robots.txt", url]
    assert article["page_title"] == "Page title"
    assert article["title"] == "Example article"
    assert article["description"] == "Article description"
    assert article["author"] == "Example Author"
    assert article["publisher"] == "Example Publisher"
    assert article["published"] == "2026-11-01"
    assert article["canonical_url"] == "https://example.com/canonical-article"
    assert article["url"] == article["canonical_url"]
    assert article["full_text"] == "First paragraph.\n\nSecond paragraph."


def test_article_extractor_skips_disallowed_url(monkeypatch):
    calls = []

    def mock_get(url, **kwargs):
        calls.append(url)
        return MockResponse(
            text="User-agent: *\nDisallow: /private\n"
        )

    monkeypatch.setattr(article_extractor.requests, "get", mock_get)

    article = article_extractor.extract_article_content(
        "https://example.com/private/article"
    )

    assert article is None
    assert calls == ["https://example.com/robots.txt"]


def test_article_extractor_falls_back_when_page_has_no_rendered_article(
    monkeypatch
):
    def mock_get(url, **kwargs):
        if url.endswith("/robots.txt"):
            return MockResponse(text="User-agent: *\nAllow: /\n")
        return MockResponse(text="<html><body><div>JavaScript required</div></body></html>")

    monkeypatch.setattr(article_extractor.requests, "get", mock_get)

    assert article_extractor.extract_article_content(
        "https://example.com/javascript-article"
    ) is None


def test_article_extractor_does_not_fetch_non_html_responses(monkeypatch):
    calls = []

    def mock_get(url, **kwargs):
        calls.append(url)
        if url.endswith("/robots.txt"):
            return MockResponse(text="User-agent: *\nAllow: /\n")
        return MockResponse(
            text="not html",
            headers={"Content-Type": "application/pdf"}
        )

    monkeypatch.setattr(article_extractor.requests, "get", mock_get)

    assert article_extractor.extract_article_content(
        "https://example.com/document"
    ) is None
    assert calls == [
        "https://example.com/robots.txt",
        "https://example.com/document"
    ]


def test_article_extractor_checks_robots_after_cross_site_redirect(monkeypatch):
    calls = []

    def mock_get(url, **kwargs):
        calls.append(url)
        if url == "https://first.example/robots.txt":
            return MockResponse(text="User-agent: *\nAllow: /\n")
        if url == "https://first.example/article":
            return MockResponse(
                status_code=302,
                headers={"Location": "https://second.example/private"}
            )
        if url == "https://second.example/robots.txt":
            return MockResponse(text="User-agent: *\nDisallow: /private\n")
        return MockResponse(text="<html><body><article>Should not fetch</article></body></html>")

    monkeypatch.setattr(article_extractor.requests, "get", mock_get)

    result = article_extractor.extract_article_content(
        "https://first.example/article"
    )

    assert result is None
    assert calls == [
        "https://first.example/robots.txt",
        "https://first.example/article",
        "https://second.example/robots.txt"
    ]


def test_article_extractor_waits_between_requests_to_same_origin(monkeypatch):
    sleeps = []
    clock_values = iter((10, 10, 10))
    monkeypatch.setattr(article_extractor.time, "monotonic", lambda: next(clock_values))
    monkeypatch.setattr(article_extractor.time, "sleep", sleeps.append)
    monkeypatch.setattr(
        article_extractor,
        "MIN_REQUEST_INTERVAL_SECONDS",
        1
    )
    monkeypatch.setattr(
        article_extractor.requests,
        "get",
        lambda *args, **kwargs: MockResponse()
    )

    article_extractor._rate_limited_get("https://example.com/one")
    article_extractor._rate_limited_get("https://example.com/two")

    assert sleeps == [1]
