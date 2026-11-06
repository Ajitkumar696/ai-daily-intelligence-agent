from src.processing.deduplicate import (
    canonicalize_url,
    deduplicate_articles,
    normalize_title
)


def test_canonicalize_url_normalizes_host_and_removes_fragment():
    assert canonicalize_url(
        "HTTPS://EXAMPLE.COM/News?id=7#section"
    ) == "https://example.com/News?id=7"


def test_canonicalize_url_removes_known_tracking_but_keeps_query():
    assert canonicalize_url(
        "https://example.com/news?id=7&utm_source=mail"
        "&fbclid=tracking&utm_campaign=launch"
    ) == "https://example.com/news?id=7"


def test_canonicalize_url_preserves_unrecognized_query_parameters():
    assert canonicalize_url(
        "https://example.com/news?id=7&ref=homepage"
    ) == "https://example.com/news?id=7&ref=homepage"


def test_normalize_title_handles_case_and_whitespace():
    assert normalize_title("  New   AI MODEL\nReleased ") == (
        "new ai model released"
    )


def test_deduplicate_articles_removes_only_canonical_url_duplicates():
    first = {
        "source": "OpenAI",
        "title": "New AI Model Released",
        "url": "https://EXAMPLE.com/article?utm_source=feed#top"
    }
    duplicate = {
        "source": "OpenAI",
        "title": "Different formatting",
        "url": "https://example.com/article"
    }
    same_title_distinct_url = {
        "source": "Google",
        "title": "new ai model released",
        "url": "https://example.com/another-article"
    }

    assert deduplicate_articles(
        [first, duplicate, same_title_distinct_url]
    ) == [first, same_title_distinct_url]


def test_deduplicate_articles_preserves_distinct_meaningful_queries():
    first = {
        "title": "Same headline",
        "url": "https://example.com/article?id=1"
    }
    second = {
        "title": "Same headline",
        "url": "https://example.com/article?id=2"
    }

    assert deduplicate_articles([first, second]) == [first, second]


def test_deduplicate_articles_skips_missing_url_or_title():
    articles = [
        {"title": "Missing URL"},
        {"url": "https://example.com/missing-title"},
    ]

    assert deduplicate_articles(articles) == []
