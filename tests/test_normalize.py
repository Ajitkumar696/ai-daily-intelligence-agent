from src.processing.normalize import normalize_article


def test_normalize_article_cleans_summary_html():
    entry = {
        "title": " Example AI Update ",
        "link": " https://example.com/ai-update ",
        "published": "Sun, 01 Nov 2026 10:00:00 GMT",
        "summary": "<p>A new <strong>AI</strong> model was released.</p>"
    }

    article = normalize_article("OpenAI", entry)

    assert article["source"] == "OpenAI"
    assert article["title"] == "Example AI Update"
    assert article["url"] == "https://example.com/ai-update"
    assert article["published"] == "Sun, 01 Nov 2026 10:00:00 GMT"
    assert article["summary"] == "A new AI model was released."
    assert article["collected_at"]


def test_normalize_article_skips_missing_title_or_url():
    assert normalize_article("OpenAI", {"link": "https://example.com"}) is None
    assert normalize_article("OpenAI", {"title": "An update"}) is None
