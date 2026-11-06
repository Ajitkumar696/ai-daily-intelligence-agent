from src.research.grouping import group_related_articles


def article(source, title, url, **updates):
    value = {
        "source": source,
        "publisher": source,
        "source_type": "rss",
        "title": title,
        "url": url,
        "published": "",
        "summary": "A collected source description supporting a sufficiently detailed research discussion. " * 2
    }
    value.update(updates)
    return value


def test_related_articles_are_grouped_with_source_comparison():
    articles = [
        article(
            "OpenAI",
            "OpenAI introduces the Orion model",
            "https://openai.example/orion?utm_source=feed",
            published="2026-11-01",
            extracted_content={
                "full_text": "The original report text.",
                "canonical_url": "https://openai.example/orion"
            },
        ),
        article(
            "AI News",
            "Orion model announcement from OpenAI",
            "https://news.example/orion",
            published="2026-11-02"
        ),
        article(
            "Robotics Daily",
            "A new robotics research paper",
            "https://robotics.example/paper"
        )
    ]

    groups = group_related_articles(articles)

    assert len(groups) == 2
    cross_checked = groups[0]
    assert cross_checked["research_scope"] == "multi_source"
    assert cross_checked["source_count"] == 2
    assert len(cross_checked["sources"]) == 2
    assert cross_checked["sources"][0]["content_source"] == "full_article"
    assert cross_checked["sources"][0]["canonical_url"] == (
        "https://openai.example/orion"
    )
    assert cross_checked["sources"][1]["content_source"] == "rss_fallback"
    assert cross_checked["cross_check"]["shared_topic_terms"] == [
        "model",
        "openai",
        "orion"
    ]
    assert cross_checked["cross_check"]["metadata_differences"][0]["field"] == (
        "reported_title"
    )
    assert cross_checked["cross_check"]["metadata_differences"][1]["field"] == (
        "publication_date"
    )
    assert "Not assessed" in cross_checked["cross_check"]["claim_agreement"]
    assert groups[1]["research_scope"] == "single_source"


def test_grouping_deduplicates_urls_and_does_not_force_unrelated_sources():
    articles = [
        article(
            "OpenAI",
            "OpenAI launches Orion model",
            "https://example.com/orion?utm_source=rss"
        ),
        article(
            "OpenAI mirror",
            "OpenAI launches Orion model",
            "https://example.com/orion"
        ),
        article(
            "Research",
            "Researchers evaluate medical image segmentation",
            "https://research.example/medical-segmentation"
        )
    ]

    groups = group_related_articles(articles)

    assert len(groups) == 2
    assert [group["source_count"] for group in groups] == [1, 1]
    assert all(group["research_scope"] == "single_source" for group in groups)


def test_huggingface_space_is_enriched_without_replacing_trusted_metadata(monkeypatch):
    from src.collectors import article_extractor

    original = article(
        "Hugging Face Spaces",
        "Idol (speridlabs/idol)",
        "https://huggingface.co/spaces/speridlabs/idol",
        source_type="space",
        publisher="Hugging Face",
        summary="Built with Gradio. Likes: 20.",
        metadata={"space_id": "speridlabs/idol"},
    )
    monkeypatch.setattr(
        article_extractor,
        "extract_article_content",
        lambda url: {
            "full_text": "Public Space description. " + "Supported details. " * 12,
            "canonical_url": "https://untrusted.example/other",
        },
    )

    groups = group_related_articles([original])

    source = groups[0]["sources"][0]
    assert source["content_source"] == "full_article"
    assert len(source["content"]) >= 120
    assert source["url"] == "https://huggingface.co/spaces/speridlabs/idol"
    assert source["canonical_url"] == source["url"]
    assert original["metadata"] == {"space_id": "speridlabs/idol"}


def test_short_space_summary_and_failed_extraction_are_insufficient(monkeypatch):
    from src.collectors import article_extractor

    monkeypatch.setattr(article_extractor, "extract_article_content", lambda _url: None)
    monkeypatch.setattr(
        "src.collectors.crawl4ai_collector.Crawl4AIAdapter.collect",
        lambda self, source, urls, allowed: type(
            "Result", (), {"articles": [], "warnings": ["robots denied"]}
        )(),
    )
    space = article(
        "Hugging Face Spaces",
        "Idol (speridlabs/idol)",
        "https://huggingface.co/spaces/speridlabs/idol",
        source_type="space",
        summary="Likes: 20.",
    )

    group = group_related_articles([space])[0]

    assert group["sources"][0]["content"] == ""
    assert group["sources"][0]["content_source"] == "metadata_only"


def test_crawl4ai_fallback_uses_huggingface_allowlist_and_clean_body(monkeypatch):
    from src.collectors import article_extractor

    monkeypatch.setattr(article_extractor, "extract_article_content", lambda _url: None)
    requested = {}

    def crawl(_self, source, urls, allowed_domains):
        requested.update(source=source, urls=urls, allowed_domains=allowed_domains)
        return type("Result", (), {"articles": [{
            "url": urls[0],
            "extracted_content": {
                "full_text": "<p>Public content. " + "Evidence. " * 20 + "</p>",
                "canonical_url": "https://attacker.example/fake",
            },
        }]})()

    monkeypatch.setattr(
        "src.collectors.crawl4ai_collector.Crawl4AIAdapter.collect", crawl
    )
    space = article(
        "Hugging Face Spaces", "Idol", "https://huggingface.co/spaces/owner/idol",
        source_type="space", summary="Likes: 20.",
    )

    source = group_related_articles([space])[0]["sources"][0]

    assert requested["allowed_domains"] == ["huggingface.co"]
    assert requested["urls"] == [space["url"]]
    assert source["content_source"] == "full_article"
    assert "<p>" not in source["content"]
    assert source["canonical_url"] == space["url"]


def test_non_space_rss_summary_keeps_existing_fallback_behavior():
    source = article(
        "Example", "A short source summary", "https://example.com/article",
        summary="Only a few words.",
    )

    group = group_related_articles([source])[0]

    assert group["sources"][0]["content"] == "Only a few words."
    assert group["sources"][0]["content_source"] == "rss_fallback"
