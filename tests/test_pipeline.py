import json

import pytest
import requests

import src.collect as collect
import src.processing.json_storage as json_storage
from src.collectors.collection_result import (
    CollectionResult,
    CollectionStatus
)
from src.configuration import SourceConfig


def test_one_source_failure_does_not_stop_successful_sources(monkeypatch, capsys):
    sources = [
        SourceConfig("Failed feed", "rss", "https://example.com/broken", 5),
        SourceConfig("Working feed", "rss", "https://example.com/good", 5)
    ]
    article = {
        "source": "Working feed",
        "title": "Collected article",
        "url": "https://example.com/article",
        "published": "",
        "summary": "Summary",
        "collected_at": "2026-11-02T00:00:00+00:00"
    }
    calls = []

    def fake_collect_rss(source, feed_url, max_articles):
        calls.append(source)
        if source == "Failed feed":
            raise requests.Timeout("source timed out")
        return CollectionResult.from_articles(source, [article])

    monkeypatch.setattr(collect, "collect_rss", fake_collect_rss)

    results = collect.collect_all_sources(sources)
    collect.print_source_summary(results)

    assert calls == ["Failed feed", "Working feed"]
    assert [result.status for result in results] == [
        CollectionStatus.FAILED,
        CollectionStatus.SUCCESS
    ]
    summary = capsys.readouterr().out
    assert "Failed feed: FAILED (0 articles)" in summary
    assert "source timed out" in summary
    assert "Working feed: SUCCESS (1 articles)" in summary


def test_extraction_enriches_article_without_replacing_rss_fields(monkeypatch):
    source = SourceConfig(
        "Google DeepMind",
        "rss",
        "https://deepmind.google/blog/rss.xml",
        5,
        extract_articles=True
    )
    article = {
        "source": "Google DeepMind",
        "title": "RSS title",
        "url": "https://deepmind.google/blog/example/",
        "published": "RSS date",
        "summary": "RSS summary"
    }
    extracted = {"title": "Page title", "full_text": "Article body"}
    monkeypatch.setattr(
        collect,
        "collect_rss",
        lambda *args, **kwargs: CollectionResult.from_articles(
            source.name,
            [article.copy()]
        )
    )
    monkeypatch.setattr(
        collect,
        "extract_article_content",
        lambda url: extracted
    )

    result = collect.collect_all_sources([source])[0]

    assert result.status == CollectionStatus.SUCCESS
    enriched = result.articles[0]
    assert enriched["title"] == "RSS title"
    assert enriched["url"] == article["url"]
    assert enriched["summary"] == "RSS summary"
    assert enriched["extracted_content"] == extracted
    assert enriched["extraction_status"] == "full_article"
    assert enriched["content_source_type"] == "full_article"


@pytest.mark.parametrize("extractor_result", [None, RuntimeError("blocked")])
def test_extraction_failure_keeps_rss_metadata(
    monkeypatch,
    extractor_result
):
    source = SourceConfig(
        "Google DeepMind",
        "rss",
        "https://deepmind.google/blog/rss.xml",
        5,
        extract_articles=True
    )
    article = {
        "source": source.name,
        "title": "RSS title",
        "url": "https://deepmind.google/blog/example/",
        "published": "RSS date",
        "summary": "RSS summary"
    }
    monkeypatch.setattr(
        collect,
        "collect_rss",
        lambda *args, **kwargs: CollectionResult.from_articles(
            source.name,
            [article.copy()]
        )
    )

    def extract(url):
        if isinstance(extractor_result, Exception):
            raise extractor_result
        return extractor_result

    monkeypatch.setattr(collect, "extract_article_content", extract)

    result = collect.collect_all_sources([source])[0]

    assert result.status == CollectionStatus.SUCCESS
    assert result.articles[0]["title"] == article["title"]
    assert result.articles[0]["url"] == article["url"]
    assert result.articles[0]["summary"] == article["summary"]
    assert result.articles[0]["extraction_status"] == "rss_fallback"
    assert result.articles[0]["content_source_type"] == "rss_fallback"
    assert result.warnings


def test_extraction_is_not_run_for_unconfigured_sources(monkeypatch):
    sources = [
        SourceConfig(
            "Latent Space",
            "rss",
            "https://podcast.latent.space/feed.xml",
            5
        ),
        SourceConfig(
            "Hugging Face Papers",
            "api",
            "https://huggingface.co/api/daily_papers",
            5
        )
    ]
    monkeypatch.setattr(
        collect,
        "collect_rss",
        lambda source, **kwargs: CollectionResult.from_articles(
            source,
            [{"source": source, "title": "Episode", "url": "https://podcast.latent.space/1"}]
        )
    )
    monkeypatch.setattr(
        collect,
        "collect_huggingface_papers",
        lambda **kwargs: CollectionResult.from_articles(
            "Hugging Face Papers",
            [{"source": "Hugging Face Papers", "title": "Paper", "url": "https://huggingface.co/papers/1", "summary": "API summary"}]
        )
    )
    monkeypatch.setattr(
        collect,
        "extract_article_content",
        lambda url: pytest.fail("Extraction should be disabled.")
    )

    results = collect.collect_all_sources(sources)

    assert len(results) == 2
    assert all(result.status == CollectionStatus.SUCCESS for result in results)


def test_pipeline_saves_only_new_articles_then_updates_history(
    tmp_path,
    monkeypatch
):
    output_file = tmp_path / "updates.json"
    history_file = tmp_path / "processed_history.json"
    history_file.write_text(
        json.dumps(["https://example.com/old"]),
        encoding="utf-8"
    )
    old_article = {
        "source": "OpenAI",
        "title": "Already reported",
        "url": "https://example.com/old"
    }
    new_article = {
        "source": "OpenAI",
        "title": "New article",
        "url": "https://example.com/new",
        "summary": "Summary"
    }
    collection_results = [
        CollectionResult.from_articles(
            "OpenAI",
            [old_article, new_article]
        )
    ]
    monkeypatch.setattr(
        collect,
        "collect_all_sources",
        lambda sources=None: collection_results
    )

    result = collect.run_pipeline(
        sources=[],
        output_file=output_file,
        history_file=history_file
    )

    assert result.saved_articles == [new_article]
    assert result.research_groups[0]["research_scope"] == "single_source"
    assert json.loads(output_file.read_text(encoding="utf-8")) == [new_article]
    assert json.loads(history_file.read_text(encoding="utf-8")) == [
        "https://example.com/new",
        "https://example.com/old"
    ]


def test_pipeline_writes_empty_output_for_empty_collection(tmp_path, monkeypatch):
    output_file = tmp_path / "updates.json"
    history_file = tmp_path / "processed_history.json"
    output_file.write_text('[{"old": true}]', encoding="utf-8")
    history_file.write_text(
        json.dumps(["https://example.com/old"]),
        encoding="utf-8"
    )
    monkeypatch.setattr(
        collect,
        "collect_all_sources",
        lambda sources=None: [
            CollectionResult(
                source="OpenAI",
                status=CollectionStatus.EMPTY
            )
        ]
    )

    result = collect.run_pipeline(
        sources=[],
        output_file=output_file,
        history_file=history_file
    )

    assert result.saved_articles == []
    assert json.loads(output_file.read_text(encoding="utf-8")) == []
    assert json.loads(history_file.read_text(encoding="utf-8")) == [
        "https://example.com/old"
    ]


def test_pipeline_does_not_update_history_when_output_save_fails(
    tmp_path,
    monkeypatch
):
    output_file = tmp_path / "updates.json"
    history_file = tmp_path / "processed_history.json"
    previous_output = '{"previous": "output"}'
    previous_history = json.dumps(["https://example.com/old"])
    output_file.write_text(previous_output, encoding="utf-8")
    history_file.write_text(previous_history, encoding="utf-8")
    article = {
        "source": "OpenAI",
        "title": "New article",
        "url": "https://example.com/new"
    }
    monkeypatch.setattr(
        collect,
        "collect_all_sources",
        lambda sources=None: [
            CollectionResult.from_articles("OpenAI", [article])
        ]
    )

    def fail_replace(source, destination):
        raise OSError("simulated atomic replace failure")

    monkeypatch.setattr(json_storage.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated atomic replace failure"):
        collect.run_pipeline(
            sources=[],
            output_file=output_file,
            history_file=history_file
        )

    assert output_file.read_text(encoding="utf-8") == previous_output
    assert history_file.read_text(encoding="utf-8") == previous_history
    assert list(tmp_path.glob("*.tmp")) == []
