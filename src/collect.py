
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

from src.collectors.anthropic_collector import collect_anthropic_news
from src.collectors.article_extractor import extract_article_content
from src.collectors.collection_result import (
    CollectionResult,
    CollectionStatus
)
from src.collectors.crawl4ai_collector import Crawl4AIAdapter
from src.collectors.huggingface_collector import (
    collect_huggingface_papers
)
from src.collectors.producthunt_collector import collect_producthunt
from src.collectors.rss_collector import collect_rss
from src.collectors.spaces_collector import collect_hf_spaces
from src.collectors.twitter_collector import collect_twitter
from src.configuration import load_sources_config
from src.processing.deduplicate import canonicalize_url, deduplicate_articles
from src.processing.history import filter_previously_reported, mark_as_reported
from src.processing.json_storage import atomic_write_json
from src.research.grouping import group_related_articles


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_FILE = OUTPUT_DIR / "updates.json"


@dataclass
class PipelineResult:
    source_results: List[CollectionResult]
    saved_articles: List[Dict]
    research_groups: List[Dict]


def collect_all_sources(sources=None):
    """
    Collect each configured source independently.
    """

    if sources is None:
        sources = load_sources_config()

    results = []

    for source in sources:
        try:
            if source.type == "rss":
                result = collect_rss(
                    source=source.name,
                    feed_url=source.url,
                    max_articles=source.max_articles
                )
            elif source.type == "api":
                result = collect_huggingface_papers(
                    max_articles=source.max_articles,
                    api_url=source.url
                )
            elif source.type == "sitemap":
                result = collect_anthropic_news(
                    max_articles=source.max_articles,
                    sitemap_url=source.url,
                )
            elif source.type == "api_hf_spaces":
                result = collect_hf_spaces(
                    max_articles=source.max_articles,
                    api_url=source.url,
                )
            elif source.type == "twitter":
                result = collect_twitter(
                    max_articles=source.max_articles,
                    search_url=source.url,
                )
            elif source.type == "producthunt":
                result = collect_producthunt(
                    max_articles=source.max_articles,
                    api_url=source.url,
                )
            else:
                raise ValueError(
                    f"Unsupported source type: {source.type!r}"
                )

            if not isinstance(result, CollectionResult):
                raise TypeError(
                    "Collector did not return a CollectionResult."
                )
        except Exception as error:
            result = CollectionResult.failed(
                source.name,
                f"Unexpected collection error: {error}"
            )

        if (
            source.extract_articles
            and source.type == "rss"
            and result.status != CollectionStatus.FAILED
        ):
            crawl_candidates = []
            for article in result.articles:
                article["extraction_status"] = "rss_fallback"
                article["content_source_type"] = "rss_fallback"
                try:
                    extracted = extract_article_content(article["url"])
                    if extracted is None:
                        result.warnings.append(
                            f"Article extraction failed for {article['url']}."
                        )
                        if getattr(source, "crawl4ai_fallback", False):
                            crawl_candidates.append(article["url"])
                    else:
                        article["extracted_content"] = extracted
                        article["extraction_status"] = "full_article"
                        article["content_source_type"] = "full_article"
                except Exception as error:
                    result.warnings.append(
                        f"Article extraction failed for "
                        f"{article.get('url', 'unknown URL')}: {error}"
                    )
                    if getattr(source, "crawl4ai_fallback", False):
                        crawl_candidates.append(article.get("url", ""))

            if crawl_candidates and getattr(source, "crawl4ai_fallback", False):
                crawl_result = Crawl4AIAdapter().collect(
                    source.name,
                    crawl_candidates,
                    source.crawl4ai_allowed_domains,
                )
                if crawl_result.status == CollectionStatus.FAILED:
                    result.warnings.append(crawl_result.error)
                else:
                    extracted_by_url = {
                        canonicalize_url(crawled.get("url", "")): crawled
                        for crawled in crawl_result.articles
                    }
                    for article in result.articles:
                        crawled = extracted_by_url.get(
                            canonicalize_url(article.get("url", ""))
                        )
                        if crawled:
                            article["extracted_content"] = crawled["extracted_content"]
                            article["extraction_status"] = "full_article"
                            article["content_source_type"] = "full_article"
                    result.warnings.extend(crawl_result.warnings)
        elif (
            source.type not in ("sitemap", "api_hf_spaces", "twitter", "producthunt")
            and result.status != CollectionStatus.FAILED
        ):
            for article in result.articles:
                article["content_source_type"] = (
                    "rss_fallback" if source.type == "rss"
                    else "source_summary"
                )
                article["extraction_status"] = "not_attempted"

        # Stamp the content_type from the source config onto every article
        # (collectors that set it themselves are not overwritten here because
        # their value will be the same or more specific).
        if result.status != CollectionStatus.FAILED:
            for article in result.articles:
                if "content_type" not in article:
                    article["content_type"] = source.content_type

        results.append(result)

    return results


def save_articles(articles, output_file=None):
    """Atomically save the latest collection result as JSON."""

    path = Path(output_file) if output_file is not None else OUTPUT_FILE
    atomic_write_json(path, articles)
    print("\n[OK] Articles saved to:", path)
    return path


def print_source_summary(results):
    """Display collection status, article count, and any source issues."""

    print("\nSOURCE COLLECTION SUMMARY")
    for result in results:
        message = (
            f"{result.source}: {result.status.value} "
            f"({len(result.articles)} articles)"
        )
        if result.error:
            message += f" — {result.error}"
        print(message)
        for warning in result.warnings:
            print(f"  WARNING: {warning}")


def run_pipeline(sources=None, output_file=None, history_file=None):
    """Collect, filter, save, then record only successfully saved articles."""

    source_results = collect_all_sources(sources)
    print_source_summary(source_results)

    collected_articles = [
        article
        for result in source_results
        if result.status != CollectionStatus.FAILED
        for article in result.articles
    ]
    unique_articles = deduplicate_articles(collected_articles)
    new_articles = filter_previously_reported(
        unique_articles,
        history_file=history_file
    )

    save_articles(new_articles, output_file=output_file)
    mark_as_reported(new_articles, history_file=history_file)

    return PipelineResult(
        source_results=source_results,
        saved_articles=new_articles,
        research_groups=group_related_articles(new_articles)
    )


if __name__ == "__main__":

    print("=" * 50)
    print("AI DAILY INTELLIGENCE AGENT")
    print("=" * 50)

    print("\nCOLLECTING ARTICLES")
    pipeline_result = run_pipeline(load_sources_config())
    print("\nArticles saved:", len(pipeline_result.saved_articles))

    print("\n" + "=" * 50)
    print("COLLECTION COMPLETED")
    print("=" * 50)
