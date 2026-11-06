import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from ..processing.json_storage import atomic_write_json
from .cache import (
    SummaryCache,
    content_fingerprint
)
from .gemini_client import (
    GeminiClientError
)
from .providers import GroqProviderError, create_llm_client
from .schemas import validate_summary_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_FILE = PROJECT_ROOT / "output" / "updates.json"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "output" / "summarized_updates.json"
DEFAULT_MAX_ARTICLES_PER_RUN = 3


def get_max_articles_per_run(value=None):
    load_dotenv(PROJECT_ROOT / ".env")
    raw_value = (
        value
        if value is not None
        else os.getenv(
            "MAX_ARTICLES_PER_RUN",
            str(DEFAULT_MAX_ARTICLES_PER_RUN)
        )
    )
    try:
        limit = int(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "MAX_ARTICLES_PER_RUN must be zero or a positive integer."
        ) from error
    if limit < 0:
        raise ValueError(
            "MAX_ARTICLES_PER_RUN must be zero or a positive integer."
        )
    return limit


def select_article_content(article):
    """Choose source text in order: extracted text, RSS summary, metadata."""

    extracted = article.get("extracted_content")
    if isinstance(extracted, dict):
        full_text = extracted.get("full_text")
        if isinstance(full_text, str) and full_text.strip():
            return full_text.strip(), "extracted_full_text"
    elif isinstance(extracted, str) and extracted.strip():
        return extracted.strip(), "extracted_full_text"

    summary = article.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip(), "source_summary"

    metadata_values = []
    for value in (
        article.get("description"),
        article.get("abstract"),
        article.get("content"),
        extracted.get("description") if isinstance(extracted, dict) else None
    ):
        if isinstance(value, str) and value.strip():
            metadata_values.append(value.strip())

    return "\n\n".join(metadata_values), "descriptive_metadata"


def has_sufficient_content(content):
    words = content.split()
    return len(content.strip()) >= 50 and len(words) >= 8


def _base_record(article, status):
    return {
        "title": str(article.get("title") or "").strip(),
        "source": str(article.get("source") or "Unknown").strip(),
        "source_url": str(article.get("url") or "").strip(),
        "published_date": str(article.get("published") or "").strip(),
        "category": "Other",
        "short_summary": "",
        "key_points": [],
        "why_it_matters": "",
        "importance": "Low",
        "confidence": 0.0,
        "summary_status": status
    }


def _failed_record(article, error, api_key=""):
    record = _base_record(article, "FAILED")
    record["error"] = _safe_error_message(error, api_key)
    if isinstance(error, GeminiClientError):
        if error.status_code is not None:
            record["error_status_code"] = error.status_code
        if error.error_type:
            record["error_type"] = error.error_type
    return record


def _insufficient_record(article):
    record = _base_record(article, "INSUFFICIENT_CONTENT")
    record["short_summary"] = "Insufficient source content to summarize reliably."
    record["why_it_matters"] = "Not specified in the source."
    return record


def _safe_error_message(error, api_key):
    message = str(error)
    for secret in (
        os.getenv("GROQ_API_KEY", "").strip(),
        os.getenv("GEMINI_API_KEY", "").strip(),
        api_key,
    ):
        if secret:
            message = message.replace(secret, "[REDACTED]")
    message = re.sub(
        r"(?i)\bAIza[A-Za-z0-9_-]{20,}\b",
        "[REDACTED]",
        message
    )
    message = re.sub(
        r"(?i)\bBearer\s+\S+",
        "Bearer [REDACTED]",
        message
    )
    message = re.sub(
        r"(?im)^\s*(?:authorization|x-goog-api-key|gemini-api-key)\s*:\s*.*$",
        "[REDACTED HEADER]",
        message
    )
    return message[:500]


def summarize_articles(
    articles,
    client=None,
    cache=None,
    max_articles=None
):
    """Summarize selected articles, preserving failures and cache successes."""

    if max_articles is None:
        max_articles = get_max_articles_per_run()

    if max_articles is not None and max_articles < 0:
        raise ValueError("max_articles must be zero, positive, or None.")

    selected_articles = (
        articles[:max_articles]
        if max_articles
        else articles
    )
    cache = cache or SummaryCache()
    results = []
    prepared = []
    for article in selected_articles:
        title = str(article.get("title") or "").strip()
        source_content, content_source = select_article_content(article)
        fingerprint = content_fingerprint(title, source_content)
        cached = None

        if has_sufficient_content(source_content):
            cached_entry = cache.get(article.get("url", ""), fingerprint)
            if cached_entry is not None:
                try:
                    cached = validate_summary_data(cached_entry, title)
                except ValueError as error:
                    print(
                        "[WARNING] Ignoring invalid cached summary "
                        f"({type(error).__name__})."
                    )

        prepared.append(
            (article, title, source_content, content_source, fingerprint, cached)
        )

    needs_client = any(
        has_sufficient_content(source_content) and cached is None
        for _, _, source_content, _, _, cached in prepared
    )

    if client is None and needs_client:
        client = create_llm_client()
        api_key = os.getenv("GROQ_API_KEY", "").strip()
    else:
        api_key = os.getenv("GROQ_API_KEY", "").strip()

    cache_hits = 0
    api_calls = 0

    for (
        article,
        title,
        source_content,
        content_source,
        fingerprint,
        cached
    ) in prepared:
        if not has_sufficient_content(source_content):
            results.append(_insufficient_record(article))
            continue

        if cached is not None:
            generated = cached
            cache_hits += 1
        else:
            if client is None:
                raise RuntimeError(
                    "LLM client was not initialized for an uncached "
                    "article."
                )

            api_calls += 1
            print(
                f"[API] Summarizing {article.get('source', 'Unknown')} "
                f"article with model="
                f"{getattr(client, 'model', 'configured model')}, "
                f"max_output_tokens="
                f"{getattr(client, 'max_output_tokens', 'configured limit')}."
            )
            prompt_content = source_content[:12000]
            if len(source_content) > len(prompt_content):
                prompt_content += "\n[Source text truncated for token control.]"

            try:
                generated = client.generate_summary(title, prompt_content)
                generated = validate_summary_data(generated, title)
            except (GeminiClientError, ValueError) as error:
                if isinstance(error, GroqProviderError) and error.error_type in (
                    "TOKEN_BUDGET_EXHAUSTED", "DAILY_QUOTA_EXHAUSTED"
                ):
                    raise GroqProviderError(
                        f"{error} Summary processing stopped at article "
                        f"{len(results) + 1}/{len(selected_articles)}; completed "
                        "summaries remain cached for resume.",
                        status_code=error.status_code,
                        error_type=error.error_type,
                        transient=error.transient,
                        stage=error.stage,
                        api_detail=error.api_detail,
                    ) from error
                results.append(
                    _failed_record(
                        article,
                        error,
                        api_key
                    )
                )
                continue
            try:
                cache.set(article.get("url", ""), fingerprint, generated)
            except Exception as error:
                print(
                    "[WARNING] Summary succeeded but could not be cached "
                    f"({type(error).__name__})."
                )

        generated["title"] = title or generated["title"]
        record = _base_record(article, "SUCCESS")
        record.update(generated)
        record["title"] = title or generated["title"]
        record["source_url"] = str(article.get("url") or "").strip()
        record["published_date"] = str(
            article.get("published") or ""
        ).strip()
        record["content_source_type"] = content_source
        results.append(record)

    print(
        f"Summarization cost controls: API calls={api_calls}, "
        f"successful cache hits={cache_hits}, "
        f"selected articles={len(selected_articles)}."
    )
    return results


def load_articles(input_file=None):
    path = Path(input_file) if input_file is not None else DEFAULT_INPUT_FILE
    with path.open("r", encoding="utf-8") as file:
        articles = json.load(file)
    if not isinstance(articles, list) or any(
        not isinstance(article, dict) for article in articles
    ):
        raise ValueError(
            f"Input file '{path}' must contain a JSON list of article objects."
        )
    return articles


def save_summaries(summaries, output_file=None):
    path = Path(output_file) if output_file is not None else DEFAULT_OUTPUT_FILE
    atomic_write_json(path, summaries)
    return path
