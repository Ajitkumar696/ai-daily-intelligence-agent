import re
from collections import defaultdict
from urllib.parse import urlsplit

from ..processing.deduplicate import canonicalize_url


_STOPWORDS = {
    "about", "after", "amid", "and", "announcement", "announces",
    "announced", "are", "as", "at", "by", "for", "from", "in", "into",
    "introduces", "is", "its", "launch", "launches", "launched", "new",
    "of", "on", "or", "reveals", "says", "the", "to", "with"
}
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
MIN_RESEARCH_TEXT_CHARS = 120
HF_SPACES_HOST = "huggingface.co"


def _topic_tokens(article):
    title = str(article.get("title") or "").casefold()
    return {
        token
        for token in _TOKEN_PATTERN.findall(title)
        if token not in _STOPWORDS and (len(token) >= 3 or token.isdigit())
    }


def _related(left, right):
    left_tokens = _topic_tokens(left)
    right_tokens = _topic_tokens(right)
    shared = left_tokens & right_tokens
    return len(shared) >= 2 and len(shared) / len(left_tokens | right_tokens) >= 0.25


def _article_content(article):
    is_space = article.get("source_type") == "space"
    extracted = article.get("extracted_content")
    if isinstance(extracted, dict):
        body = extracted.get("full_text")
        cleaned_body = _clean_article_text(body)
        if cleaned_body and (not is_space or len(cleaned_body) >= MIN_RESEARCH_TEXT_CHARS):
            return cleaned_body, "full_article"

    summary = article.get("summary")
    if isinstance(summary, str) and summary.strip() and (
        not is_space
        or len(_clean_article_text(summary)) >= MIN_RESEARCH_TEXT_CHARS
    ):
        return summary.strip(), (
            "rss_fallback" if article.get("source_type", "rss") == "rss"
            else "source_summary"
        )
    return "", "metadata_only"


def _clean_article_text(value):
    """Normalize extracted text and reject markup or non-content responses."""
    from bs4 import BeautifulSoup

    if not isinstance(value, str):
        return ""
    text = BeautifulSoup(value, "html.parser").get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def _enrich_huggingface_spaces(articles):
    """Fetch short or missing public Space text under existing robots limits."""
    candidates = []
    for article in articles:
        if not isinstance(article, dict) or article.get("source_type") != "space":
            continue
        try:
            parsed = urlsplit(article.get("url", ""))
        except (TypeError, ValueError):
            continue
        if (
            parsed.scheme not in ("http", "https")
            or (parsed.hostname or "").lower().rstrip(".") != HF_SPACES_HOST
            or _article_content(article)[1] != "metadata_only"
        ):
            continue
        candidates.append(article)

    if not candidates:
        return

    from ..collectors import article_extractor
    from ..collectors.crawl4ai_collector import Crawl4AIAdapter

    failed = []
    for article in candidates:
        try:
            extracted = article_extractor.extract_article_content(article["url"])
        except Exception:
            extracted = None
        body = _clean_article_text(
            extracted.get("full_text") if isinstance(extracted, dict) else ""
        )
        if len(body) >= MIN_RESEARCH_TEXT_CHARS:
            article["extracted_content"] = {
                "full_text": body,
                "publisher": article.get("publisher") or article.get("source"),
                "canonical_url": article["url"],
            }
            article["content_source_type"] = "full_article"
        else:
            failed.append(article)

    if not failed:
        return
    try:
        crawl_result = Crawl4AIAdapter().collect(
            "Hugging Face Spaces",
            [article["url"] for article in failed],
            [HF_SPACES_HOST],
        )
    except Exception:
        return
    crawled_by_url = {
        item.get("url"): item
        for item in crawl_result.articles
        if isinstance(item, dict)
    }
    for article in failed:
        crawled = crawled_by_url.get(article["url"])
        content = crawled.get("extracted_content") if crawled else None
        body = _clean_article_text(
            content.get("full_text") if isinstance(content, dict) else ""
        )
        if len(body) < MIN_RESEARCH_TEXT_CHARS:
            continue
        article["extracted_content"] = {
            "full_text": body,
            "publisher": article.get("publisher") or article.get("source"),
            "canonical_url": article["url"],
        }
        article["content_source_type"] = "full_article"


def _source_evidence(article):
    extracted = article.get("extracted_content")
    if not isinstance(extracted, dict):
        extracted = {}
    content, content_quality = _article_content(article)
    return {
        "publisher": str(
            extracted.get("publisher")
            or article.get("publisher")
            or article.get("source")
            or "Unknown"
        ).strip(),
        "title": str(article.get("title") or "").strip(),
        "url": str(article.get("url") or "").strip(),
        "canonical_url": str(
            extracted.get("canonical_url") or extracted.get("url")
            or article.get("url") or ""
        ).strip(),
        "published": str(
            extracted.get("published") or article.get("published") or ""
        ).strip(),
        "author": str(extracted.get("author") or "").strip(),
        "content": content,
        "content_source": content_quality
    }


def _cross_check(sources):
    titles = [source["title"] for source in sources]
    dates = {source["published"] for source in sources if source["published"]}
    shared_terms = (
        set.intersection(*[
            _topic_tokens({"title": source["title"]}) for source in sources
        ])
        if sources else set()
    )
    differences = []
    if len(set(titles)) > 1:
        differences.append({
            "field": "reported_title",
            "values": [
                {"publisher": source["publisher"], "value": source["title"]}
                for source in sources
            ]
        })
    if len(dates) > 1:
        differences.append({
            "field": "publication_date",
            "values": [
                {"publisher": source["publisher"], "value": source["published"]}
                for source in sources
                if source["published"]
            ]
        })

    missing_metadata = []
    for source in sources:
        missing = [
            field
            for field in ("published", "author")
            if not source[field]
        ]
        if missing:
            missing_metadata.append({
                "publisher": source["publisher"],
                "url": source["url"],
                "missing_fields": missing
            })

    return {
        "shared_topic_terms": sorted(shared_terms),
        "metadata_differences": differences,
        "missing_metadata": missing_metadata,
        "claim_agreement": "Not assessed; source-grounded analysis is pending."
    }


def group_related_articles(articles):
    """Build source-linked research groups without asserting unverified facts."""

    _enrich_huggingface_spaces(articles)

    candidates = []
    seen_urls = set()
    for article in articles:
        if not isinstance(article, dict):
            continue
        url = canonicalize_url(article.get("url", ""))
        title = str(article.get("title") or "").strip()
        try:
            parsed_url = urlsplit(url)
        except ValueError:
            continue
        if (
            not url
            or parsed_url.scheme not in ("http", "https")
            or not parsed_url.netloc
            or not title
            or url in seen_urls
        ):
            continue
        seen_urls.add(url)
        candidates.append(article)

    parents = list(range(len(candidates)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            if _related(candidates[left], candidates[right]):
                union(left, right)

    grouped = defaultdict(list)
    for index, article in enumerate(candidates):
        grouped[find(index)].append(article)

    research_groups = []
    for articles_in_group in grouped.values():
        sources = [_source_evidence(article) for article in articles_in_group]
        distinct_publishers = {
            source["publisher"].casefold() for source in sources
        }
        research_groups.append({
            "topic": sources[0]["title"],
            "headline": sources[0]["title"],
            "research_scope": (
                "multi_source" if len(distinct_publishers) > 1 else "single_source"
            ),
            "source_count": len(distinct_publishers),
            "sources": sources,
            "cross_check": _cross_check(sources)
        })

    return research_groups
