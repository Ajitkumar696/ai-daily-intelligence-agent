"""
Anthropic News collector — sitemap-based discovery.

Anthropic does not publish an RSS feed.  Instead we parse their sitemap at
https://www.anthropic.com/sitemap.xml, filter for /news/ URLs, sort by the
<lastmod> date, and extract each article using the shared article_extractor
(which enforces robots.txt compliance and 1-second rate-limiting).
"""

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from urllib.parse import urlsplit

import requests

from .article_extractor import extract_article_content
from .collection_result import CollectionResult


SITEMAP_URL = "https://www.anthropic.com/sitemap.xml"
NEWS_PATH_RE = re.compile(r"^/news/", re.IGNORECASE)


def _parse_sitemap(xml_text):
    """Return list of (url, lastmod_str) for /news/ paths, newest first."""
    entries = []
    if not xml_text or not xml_text.strip():
        return entries

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return entries

    for url_node in root.findall(".//{*}url"):
        loc_node = url_node.find("{*}loc")
        if loc_node is None or not loc_node.text:
            continue

        url = loc_node.text.strip()
        parsed = urlsplit(url)
        if not NEWS_PATH_RE.match(parsed.path):
            continue

        lastmod_node = url_node.find("{*}lastmod")
        lastmod = (
            lastmod_node.text.strip()
            if (lastmod_node is not None and lastmod_node.text)
            else ""
        )
        entries.append((url, lastmod))

    # Sort newest first; ISO date strings sort correctly as strings.
    entries.sort(key=lambda item: item[1], reverse=True)
    return entries


def collect_anthropic_news(max_articles=20, sitemap_url=SITEMAP_URL):
    """
    Collect recent Anthropic news articles via sitemap discovery.

    Returns a CollectionResult with fully-extracted articles where possible,
    or metadata-only entries where extraction is blocked/unavailable.
    """
    print("\nCollecting from Anthropic News...")

    try:
        response = requests.get(
            sitemap_url,
            timeout=(10, 30),
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()

        entries = _parse_sitemap(response.text)
        if not entries:
            return CollectionResult.failed(
                "Anthropic News",
                "No /news/ URLs found in sitemap.",
            )

        print(f"Sitemap entries found: {len(entries)}, taking top {max_articles}")
        entries = entries[:max_articles]

        articles = []
        for url, lastmod in entries:
            article = _extract_or_stub(url, lastmod)
            if article:
                articles.append(article)

        result = CollectionResult.from_articles("Anthropic News", articles)

    except requests.exceptions.RequestException as error:
        result = CollectionResult.failed(
            "Anthropic News",
            f"Network or HTTP request failed: {error}",
        )
    except Exception as error:
        result = CollectionResult.failed("Anthropic News", str(error))

    if result.status.value == "FAILED":
        print(f"[ERROR] Anthropic News: {result.error}")
    else:
        print(
            f"[{result.status.value}] Anthropic News: "
            f"collected {len(result.articles)} articles"
        )

    return result


def _extract_or_stub(url, lastmod):
    """
    Try full extraction via article_extractor; fall back to a stub article
    with just the URL and date if extraction returns nothing.
    """
    slug = urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1]
    title_fallback = slug.replace("-", " ").title()

    extracted = extract_article_content(url)

    if extracted:
        return {
            "source": "Anthropic News",
            "publisher": "Anthropic",
            "source_type": "full_article",
            "content_type": "article",
            "title": extracted.get("title") or title_fallback,
            "url": extracted.get("url") or url,
            "published": extracted.get("published") or lastmod,
            "summary": extracted.get("description") or "",
            "content": extracted.get("full_text") or "",
            "authors": (
                [extracted["author"]] if extracted.get("author") else []
            ),
            "metadata": {
                "lastmod": lastmod,
                "image": extracted.get("image", ""),
                "canonical_url": extracted.get("canonical_url", ""),
            },
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "extraction_status": "full_article",
            "content_source_type": "full_article",
        }

    # Robots.txt allows the path but extraction yielded no body — return a stub
    # so at least the URL/date is tracked.
    return {
        "source": "Anthropic News",
        "publisher": "Anthropic",
        "source_type": "metadata_only",
        "content_type": "article",
        "title": title_fallback,
        "url": url,
        "published": lastmod,
        "summary": "",
        "content": "",
        "authors": [],
        "metadata": {"lastmod": lastmod},
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "extraction_status": "metadata_only",
        "content_source_type": "metadata_only",
    }
