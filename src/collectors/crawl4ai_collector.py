"""Optional, robots-aware Crawl4AI fallback for opted-in article sources."""

import asyncio
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from . import article_extractor
from .collection_result import CollectionResult


REQUEST_INTERVAL_SECONDS = 1.0
PAGE_TIMEOUT_SECONDS = 30


def _host_allowed(url, allowed_domains):
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        return (
            parsed.scheme in ("http", "https")
            and bool(host)
            and host in {domain.lower().rstrip(".") for domain in allowed_domains}
        )
    except (TypeError, ValueError):
        return False


def _metadata_from_html(html, requested_url, final_url, allowed_domains):
    soup = BeautifulSoup(html, "html.parser")
    structured = article_extractor._jsonld_article_metadata(soup)
    title_tag = soup.title
    page_title = title_tag.get_text(" ", strip=True) if title_tag else ""
    canonical_tag = soup.find("link", rel=lambda value: value and "canonical" in value)
    canonical = (
        structured.get("canonical_url")
        or (canonical_tag.get("href", "").strip() if canonical_tag else "")
        or final_url
    )
    canonical = urljoin(final_url, canonical)
    if not _host_allowed(canonical, allowed_domains):
        canonical = final_url

    container = article_extractor._article_container(soup)
    full_text = (
        article_extractor._extract_body(container) if container is not None else ""
    ) or structured.get("article_body", "").strip()
    if not full_text:
        return None

    description = (
        structured.get("description")
        or article_extractor._meta_content(
            soup, {"name": "description"}, {"property": "og:description"}
        )
    )
    published = (
        structured.get("published")
        or article_extractor._meta_content(
            soup, {"property": "article:published_time"},
            {"name": "date"}, {"name": "pubdate"}
        )
    )
    return {
        "page_title": page_title,
        "title": structured.get("title") or article_extractor._meta_content(
            soup, {"property": "og:title"}, {"name": "twitter:title"}
        ) or page_title,
        "description": description,
        "author": structured.get("author") or article_extractor._meta_content(
            soup, {"name": "author"}, {"property": "article:author"}
        ),
        "publisher": structured.get("publisher") or article_extractor._meta_content(
            soup, {"property": "og:site_name"}
        ),
        "published": published,
        "modified": structured.get("modified", ""),
        "image": structured.get("image", ""),
        "url": canonical,
        "canonical_url": canonical,
        "full_text": full_text,
        "requested_url": requested_url,
    }


class Crawl4AIAdapter:
    """Run a single browser session for a batch of permitted article URLs."""

    def __init__(self, crawler_factory=None):
        self._crawler_factory = crawler_factory

    def collect(self, source, urls, allowed_domains):
        unique_urls = list(dict.fromkeys(urls))
        if not unique_urls:
            return CollectionResult.from_articles(source, [])
        if any(not _host_allowed(url, allowed_domains) for url in unique_urls):
            return CollectionResult.failed(
                source, "Crawl4AI refused a URL outside the configured domain allowlist."
            )

        try:
            if self._crawler_factory is None:
                from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

                crawler_factory = lambda: AsyncWebCrawler(
                    config=BrowserConfig(headless=True)
                )
                run_config_factory = CrawlerRunConfig
            else:
                crawler_factory, run_config_factory = self._crawler_factory
        except ImportError as error:
            return CollectionResult.failed(
                source,
                "Crawl4AI is unavailable; install the optional requirements-crawl4ai.txt. "
                f"Details: {error}",
            )

        async def crawl_batch():
            articles = []
            warnings = []
            async with crawler_factory() as crawler:
                for index, url in enumerate(unique_urls):
                    if index:
                        await asyncio.sleep(REQUEST_INTERVAL_SECONDS)
                    allowed, reason = article_extractor._is_allowed_by_robots(url)
                    if not allowed:
                        warnings.append(f"Skipped {url}: {reason}")
                        continue
                    try:
                        config = run_config_factory(
                            check_robots_txt=True,
                            page_timeout=PAGE_TIMEOUT_SECONDS * 1000,
                            wait_until="domcontentloaded",
                        )
                        result = await asyncio.wait_for(
                            crawler.arun(url=url, config=config),
                            timeout=PAGE_TIMEOUT_SECONDS,
                        )
                    except (asyncio.TimeoutError, Exception) as error:
                        warnings.append(f"Crawl failed for {url}: {error}")
                        continue

                    final_url = getattr(result, "url", "") or url
                    if not _host_allowed(final_url, allowed_domains):
                        warnings.append(f"Skipped cross-domain redirect from {url}.")
                        continue
                    if not getattr(result, "success", False):
                        warnings.append(
                            f"Crawl failed for {url}: "
                            f"{getattr(result, 'error_message', 'unknown crawl error')}"
                        )
                        continue
                    html = getattr(result, "cleaned_html", "") or getattr(result, "html", "")
                    if not html:
                        warnings.append(f"Crawl returned no HTML for {url}.")
                        continue
                    try:
                        metadata = _metadata_from_html(
                            html, url, final_url, allowed_domains
                        )
                    except (TypeError, ValueError) as error:
                        warnings.append(f"Could not parse {url}: {error}")
                        continue
                    if metadata is None:
                        warnings.append(f"Crawl returned empty article text for {url}.")
                        continue
                    articles.append({
                        "source": source,
                        "url": url,
                        "title": metadata.get("title") or "",
                        "summary": metadata.get("description") or "",
                        "published": metadata.get("published") or "",
                        "extracted_content": metadata,
                    })
            result = CollectionResult.from_articles(source, articles)
            result.warnings.extend(warnings)
            return result

        try:
            return asyncio.run(crawl_batch())
        except Exception as error:
            return CollectionResult.failed(source, f"Crawl4AI browser failure: {error}")
