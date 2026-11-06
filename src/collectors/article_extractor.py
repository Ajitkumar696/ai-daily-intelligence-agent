import json
import re
import time
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup


USER_AGENT = "AI-Daily-Intelligence-Agent"
MIN_REQUEST_INTERVAL_SECONDS = 1.0
_ROBOTS_CACHE = {}
_LAST_REQUEST_AT = {}
_NON_CONTENT_SELECTOR = re.compile(
    r"(advert|banner|breadcrumb|cookie|footer|header|menu|nav|newsletter|"
    r"promo|related|share|sidebar|social|subscribe)",
    re.IGNORECASE
)


def _rate_limited_get(url, **kwargs):
    parsed_url = urlsplit(url)
    origin = urlunsplit((parsed_url.scheme, parsed_url.netloc, "", "", ""))
    previous_request = _LAST_REQUEST_AT.get(origin)
    if previous_request is not None:
        elapsed = time.monotonic() - previous_request
        if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
            time.sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)

    try:
        return requests.get(url, **kwargs)
    finally:
        _LAST_REQUEST_AT[origin] = time.monotonic()


def _is_allowed_by_robots(url):
    """Fail closed when robots.txt cannot be checked or disallows the URL."""

    parsed_url = urlsplit(url)
    if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
        return False, "Article URL is not a valid HTTP(S) URL."

    origin = urlunsplit(
        (parsed_url.scheme, parsed_url.netloc, "", "", "")
    )
    if origin not in _ROBOTS_CACHE:
        robots_url = f"{origin}/robots.txt"

        try:
            response = _rate_limited_get(
                robots_url,
                headers={"User-Agent": USER_AGENT},
                timeout=(10, 20)
            )
            response.raise_for_status()

            parser = RobotFileParser()
            parser.set_url(robots_url)
            parser.parse(response.text.splitlines())
            _ROBOTS_CACHE[origin] = (parser, None)
        except requests.exceptions.RequestException as error:
            _ROBOTS_CACHE[origin] = (
                None,
                f"Unable to verify robots.txt: {error}"
            )

    parser, error = _ROBOTS_CACHE[origin]
    if parser is None:
        return False, error

    if not parser.can_fetch(USER_AGENT, url):
        return False, "robots.txt disallows article extraction."

    return True, None


def _fetch_permitted_article(url, headers):
    current_url = url
    for _ in range(6):
        allowed, reason = _is_allowed_by_robots(current_url)
        if not allowed:
            return None, reason, current_url
        response = _rate_limited_get(
            current_url,
            headers=headers,
            timeout=(15, 30),
            allow_redirects=False
        )
        if response.status_code not in (301, 302, 303, 307, 308):
            response.raise_for_status()
            return response, None, current_url
        location = response.headers.get("Location")
        if not location:
            response.raise_for_status()
            raise requests.exceptions.RequestException(
                "Article redirect did not include a destination."
            )
        current_url = urljoin(current_url, location)

    raise requests.exceptions.TooManyRedirects(
        "Article URL exceeded the allowed redirect limit."
    )


def _meta_content(soup, *selectors):
    for selector in selectors:
        tag = soup.find("meta", attrs=selector)
        if tag:
            value = tag.get("content", "").strip()
            if value:
                return value
    return ""


def _article_candidates(data):
    if isinstance(data, list):
        for item in data:
            yield from _article_candidates(item)
    elif isinstance(data, dict):
        yield data
        graph = data.get("@graph")
        if isinstance(graph, list):
            yield from _article_candidates(graph)


def _jsonld_article_metadata(soup):
    metadata = {}
    article_body = ""
    for script in soup.find_all(
        "script",
        attrs={"type": "application/ld+json"}
    ):
        try:
            data = json.loads(script.string or script.get_text())
        except (json.JSONDecodeError, TypeError):
            continue

        for item in _article_candidates(data):
            article_types = item.get("@type", [])
            if isinstance(article_types, str):
                article_types = [article_types]
            article_types = [
                str(article_type).rsplit("/", 1)[-1].rsplit("#", 1)[-1]
                for article_type in article_types
            ]
            if not any(
                article_type in ("NewsArticle", "Article", "BlogPosting")
                for article_type in article_types
            ):
                continue

            for key, value in (
                ("title", item.get("headline", "")),
                ("description", item.get("description", "")),
                ("published", item.get("datePublished", "")),
                ("modified", item.get("dateModified", ""))
            ):
                if isinstance(value, str) and value and not metadata.get(key):
                    metadata[key] = value

            author = item.get("author", [])
            if isinstance(author, (str, dict)):
                author = [author]
            if isinstance(author, list):
                names = [
                    person if isinstance(person, str) else person.get("name", "")
                    for person in author
                    if isinstance(person, (str, dict))
                ]
                if any(names) and not metadata.get("author"):
                    metadata["author"] = ", ".join(
                        name for name in names if name
                    )

            publisher = item.get("publisher", "")
            if isinstance(publisher, dict):
                publisher = publisher.get("name", "")
            if (
                isinstance(publisher, str)
                and publisher
                and not metadata.get("publisher")
            ):
                metadata["publisher"] = publisher

            article_url = item.get("mainEntityOfPage", "")
            if isinstance(article_url, dict):
                article_url = article_url.get("url") or article_url.get("@id", "")
            if (
                isinstance(article_url, str)
                and article_url
                and not metadata.get("canonical_url")
            ):
                metadata["canonical_url"] = article_url
            body = item.get("articleBody", "")
            if isinstance(body, str):
                article_body = article_body or body

            image = item.get("image", "")
            if isinstance(image, dict):
                image = image.get("url", "")
            elif isinstance(image, list):
                image = image[0] if image else ""
                if isinstance(image, dict):
                    image = image.get("url", "")
            if isinstance(image, str) and image and not metadata.get("image"):
                metadata["image"] = image
            break
        if metadata.get("title"):
            break

    if article_body:
        metadata["article_body"] = article_body
    return metadata


def _article_container(soup):
    for selector in (
        '[itemprop="articleBody"]',
        ".article-body",
        ".article-content",
        ".post-content",
        ".entry-content",
        "article",
        "main"
    ):
        container = soup.select_one(selector)
        if container:
            return container
    return None


def _extract_body(container):
    for element in container.select(
        "script, style, noscript, nav, aside, footer, header, form, "
        "button, iframe, svg, [role='navigation'], [role='complementary']"
    ):
        element.decompose()
    for element in container.find_all(True):
        attributes = element.attrs or {}
        identity = " ".join(
            [str(attributes.get("id", ""))]
            + (
                attributes.get("class", [])
                if isinstance(attributes.get("class"), list)
                else [str(attributes.get("class", ""))]
            )
        )
        if _NON_CONTENT_SELECTOR.search(identity):
            element.decompose()

    blocks = []
    for element in container.find_all(["h2", "h3", "p", "blockquote", "li"]):
        text = element.get_text(" ", strip=True)
        if text and (not blocks or blocks[-1] != text):
            blocks.append(text)
    return "\n\n".join(blocks)


def extract_article_content(url):
    """Extract permitted article text and metadata; return None for fallbacks."""

    headers = {"User-Agent": USER_AGENT}

    try:
        response, reason, response_url = _fetch_permitted_article(
            url,
            headers
        )
        if response is None:
            print(f"[WARNING] Article extraction skipped: {reason}")
            return None

        content_type = getattr(response, "headers", {}).get(
            "Content-Type", ""
        )
        if content_type and "html" not in content_type.casefold():
            print(
                "[WARNING] Article extraction skipped: response is not HTML."
            )
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        structured = _jsonld_article_metadata(soup)
        page_title = soup.title.get_text(" ", strip=True) if soup.title else ""

        title = (
            structured.get("title")
            or _meta_content(
                soup,
                {"property": "og:title"},
                {"name": "twitter:title"}
            )
            or page_title
        )
        description = (
            structured.get("description")
            or _meta_content(
                soup,
                {"name": "description"},
                {"property": "og:description"}
            )
        )
        author = (
            structured.get("author")
            or _meta_content(
                soup,
                {"name": "author"},
                {"property": "article:author"}
            )
        )
        published = (
            structured.get("published")
            or _meta_content(
                soup,
                {"property": "article:published_time"},
                {"name": "date"},
                {"name": "pubdate"}
            )
        )
        publisher = (
            structured.get("publisher")
            or _meta_content(soup, {"property": "og:site_name"})
        )
        canonical_tag = soup.find("link", rel=lambda value: (
            value and "canonical" in value
        ))
        canonical_url = (
            structured.get("canonical_url")
            or (canonical_tag.get("href", "").strip() if canonical_tag else "")
            or response_url
        )
        canonical_url = urljoin(response_url, canonical_url)
        canonical_parts = urlsplit(canonical_url)
        if (
            canonical_parts.scheme not in ("http", "https")
            or not canonical_parts.netloc
        ):
            canonical_url = response_url

        container = _article_container(soup)
        full_text = (
            _extract_body(container) if container is not None else ""
        ) or structured.get("article_body", "").strip()
        if not full_text:
            print(
                "[WARNING] Article body is unavailable; using source "
                "summary fallback."
            )
            return None

        return {
            "page_title": page_title,
            "title": title,
            "description": description,
            "author": author,
            "publisher": publisher,
            "published": published,
            "modified": structured.get("modified", ""),
            "image": structured.get("image", ""),
            "url": canonical_url,
            "canonical_url": canonical_url,
            "full_text": full_text
        }

    except requests.exceptions.RequestException as error:
        print(f"[WARNING] Article extraction failed: {error}")
        return None
