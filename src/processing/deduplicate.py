
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


TRACKING_PARAMETERS = {
    "utm_campaign",
    "utm_content",
    "utm_id",
    "utm_medium",
    "utm_source",
    "utm_term",
    "dclid",
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid"
}


def normalize_title(title):
    """
    Normalize a title for duplicate comparison.
    """
    return " ".join(title.casefold().split())


def canonicalize_url(url):
    """Normalize URL components that do not identify a distinct article."""

    if not isinstance(url, str) or not url.strip():
        return ""

    raw_url = url.strip()
    parsed = urlsplit(raw_url)

    if not parsed.netloc:
        return urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path, parsed.query, "")
        )

    hostname = parsed.hostname
    if hostname is None:
        return urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path, parsed.query, "")
        )

    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"

    user_info = ""
    if "@" in parsed.netloc:
        user_info = parsed.netloc.rsplit("@", 1)[0] + "@"

    try:
        port = parsed.port
    except ValueError:
        return urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path, parsed.query, "")
        )

    netloc = user_info + hostname.lower()
    if port is not None:
        netloc = f"{netloc}:{port}"

    query = urlencode(
        [
            (key, value)
            for key, value in parse_qsl(
                parsed.query,
                keep_blank_values=True
            )
            if key.casefold() not in TRACKING_PARAMETERS
        ]
    )

    return urlunsplit(
        (parsed.scheme.lower(), netloc, parsed.path, query, "")
    )


def deduplicate_articles(articles):
    """
    Remove confirmed URL duplicates and keep the first occurrence.

    Titles are normalized to reject blank titles, but matching titles alone
    never removes an article because separate articles can share a headline.
    """

    unique_articles = []

    seen_urls = set()
    for article in articles:

        url = canonicalize_url(article.get("url", ""))
        title = normalize_title(article.get("title", ""))

        # Skip articles without a URL or title
        if not url or not title:
            continue

        # URL equivalence confirms duplication; a title match alone does not.
        if url in seen_urls:
            continue

        seen_urls.add(url)

        unique_articles.append(article)

    removed_count = len(articles) - len(unique_articles)

    print("\nDEDUPLICATION SUMMARY")
    print("Original articles:", len(articles))
    print("Unique articles:", len(unique_articles))
    print("Duplicates/invalid articles removed:", removed_count)

    return unique_articles
