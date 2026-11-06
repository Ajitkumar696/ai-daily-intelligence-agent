
import requests
import feedparser

from .collection_result import CollectionResult
from ..processing.normalize import normalize_article


def collect_rss(source, feed_url, max_articles=None):
    """
    Collect articles from an RSS feed
    and convert them into our common format.
    """

    print(f"\nCollecting from {source}...")

    try:
        # Step 1: Download the RSS feed
        response = requests.get(
            feed_url,
            timeout=(10, 30),
            headers={"User-Agent": "Mozilla/5.0"}
        )

        response.raise_for_status()

        # Step 2: Parse the RSS/XML response
        feed = feedparser.parse(response.content)

        if feed.bozo:
            raise ValueError(
                f"Unable to parse RSS feed: {feed.bozo_exception}"
            )

        print("Feed title:", feed.feed.get("title", "Unknown"))
        print("Entries found:", len(feed.entries))

        # Step 3: Select entries to process
        entries = feed.entries

        if max_articles is not None:
            entries = entries[:max_articles]

        # Step 4: Normalize every entry
        articles = []
        for entry in entries:

            article = normalize_article(source, entry)

            if article is not None:
                articles.append(article)

        result = CollectionResult.from_articles(source, articles)

    except requests.exceptions.RequestException as error:
        result = CollectionResult.failed(
            source,
            f"Network or HTTP request failed: {error}"
        )

    except Exception as error:
        result = CollectionResult.failed(source, str(error))

    if result.status.value == "FAILED":
        print(f"[ERROR] {source}: {result.error}")
    else:
        print(
            f"[{result.status.value}] {source}: "
            f"collected {len(result.articles)} articles"
        )

    return result
