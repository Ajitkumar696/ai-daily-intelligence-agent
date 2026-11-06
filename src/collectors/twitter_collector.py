"""
Official X / Twitter API v2 Collector.

Collects recent AI-related posts using the official Twitter API v2 endpoint:
https://api.twitter.com/2/tweets/search/recent

Authentication:
Requires an official OAuth 2.0 Bearer Token provided via the environment variable
TWITTER_BEARER_TOKEN (or X_BEARER_TOKEN).

If the token is missing, the collector returns a clear FAILED CollectionResult with
step-by-step instructions for obtaining developer credentials.
Never bypasses authentication or scrapes authenticated private endpoints.
"""

import os
from datetime import datetime, timezone

import requests

from .collection_result import CollectionResult


TWITTER_SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"
SETUP_INSTRUCTIONS = (
    "Missing TWITTER_BEARER_TOKEN in environment. Setup instructions:\n"
    "  1. Register for an official X Developer account at https://developer.x.com/\n"
    "  2. Create a Project and an App inside the developer portal.\n"
    "  3. Generate a Bearer Token (under Keys and Tokens).\n"
    "  4. Add 'TWITTER_BEARER_TOKEN=<your_token>' to your .env file."
)


def collect_twitter(
    max_articles=10,
    search_url=TWITTER_SEARCH_URL,
    query="AI OR (Artificial Intelligence) -is:retweet lang:en",
    bearer_token=None,
):
    """
    Collect recent posts from the official Twitter API v2.

    Returns a CollectionResult. If no Bearer Token is configured, returns
    CollectionResult.failed with configuration instructions.
    """
    print("\nCollecting from X / Twitter (Official API v2)...")

    token = (
        bearer_token
        or os.getenv("TWITTER_BEARER_TOKEN")
        or os.getenv("X_BEARER_TOKEN")
    )

    if not token or not token.strip():
        print("[WARNING] X / Twitter: Missing official Bearer Token.")
        return CollectionResult.failed("X / Twitter", SETUP_INSTRUCTIONS)

    # Twitter API v2 max_results must be between 10 and 100
    limit = max(10, min(100, max_articles))

    try:
        response = requests.get(
            search_url,
            params={
                "query": query,
                "max_results": limit,
                "tweet.fields": "created_at,author_id,public_metrics,lang",
            },
            headers={
                "Authorization": f"Bearer {token.strip()}",
                "User-Agent": "AI-Daily-Intelligence-Agent/1.0",
            },
            timeout=(10, 30),
        )

        if response.status_code == 401:
            return CollectionResult.failed(
                "X / Twitter",
                "HTTP 401 Unauthorized: Invalid or expired TWITTER_BEARER_TOKEN.",
            )

        if response.status_code == 403:
            return CollectionResult.failed(
                "X / Twitter",
                "HTTP 403 Forbidden: Your X developer tier does not permit "
                "search/recent access. Check your API access level.",
            )

        if response.status_code == 429:
            return CollectionResult.failed(
                "X / Twitter",
                "HTTP 429 Too Many Requests: Twitter API rate limit reached. "
                "Please retry after the rate limit window resets.",
            )

        response.raise_for_status()

        data = response.json()
        tweets = data.get("data", [])

        if not isinstance(tweets, list):
            return CollectionResult.failed(
                "X / Twitter",
                "Invalid response structure from Twitter API (missing 'data' array).",
            )

        articles = []
        for tweet in tweets:
            tweet_id = tweet.get("id")
            text = (tweet.get("text") or "").strip()
            if not tweet_id or not text:
                continue

            # Title is first line / first 80 characters
            first_line = text.splitlines()[0] if text else "Tweet"
            title = (first_line[:80] + "...") if len(first_line) > 80 else first_line
            tweet_url = f"https://x.com/i/web/status/{tweet_id}"
            created_at = tweet.get("created_at") or datetime.now(timezone.utc).isoformat()
            author_id = tweet.get("author_id", "")
            metrics = tweet.get("public_metrics", {})

            article = {
                "source": "X / Twitter",
                "publisher": "X / Twitter",
                "source_type": "social_post",
                "content_type": "social",
                "title": title,
                "url": tweet_url,
                "published": created_at,
                "summary": text,
                "content": text,
                "authors": [author_id] if author_id else [],
                "metadata": {
                    "tweet_id": tweet_id,
                    "author_id": author_id,
                    "public_metrics": metrics,
                    "lang": tweet.get("lang", "en"),
                },
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "extraction_status": "not_attempted",
                "content_source_type": "source_summary",
            }
            articles.append(article)

        result = CollectionResult.from_articles("X / Twitter", articles)

    except requests.exceptions.RequestException as error:
        result = CollectionResult.failed(
            "X / Twitter",
            f"Network or HTTP error communicating with Twitter API: {error}",
        )
    except Exception as error:
        result = CollectionResult.failed("X / Twitter", str(error))

    if result.status.value == "FAILED":
        print(f"[ERROR] X / Twitter: {result.error}")
    else:
        print(
            f"[{result.status.value}] X / Twitter: "
            f"collected {len(result.articles)} posts"
        )

    return result

