"""
Official Product Hunt GraphQL API Collector.

Collects recent AI product launches using the official Product Hunt GraphQL API:
https://api.producthunt.com/v2/api/graphql

Authentication:
Requires an official OAuth 2.0 Developer Token provided via the environment variable
PRODUCT_HUNT_DEVELOPER_TOKEN (or PRODUCTHUNT_TOKEN).

If the token is missing, the collector returns a clear FAILED CollectionResult with
step-by-step instructions for obtaining developer credentials.
Never bypasses authentication or scrapes authenticated private endpoints.
"""

import os
from datetime import datetime, timezone

import requests

from .collection_result import CollectionResult


PRODUCTHUNT_GRAPHQL_URL = "https://api.producthunt.com/v2/api/graphql"
SETUP_INSTRUCTIONS = (
    "Missing PRODUCT_HUNT_DEVELOPER_TOKEN in environment. Setup instructions:\n"
    "  1. Go to https://www.producthunt.com/v2/oauth/applications and create an app.\n"
    "  2. Click 'Add Token' / 'Create Developer Token' to generate a permanent token.\n"
    "  3. Add 'PRODUCT_HUNT_DEVELOPER_TOKEN=<your_token>' to your .env file."
)

GRAPHQL_QUERY = """
query GetAIPosts($first: Int!) {
  posts(order: NEWEST, first: $first) {
    edges {
      node {
        id
        name
        tagline
        description
        url
        createdAt
        votesCount
        topics {
          edges {
            node {
              name
              slug
            }
          }
        }
      }
    }
  }
}
"""


def collect_producthunt(
    max_articles=20,
    api_url=PRODUCTHUNT_GRAPHQL_URL,
    developer_token=None,
):
    """
    Collect recent product launches from the official Product Hunt GraphQL API.

    Returns a CollectionResult. If no Developer Token is configured, returns
    CollectionResult.failed with configuration instructions.
    """
    print("\nCollecting from Product Hunt (Official GraphQL API)...")

    token = (
        developer_token
        or os.getenv("PRODUCT_HUNT_DEVELOPER_TOKEN")
        or os.getenv("PRODUCTHUNT_TOKEN")
    )

    if not token or not token.strip():
        print("[WARNING] Product Hunt: Missing official Developer Token.")
        return CollectionResult.failed("Product Hunt", SETUP_INSTRUCTIONS)

    limit = max(1, min(50, max_articles))

    try:
        response = requests.post(
            api_url,
            json={
                "query": GRAPHQL_QUERY,
                "variables": {"first": limit},
            },
            headers={
                "Authorization": f"Bearer {token.strip()}",
                "Content-Type": "application/json",
                "User-Agent": "AI-Daily-Intelligence-Agent/1.0",
            },
            timeout=(10, 30),
        )

        if response.status_code == 401:
            return CollectionResult.failed(
                "Product Hunt",
                "HTTP 401 Unauthorized: Invalid or expired PRODUCT_HUNT_DEVELOPER_TOKEN.",
            )

        if response.status_code == 429:
            return CollectionResult.failed(
                "Product Hunt",
                "HTTP 429 Rate Limit Exceeded: Product Hunt API quota reached.",
            )

        response.raise_for_status()

        data = response.json()
        if "errors" in data and not data.get("data"):
            error_msg = "; ".join(e.get("message", "Error") for e in data["errors"])
            return CollectionResult.failed(
                "Product Hunt",
                f"GraphQL query error: {error_msg}",
            )

        posts_data = (data.get("data") or {}).get("posts", {}).get("edges", [])

        articles = []
        for edge in posts_data:
            node = edge.get("node", {})
            post_id = node.get("id")
            name = (node.get("name") or "").strip()
            tagline = (node.get("tagline") or "").strip()
            description = (node.get("description") or "").strip()
            url = node.get("url") or f"https://www.producthunt.com/posts/{post_id}"
            created_at = node.get("createdAt") or datetime.now(timezone.utc).isoformat()
            votes = node.get("votesCount", 0)

            if not name:
                continue

            # Check topics if available
            topic_edges = (node.get("topics") or {}).get("edges", [])
            topics = [
                t.get("node", {}).get("name", "")
                for t in topic_edges
                if t.get("node", {}).get("name")
            ]

            summary = description or tagline or name
            full_title = f"{name}: {tagline}" if tagline else name

            article = {
                "source": "Product Hunt",
                "publisher": "Product Hunt",
                "source_type": "product_launch",
                "content_type": "launch",
                "title": full_title,
                "url": url,
                "published": created_at,
                "summary": summary,
                "content": description or tagline,
                "authors": [],
                "metadata": {
                    "product_id": post_id,
                    "votes_count": votes,
                    "topics": topics,
                    "tagline": tagline,
                },
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "extraction_status": "not_attempted",
                "content_source_type": "source_summary",
            }
            articles.append(article)

        result = CollectionResult.from_articles("Product Hunt", articles)

    except requests.exceptions.RequestException as error:
        result = CollectionResult.failed(
            "Product Hunt",
            f"Network or HTTP error communicating with Product Hunt API: {error}",
        )
    except Exception as error:
        result = CollectionResult.failed("Product Hunt", str(error))

    if result.status.value == "FAILED":
        print(f"[ERROR] Product Hunt: {result.error}")
    else:
        print(
            f"[{result.status.value}] Product Hunt: "
            f"collected {len(result.articles)} product launches"
        )

    return result

