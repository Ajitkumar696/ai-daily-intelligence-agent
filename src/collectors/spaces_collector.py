"""
Hugging Face Spaces collector — trending spaces via public API.

Fetches the top N trending Hugging Face Spaces from the public API and
normalises each one into the common article schema using source_type "space".
No authentication is required; the endpoint is fully public.
"""

from datetime import datetime, timezone

import requests

from .collection_result import CollectionResult


SPACES_API_URL = "https://huggingface.co/api/spaces"


def collect_hf_spaces(max_articles=20, api_url=SPACES_API_URL):
    """
    Collect trending Hugging Face Spaces from the public API.

    Returns a CollectionResult.  Each article represents one Space with
    source_type='space' and content_type='space'.
    """
    print("\nCollecting from Hugging Face Spaces...")

    try:
        params = {}
        if max_articles is not None:
            params["limit"] = max_articles

        response = requests.get(
            api_url,
            params=params,
            timeout=(10, 30),
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()

        spaces = response.json()

        if not isinstance(spaces, list):
            raise ValueError("API response must be a list of spaces.")

        print(f"Spaces received: {len(spaces)}")

        articles = []
        for item in spaces:
            if not isinstance(item, dict):
                continue

            space_id = item.get("id", "")
            if not space_id:
                continue

            # The id is "owner/name" — build a friendly display title.
            parts = space_id.split("/", 1)
            title = parts[1].replace("-", " ").title() if len(parts) == 2 else space_id

            space_url = f"https://huggingface.co/spaces/{space_id}"
            created_at = item.get("createdAt", "")
            updated_at = item.get("lastModified", created_at)

            tags = item.get("tags") or []
            sdk = item.get("sdk", "")
            likes = item.get("likes", 0)
            trending_score = item.get("trendingScore", 0)

            # Build a brief summary from available metadata.
            tag_str = ", ".join(str(t) for t in tags[:6]) if tags else ""
            summary_parts = []
            if sdk:
                summary_parts.append(f"Built with {sdk}.")
            if tag_str:
                summary_parts.append(f"Tags: {tag_str}.")
            summary_parts.append(f"Likes: {likes}.")
            summary = " ".join(summary_parts)

            article = {
                "source": "Hugging Face Spaces",
                "publisher": "Hugging Face",
                "source_type": "space",
                "content_type": "space",
                "title": f"{title} ({space_id})",
                "url": space_url,
                "published": updated_at or created_at,
                "summary": summary,
                "content": "",
                "authors": [parts[0]] if parts else [],
                "metadata": {
                    "space_id": space_id,
                    "sdk": sdk,
                    "likes": likes,
                    "trending_score": trending_score,
                    "tags": tags,
                    "created_at": created_at,
                },
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "extraction_status": "not_attempted",
                "content_source_type": "source_summary",
            }

            articles.append(article)

        result = CollectionResult.from_articles("Hugging Face Spaces", articles)

    except requests.exceptions.RequestException as error:
        result = CollectionResult.failed(
            "Hugging Face Spaces",
            f"Network or HTTP request failed: {error}",
        )
    except ValueError as error:
        result = CollectionResult.failed(
            "Hugging Face Spaces",
            f"Invalid API data: {error}",
        )
    except Exception as error:
        result = CollectionResult.failed("Hugging Face Spaces", str(error))

    if result.status.value == "FAILED":
        print(f"[ERROR] Hugging Face Spaces: {result.error}")
    else:
        print(
            f"[{result.status.value}] Hugging Face Spaces: "
            f"collected {len(result.articles)} spaces"
        )

    return result

